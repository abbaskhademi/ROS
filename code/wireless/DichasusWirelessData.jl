module DichasusWirelessData

using LinearAlgebra
using NPZ
using Random
using Statistics

export build_measured_wireless_arrays

function _farthest_indices(points::Matrix{Float64}, count::Int, start::Int)
    n = size(points, 1)
    1 <= count <= n || error("invalid farthest point sample size")
    selected = Vector{Int}(undef, count)
    selected[1] = mod1(start, n)
    distance2 = fill(Inf, n)
    for t in 2:count
        centre = view(points, selected[t - 1], :)
        @inbounds for index in 1:n
            candidate = sum(abs2, view(points, index, :) .- centre)
            distance2[index] = min(distance2[index], candidate)
        end
        distance2[selected[1:t - 1]] .= -Inf
        selected[t] = argmax(distance2)
    end
    return selected
end

function _normalize_columns(points::Matrix{Float64})
    result = similar(points)
    for column in axes(points, 2)
        values = view(points, :, column)
        low, high = extrema(values)
        span = max(high - low, 1.0e-12)
        result[:, column] .= (values .- low) ./ span
    end
    return result
end

"""Build a reactive-jamming instance from the public DICHASUS d036 data.

Measured links give the antenna-element-to-user and antenna-element-to-jammer gains.
The jammer-to-user term is computed from the recorded positions, and their
product gives the two-hop cascade gain.  The rows of `F` collect nearby
position-frequency jammer configurations into overlapping resource pools.
"""
function build_measured_wireless_arrays(
    data_path::AbstractString;
    seed::Int,
    n_tx::Int,
    n_j::Int,
    m_ue::Int,
    q_groups::Int,
    sparsity::Int,
    theta::Float64,
    stress_fraction::Float64,
    p_max::Float64,
    memberships_per_jammer::Int = 4,
    min_group_size::Int = 6,
    resource_capacity::Float64 = 0.95,
)
    data = npzread(data_path)
    power = Float64.(data["channel_power"])
    positions = Float64.(data["position_m"])
    times = vec(Float64.(data["time_s"]))
    record_count, antenna_count, subband_count = size(power)

    1 <= n_tx <= antenna_count || error("n_tx exceeds the measured antenna count")
    1 <= m_ue <= record_count || error("m_ue exceeds the measured record count")
    n_j + m_ue <= record_count ||
        error("the requested users and jammers exceed the measured positions")
    1 <= sparsity <= n_j || error("invalid sparsity")
    q_groups >= memberships_per_jammer + 1 || error("too few resource pools")

    rng = MersenneTwister(seed)

    # Choose antennas with the largest median measured SNR.  This avoids links
    # at the receiver noise floor while retaining all 47 antennas when n_tx=47.
    antenna_scores = [median(vec(power[:, antenna, :])) for antenna in 1:antenna_count]
    antenna_indices = sortperm(antenna_scores, rev = true)[1:n_tx]

    # Spread user locations along the measured trajectory.
    user_points = _normalize_columns(positions[:, 1:2])
    user_indices = _farthest_indices(user_points, m_ue, 19 * seed + 1)
    h = zeros(m_ue, n_tx)
    for k in 1:m_ue, i in 1:n_tx
        h[k, i] = mean(view(power, user_indices[k], antenna_indices[i], :))
    end
    for k in 1:m_ue
        row_scale = maximum(view(h, k, :))
        row_scale > 0.0 || error("zero measured channel row")
        h[k, :] ./= row_scale
    end

    # Potential jammers occupy distinct measured positions that are not used as
    # user locations.  A frequency block is assigned to each jammer.
    available_records = setdiff(collect(1:record_count), user_indices)
    jammer_records = available_records[randperm(rng, length(available_records))[1:n_j]]
    jammer_subbands = rand(rng, 1:subband_count, n_j)

    # DICHASUS gives the measured antenna-element-to-jammer link. The remaining
    # jammer-to-user hop uses the free-space basic transmission-loss expression
    # in ITU-R P.1411-13 at the measurement carrier frequency. A
    # one-metre reference distance avoids applying a far-field expression to
    # nearly coincident recorded positions.
    gain_bj = zeros(n_j, n_tx)
    for j in 1:n_j, i in 1:n_tx
        gain_bj[j, i] = power[jammer_records[j], antenna_indices[i], jammer_subbands[j]]
    end
    carrier_ghz = 1.272
    gain_ju = zeros(m_ue, n_j)
    jammer_user_distances = zeros(m_ue, n_j)
    for k in 1:m_ue, j in 1:n_j
        recorded_distance_m = norm(view(positions, user_indices[k], :) .-
                                   view(positions, jammer_records[j], :))
        jammer_user_distances[k, j] = recorded_distance_m
        distance_m = max(recorded_distance_m, 1.0)
        path_loss_db = 32.4 + 20.0 * log10(distance_m) + 20.0 * log10(carrier_ghz)
        gain_ju[k, j] = 10.0 ^ (-path_loss_db / 10.0)
    end
    raw_g = zeros(m_ue, n_j, n_tx)
    @inbounds for k in 1:m_ue, j in 1:n_j, i in 1:n_tx
        raw_g[k, j, i] = gain_ju[k, j] * gain_bj[j, i]
    end

    u_bar = ones(n_j)
    useful_equal_power = vec(sum(h, dims = 2))
    top_s_interference = zeros(m_ue)
    for k in 1:m_ue
        jammer_interference = [sum(raw_g[k, j, i] for i in 1:n_tx) for j in 1:n_j]
        top_s_interference[k] = sum(partialsort(jammer_interference, 1:sparsity, rev = true))
    end
    scale_candidates = useful_equal_power ./ (theta .* top_s_interference .+ 1.0e-14)
    jammer_scale = stress_fraction * minimum(scale_candidates)
    g = jammer_scale .* raw_g

    # Measured jammer positions and assigned frequency blocks define overlapping
    # resource pools.
    jammer_coordinates = zeros(n_j, 3)
    for j in 1:n_j
        jammer_coordinates[j, 1:2] .= positions[jammer_records[j], 1:2]
        jammer_coordinates[j, 3] = jammer_subbands[j]
    end
    normalized_coordinates = _normalize_columns(jammer_coordinates)
    centre_indices = _farthest_indices(normalized_coordinates, q_groups, 31 * seed + 1)
    centres = normalized_coordinates[centre_indices, :]
    F = zeros(q_groups, n_j)
    for j in 1:n_j
        distances = [norm(view(normalized_coordinates, j, :) .- view(centres, r, :)) for r in 1:q_groups]
        groups = partialsortperm(distances, 1:memberships_per_jammer)
        local_scale = maximum(distances[groups]) + 1.0e-12
        F[groups, j] .= 0.75 .+ 0.50 .* distances[groups] ./ local_scale
    end
    for r in 1:q_groups
        present = findall(>(0.0), view(F, r, :))
        if length(present) < min_group_size
            distances = [norm(view(normalized_coordinates, j, :) .- view(centres, r, :)) for j in 1:n_j]
            for j in partialsortperm(distances, 1:min_group_size)
                F[r, j] = F[r, j] > 0.0 ? F[r, j] : 1.0
            end
        end
    end
    d = fill(resource_capacity, q_groups)

    sigma2 = fill(0.003 * median(useful_equal_power), m_ue)
    p_bar = fill(p_max, n_tx)
    group_sizes = vec(sum(F .> 0.0, dims = 2))
    memberships = vec(sum(F .> 0.0, dims = 1))
    diagnostics = Dict(
        "dichasus_measurement" => 1.0,
        "record_count" => Float64(record_count),
        "measured_antenna_count" => Float64(antenna_count),
        "subband_count" => Float64(subband_count),
        "jammer_scale" => Float64(jammer_scale),
        "mean_group_size" => Float64(mean(group_sizes)),
        "min_group_size" => Float64(minimum(group_sizes)),
        "max_group_size" => Float64(maximum(group_sizes)),
        "mean_jammer_memberships" => Float64(mean(memberships)),
        "resource_capacity" => Float64(resource_capacity),
        "measurement_duration_s" => Float64(maximum(times) - minimum(times)),
        "minimum_recorded_jammer_user_distance_m" => Float64(minimum(jammer_user_distances)),
        "maximum_recorded_jammer_user_distance_m" => Float64(maximum(jammer_user_distances)),
        "fraction_jammer_user_distances_below_1m" =>
            Float64(mean(jammer_user_distances .< 1.0)),
    )

    return (
        h = h,
        g = g,
        sigma2 = sigma2,
        u_bar = u_bar,
        F = F,
        d = d,
        p_bar = p_bar,
        diagnostics = diagnostics,
        user_indices = user_indices,
        antenna_indices = antenna_indices,
        jammer_records = jammer_records,
        jammer_subbands = jammer_subbands,
    )
end

end
