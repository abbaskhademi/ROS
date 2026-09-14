module WirelessSparseARO

using JuMP
import Mosek
import MosekTools
using MathOptInterface
const MOI = MathOptInterface
using Random
using LinearAlgebra
using Statistics

include(joinpath(@__DIR__, "DichasusWirelessData.jl"))
using .DichasusWirelessData

export AROConfig,
       AROInstance,
       AROResult,
       generate_instance,
       load_instance_arrays,
       enumerate_sparse_vertices,
       solve_exact_full,
       solve_exact_aadr,
       solve_exact_ro,
       solve_db,
       solve_ro_el_cp,
       solve_aadr_rl,
       solve_aadr_rl_el_cp,
       solve_aadr_el_cp,
       run_one_aro,
       set_optimizer!

Base.@kwdef struct AROConfig
    seed::Int = 1
    instance_design::Symbol = :small_validation
    data_path::String = joinpath(@__DIR__, "..", "..", "data", "wireless_source", "dichasus-d036-32subbands.npz")
    n_tx::Int = 5
    n_j::Int = 5
    m_ue::Int = 5
    sparsity::Int = 3
    q_groups::Int = 5
    theta::Float64 = 0.35
    p_max::Float64 = 20.0
    strong_gain::Float64 = 1.0
    weak_gain::Float64 = 0.10
    channel_perturbation::Float64 = 0.05
    jammer_fraction::Float64 = 0.50
    jammer_perturbation::Float64 = 0.03
    spillover_fraction::Float64 = 0.0
    pair_mode_cost::Float64 = 1.40
    singleton_mode_cost::Float64 = 1.0
    resource_budget::Float64 = 1.80
    recovery_cluster_size::Int = 5
    radius_m::Float64 = 1000.0
    inter_site_distance_m::Float64 = 500.0
    memberships_per_jammer::Int = 4
    min_group_size::Int = 6
    budget_multiplier_low::Float64 = 1.5
    budget_multiplier_high::Float64 = 2.4
    jammer_stress_fraction::Float64 = 0.60
    resource_capacity::Float64 = 0.95
    normalize_sinr_rows::Bool = true
    aadr_batch_size::Int = 10
    aadr_screen_with_rl::Bool = true
    violation_tol::Float64 = 1.0e-6
    max_iterations::Int = 80
    alternating_starts::Int = 30
    alternating_max_iterations::Int = 50
    mip_time_limit_s::Float64 = 120.0
    mip_rel_gap::Float64 = 1.0e-6
    method_time_limit_s::Float64 = 1000.0
end

struct AROInstance
    h::Matrix{Float64}
    g::Array{Float64,3}
    sigma2::Vector{Float64}
    u_bar::Vector{Float64}
    F::Matrix{Float64}
    d::Vector{Float64}
    p_bar::Vector{Float64}
    B::Matrix{Float64}
    mode_cost::Vector{Float64}
    theta::Float64
    sparsity::Int
    primary_ue::Vector{Int}
    diagnostics::Dict{String,Float64}
end

struct AROResult
    method::String
    objective::Float64
    wall_time_s::Float64
    status::String
    p::Vector{Float64}
    iterations::Int
    cuts::Int
    max_violation::Float64
end

struct Cut
    u::Vector{Float64}
    lam::Vector{Float64}
end

struct SolverTimeLimit <: Exception
    context::String
    status::String
end

function Base.showerror(io::IO, err::SolverTimeLimit)
    print(io, err.context, " reached a solver resource limit: ", err.status)
end

_is_solver_resource_limit(status) =
    status == MOI.TIME_LIMIT || status == MOI.ITERATION_LIMIT

_resource_limit_label(err::SolverTimeLimit) =
    occursin("ITERATION_LIMIT", err.status) ? "ITERATION_LIMIT" : "TIME_LIMIT"

const _OPTIMIZER_FACTORY = Ref{Any}(Mosek.Optimizer)

function set_optimizer!(optimizer_factory)
    _OPTIMIZER_FACTORY[] = optimizer_factory
    return nothing
end

function _new_model(; timelimit = nothing)
    model = Model(_OPTIMIZER_FACTORY[])
    set_silent(model)
    if timelimit !== nothing
        set_time_limit_sec(model, Float64(timelimit))
    end
    return model
end

"""Prefer MOSEK simplex for continuous LPs; silently do nothing for other solvers."""
function _prefer_lp_simplex!(model::JuMP.Model)
    try
        set_attribute(model, "MSK_IPAR_OPTIMIZER", Mosek.MSK_OPTIMIZER_FREE_SIMPLEX)
    catch
        # Non-MOSEK fallback (e.g., HiGHS parity) or unsupported raw attribute.
    end
    return nothing
end

"""Prefer MOSEK's interior-point optimizer for the large AADR LPs."""
function _prefer_lp_interior_point!(model::JuMP.Model)
    try
        set_attribute(model, "MSK_IPAR_OPTIMIZER", Mosek.MSK_OPTIMIZER_INTPNT)
        set_attribute(model, "MSK_IPAR_INTPNT_SOLVE_FORM", Mosek.MSK_SOLVE_DUAL)
        set_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_REL_GAP", 1.0e-6)
        set_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_PFEAS", 1.0e-6)
        set_attribute(model, "MSK_DPAR_INTPNT_CO_TOL_DFEAS", 1.0e-6)
    catch
        # Non-MOSEK fallback or unsupported raw attribute.
    end
    return nothing
end

"""Set a relative MIP gap using MOI when supported, with a MOSEK-native fallback."""
function _set_mip_rel_gap!(model::JuMP.Model, gap::Real)
    g = Float64(gap)
    try
        set_attribute(model, MOI.RelativeGapTolerance(), g)
        return nothing
    catch generic_err
        try
            # MosekTools exposes MOSEK parameters through raw string attributes.
            set_attribute(model, "MSK_DPAR_MIO_TOL_REL_GAP", g)
            return nothing
        catch raw_err
            error(
                "Could not set the relative MIP gap. Generic MOI error: " *
                sprint(showerror, generic_err) *
                "; solver-specific fallback error: " *
                sprint(showerror, raw_err),
            )
        end
    end
end

function _solve_nominal(h, sigma2, theta, p_bar)
    K, I = size(h)
    model = _new_model()
    @variable(model, 0 <= p[i = 1:I] <= p_bar[i])
    @objective(model, Min, sum(p[i] for i in 1:I))
    @constraint(
        model,
        [k = 1:K],
        sum(h[k, i] * p[i] for i in 1:I) >= theta * sigma2[k],
    )
    optimize!(model)
    termination_status(model) == MOI.OPTIMAL || error("nominal problem infeasible")
    return value.(p)
end

function _points_in_disk(rng, n::Int, radius::Float64)
    rr = radius .* sqrt.(rand(rng, n))
    theta = 2.0 * pi .* rand(rng, n)
    return hcat(rr .* cos.(theta), rr .* sin.(theta))
end

"""Return the nearest sites of a hexagonal macro-cell layout."""
function _hexagonal_sites(n::Int, inter_site_distance::Float64)
    axial = Tuple{Int,Int}[]
    layer = 0
    while length(axial) < n
        empty!(axial)
        for q in -layer:layer, r in -layer:layer
            max(abs(q), abs(r), abs(q + r)) <= layer && push!(axial, (q, r))
        end
        layer += 1
    end
    points = [
        (inter_site_distance * (q + 0.5r), inter_site_distance * (sqrt(3) / 2) * r)
        for (q, r) in axial
    ]
    sort!(points; by = point -> (hypot(point[1], point[2]), atan(point[2], point[1])))
    return hcat(first.(points[1:n]), last.(points[1:n]))
end

function _large_scale_gain(
    dist::AbstractArray,
    alpha::Float64,
    shadow_sd_db::Float64,
    rng,
)
    dd = max.(Float64.(dist), 1.0)
    shadow = shadow_sd_db .* randn(rng, size(dd))
    return (50.0 ./ dd) .^ alpha .* 10.0 .^ (shadow ./ 10.0)
end

function _recovery_system(cfg::AROConfig)
    K = cfg.m_ue
    cluster_size = cfg.recovery_cluster_size
    cluster_size >= 3 || error("recovery_cluster_size must be at least 3")
    R = 2 * K
    B = zeros(K, R)
    mode_cost = zeros(R)
    for k in 1:K
        cluster = div(k - 1, cluster_size)
        local_index = mod(k - 1, cluster_size)
        cluster_start = cluster * cluster_size + 1
        cluster_length = min(cluster_size, K - cluster * cluster_size)
        next_user = cluster_start + mod(local_index + 1, cluster_length)
        B[k, k] = 1.0
        B[next_user, k] = 1.0
        mode_cost[k] = cfg.pair_mode_cost
        B[k, K + k] = 1.0
        mode_cost[K + k] = cfg.singleton_mode_cost
    end
    return B, mode_cost
end

"""Propagation-based scalable ARO generator with overlapping resource rows.

The channel and resource construction matches the literature-grounded static RO
benchmark.  Each SINR row is optionally divided by its strongest direct-link
gain; this is an exactly equivalent row scaling that aligns the physical
feasibility tolerance with the normalized recovery deficit.
"""
function _generate_physical_overlap_instance(cfg::AROConfig)
    cfg.q_groups >= cfg.memberships_per_jammer + 1 ||
        error("q_groups must be at least memberships_per_jammer + 1")
    cfg.budget_multiplier_low > 0.0 || error("resource budgets must be positive")
    cfg.budget_multiplier_high >= cfg.budget_multiplier_low ||
        error("invalid resource-budget interval")

    rng = MersenneTwister(cfg.seed)
    I = cfg.n_tx
    J = cfg.n_j
    K = cfg.m_ue
    Q = cfg.q_groups

    bs = _hexagonal_sites(I, cfg.inter_site_distance_m)
    ue = _points_in_disk(rng, K, 0.90 * cfg.radius_m)
    jm = _points_in_disk(rng, J, 0.90 * cfg.radius_m)

    d_bu = [hypot(norm(ue[k, :] - bs[i, :]), 25.0 - 1.5) for k in 1:K, i in 1:I]
    d_bj = [hypot(norm(bs[i, :] - jm[j, :]), 25.0 - 1.5) for i in 1:I, j in 1:J]
    d_ju = [max(norm(ue[k, :] - jm[j, :]), 1.0) for k in 1:K, j in 1:J]

    h = _large_scale_gain(d_bu, 3.5, 8.0, rng)
    gain_bj = _large_scale_gain(d_bj, 2.2, 3.0, rng)
    gain_ju = _large_scale_gain(d_ju, 2.0, 3.0, rng)

    raw_g = zeros(K, J, I)
    @inbounds for k in 1:K, j in 1:J, i in 1:I
        raw_g[k, j, i] = gain_ju[k, j] * gain_bj[i, j]
    end

    u_bar = 0.8 .+ 0.4 .* rand(rng, J)
    raw_c_equal_power = zeros(K, J)
    for k in 1:K, j in 1:J
        raw_c_equal_power[k, j] = sum(raw_g[k, j, i] for i in 1:I)
    end
    top_s_box = zeros(K)
    for k in 1:K
        vals = raw_c_equal_power[k, :] .* u_bar
        top_s_box[k] = sum(sort(collect(vals), rev = true)[1:cfg.sparsity])
    end
    useful_equal_power = vec(sum(h, dims = 2))
    scale_candidates = useful_equal_power ./ (cfg.theta .* top_s_box .+ 1.0e-14)
    jammer_scale = cfg.jammer_stress_fraction * minimum(scale_candidates)
    g = jammer_scale .* raw_g
    sigma2 = fill(0.003 * median(useful_equal_power), K)

    pool_centres = _points_in_disk(rng, Q, cfg.radius_m)
    F = zeros(Q, J)
    for j in 1:J
        distances = [norm(jm[j, :] - pool_centres[r, :]) for r in 1:Q]
        group_rows = partialsortperm(distances, 1:cfg.memberships_per_jammer)
        local_scale = maximum(distances[group_rows]) + 1.0e-9
        F[group_rows, j] .= 0.6 .+ 0.8 .* distances[group_rows] ./ local_scale
    end
    for r in 1:Q
        present = findall(>(0.0), view(F, r, :))
        if length(present) < cfg.min_group_size
            candidates = setdiff(collect(1:J), present)
            n_add = cfg.min_group_size - length(present)
            add = candidates[randperm(rng, length(candidates))[1:n_add]]
            F[r, add] .= 0.6 .+ 0.8 .* rand(rng, n_add)
        end
    end
    d = zeros(Q)
    for r in 1:Q
        idx = findall(>(0.0), view(F, r, :))
        full_use = [F[r, j] * u_bar[j] for j in idx]
        multiplier = cfg.budget_multiplier_low +
                     (cfg.budget_multiplier_high - cfg.budget_multiplier_low) * rand(rng)
        d[r] = multiplier * median(full_use)
    end

    pre_normalization_href = vec(maximum(h, dims = 2))
    if cfg.normalize_sinr_rows
        for k in 1:K
            row_scale = pre_normalization_href[k]
            row_scale > 0.0 || error("zero direct-link row in physical generator")
            h[k, :] ./= row_scale
            g[k, :, :] ./= row_scale
            sigma2[k] /= row_scale
        end
    end

    p_bar = fill(cfg.p_max, I)
    p_nom = _solve_nominal(h, sigma2, cfg.theta, p_bar)
    B, mode_cost = _recovery_system(cfg)
    primary = [
        argmax([sum(g[k, j, i] for i in 1:I) for k in 1:K])
        for j in 1:J
    ]

    group_sizes = vec(sum(F .> 0.0, dims = 2))
    jammer_memberships = vec(sum(F .> 0.0, dims = 1))
    diagnostics = Dict(
        "generator_design_physical_overlap" => 1.0,
        "nominal_objective" => Float64(sum(p_nom)),
        "jammer_scale" => Float64(jammer_scale),
        "mean_group_size" => Float64(mean(group_sizes)),
        "min_group_size" => Float64(minimum(group_sizes)),
        "max_group_size" => Float64(maximum(group_sizes)),
        "mean_jammer_resource_memberships" => Float64(mean(jammer_memberships)),
        "mean_jammer_fraction" => Float64(cfg.jammer_stress_fraction),
        "inter_site_distance_m" => Float64(cfg.inter_site_distance_m),
        "n_modes" => Float64(size(B, 2)),
        "recovery_cluster_size" => Float64(cfg.recovery_cluster_size),
        "n_recovery_clusters" => Float64(cld(K, cfg.recovery_cluster_size)),
        "min_pre_normalization_href" => Float64(minimum(pre_normalization_href)),
        "max_pre_normalization_href" => Float64(maximum(pre_normalization_href)),
    )

    return AROInstance(
        h,
        g,
        sigma2,
        u_bar,
        F,
        d,
        p_bar,
        B,
        mode_cost,
        cfg.theta,
        cfg.sparsity,
        primary,
        diagnostics,
    )
end

function _generate_dichasus_instance(cfg::AROConfig)
    arrays = build_measured_wireless_arrays(
        cfg.data_path;
        seed = cfg.seed,
        n_tx = cfg.n_tx,
        n_j = cfg.n_j,
        m_ue = cfg.m_ue,
        q_groups = cfg.q_groups,
        sparsity = cfg.sparsity,
        theta = cfg.theta,
        stress_fraction = cfg.jammer_stress_fraction,
        p_max = cfg.p_max,
        memberships_per_jammer = cfg.memberships_per_jammer,
        min_group_size = cfg.min_group_size,
        resource_capacity = cfg.resource_capacity,
    )
    B, mode_cost = _recovery_system(cfg)
    p_nom = _solve_nominal(arrays.h, arrays.sigma2, cfg.theta, arrays.p_bar)
    primary = [
        argmax([sum(arrays.g[k, j, i] for i in 1:cfg.n_tx) for k in 1:cfg.m_ue])
        for j in 1:cfg.n_j
    ]
    diagnostics = copy(arrays.diagnostics)
    diagnostics["nominal_objective"] = Float64(sum(p_nom))
    diagnostics["n_modes"] = Float64(size(B, 2))
    diagnostics["recovery_cluster_size"] = Float64(cfg.recovery_cluster_size)
    diagnostics["n_recovery_clusters"] = Float64(cld(cfg.m_ue, cfg.recovery_cluster_size))
    return AROInstance(
        arrays.h,
        arrays.g,
        arrays.sigma2,
        arrays.u_bar,
        arrays.F,
        arrays.d,
        arrays.p_bar,
        B,
        mode_cost,
        cfg.theta,
        cfg.sparsity,
        primary,
        diagnostics,
    )
end

function generate_instance(cfg::AROConfig)
    cfg.n_tx >= 2 || error("n_tx must be at least 2")
    cfg.m_ue >= 3 || error("m_ue must be at least 3")
    cfg.n_j >= 1 || error("n_j must be positive")
    1 <= cfg.sparsity <= cfg.n_j || error("invalid sparsity")

    if cfg.instance_design == :physical_overlap
        return _generate_physical_overlap_instance(cfg)
    elseif cfg.instance_design == :dichasus
        return _generate_dichasus_instance(cfg)
    elseif cfg.instance_design != :small_validation
        error("instance_design must be :small_validation, :physical_overlap, or :dichasus")
    end

    rng = MersenneTwister(cfg.seed)
    I = cfg.n_tx
    K = cfg.m_ue
    J = cfg.n_j
    cluster_size = cfg.recovery_cluster_size
    cluster_size >= 3 || error("recovery_cluster_size must be at least 3")
    n_clusters = cld(K, cluster_size)

    # Scalable instances repeat the same local cell-edge structure used by the
    # five-user validation.  Each recovery cluster has at most five users arranged
    # on a local odd cycle. Different clusters are shifted over the global
    # transmitter set, so clusters can share transmitting elements while retaining the local two-neighbor
    # channel pattern.
    h = zeros(K, I)
    for k in 1:K
        cluster = div(k - 1, cluster_size)
        local_index = mod(k - 1, cluster_size)
        cluster_start = cluster * cluster_size + 1
        cluster_length = min(cluster_size, K - cluster * cluster_size)
        offset = floor(Int, cluster * I / max(n_clusters, 1))
        b0 = mod(offset + local_index, I) + 1
        b1 = mod(offset + mod(local_index + 1, cluster_length), I) + 1
        for i in 1:I
            base_gain = (i == b0 || i == b1) ? cfg.strong_gain : cfg.weak_gain
            perturbation = 1.0 - cfg.channel_perturbation +
                           2.0 * cfg.channel_perturbation * rand(rng)
            h[k, i] = base_gain * perturbation
        end
    end

    sigma2 = fill(1.0 / cfg.theta, K)
    p_bar = fill(cfg.p_max, I)
    p_nom = _solve_nominal(h, sigma2, cfg.theta, p_bar)

    primary = [mod(j - 1, K) + 1 for j in 1:J]
    g = zeros(K, J, I)
    jammer_strength = cfg.jammer_fraction .* (
        1.0 - cfg.jammer_perturbation .+
        2.0 * cfg.jammer_perturbation .* rand(rng, J)
    )

    for j in 1:J
        k = primary[j]
        g[k, j, :] .= jammer_strength[j] .* h[k, :] ./ cfg.theta
        if cfg.spillover_fraction > 0.0
            kp = mod1(k + 1, K)
            g[kp, j, :] .+=
                cfg.spillover_fraction * jammer_strength[j] .* h[kp, :] ./ cfg.theta
        end
    end

    u_bar = ones(J)
    rows = Vector{Vector{Float64}}()
    rhs = Float64[]

    if J > K
        for k in 1:K
            row = zeros(J)
            idx = findall(==(k), primary)
            row[idx] .= 1.0
            if any(row .> 0.0)
                push!(rows, row)
                push!(rhs, 1.0)
            end
        end
    end

    q_target = max(cfg.q_groups, 1)
    for r in 1:q_target
        row = zeros(J)
        cluster = mod(div(r - 1, cluster_size), n_clusters)
        local_index = mod(r - 1, cluster_size)
        cluster_start = cluster * cluster_size + 1
        cluster_length = min(cluster_size, K - cluster * cluster_size)
        hotspots = [
            cluster_start + mod(local_index + shift, cluster_length)
            for shift in 0:2
        ]
        weights = [1.0, 0.8, 0.7]
        for (hotspot, weight) in zip(hotspots, weights)
            idx = findall(==(hotspot), primary)
            if !isempty(idx)
                row[idx] .= weight
            end
        end
        if any(row .> 0.0)
            push!(rows, row)
            push!(rhs, cfg.resource_budget)
        end
    end

    F = isempty(rows) ? zeros(0, J) : reduce(vcat, [permutedims(row) for row in rows])
    d = collect(rhs)

    R = 2 * K
    B = zeros(K, R)
    mode_cost = zeros(R)
    for k in 1:K
        cluster = div(k - 1, cluster_size)
        local_index = mod(k - 1, cluster_size)
        cluster_start = cluster * cluster_size + 1
        cluster_length = min(cluster_size, K - cluster * cluster_size)
        next_user = cluster_start + mod(local_index + 1, cluster_length)
        B[k, k] = 1.0
        B[next_user, k] = 1.0
        mode_cost[k] = cfg.pair_mode_cost
        B[k, K + k] = 1.0
        mode_cost[K + k] = cfg.singleton_mode_cost
    end

    memberships = size(F, 1) == 0 ? zeros(J) : vec(sum(F .> 0.0, dims = 1))
    diagnostics = Dict(
        "nominal_objective" => Float64(sum(p_nom)),
        "mean_strong_to_weak_gain_ratio" => Float64(cfg.strong_gain / cfg.weak_gain),
        "mean_jammer_fraction" => Float64(mean(jammer_strength)),
        "mean_jammer_resource_memberships" => Float64(mean(memberships)),
        "n_modes" => Float64(R),
        "recovery_cluster_size" => Float64(cluster_size),
        "n_recovery_clusters" => Float64(n_clusters),
    )

    return AROInstance(
        h,
        g,
        sigma2,
        u_bar,
        F,
        d,
        p_bar,
        B,
        mode_cost,
        cfg.theta,
        cfg.sparsity,
        primary,
        diagnostics,
    )
end

function load_instance_arrays(
    h,
    g,
    sigma2,
    u_bar,
    F,
    d,
    p_bar,
    B,
    mode_cost,
    theta,
    sparsity,
    primary_ue,
)
    primary = Int.(vec(primary_ue)) .+ 1
    return AROInstance(
        Float64.(h),
        Float64.(g),
        vec(Float64.(sigma2)),
        vec(Float64.(u_bar)),
        Float64.(F),
        vec(Float64.(d)),
        vec(Float64.(p_bar)),
        Float64.(B),
        vec(Float64.(mode_cost)),
        Float64(theta),
        Int(sparsity),
        primary,
        Dict{String,Float64}(),
    )
end

function hard_threshold(u, s::Int)
    v = max.(Float64.(u), 0.0)
    if s >= length(v)
        return copy(v)
    end
    idx = sortperm(v, rev = true)[1:s]
    out = zeros(length(v))
    out[idx] = v[idx]
    return out
end



"""Scale a nonnegative scenario downward until it satisfies `F * u <= d`.

The wireless uncertainty sets have `F >= 0`, so common downward scaling
preserves the box constraints, support, and every already feasible resource
row.  `prevfloat` gives the corrected point a one-ulp interior margin instead
of leaving it exactly on a numerically rounded boundary.
"""
function _scale_to_polyhedron!(
    use::Vector{Float64},
    F::Matrix{Float64},
    d::Vector{Float64},
)
    size(F, 1) == 0 && return 1.0, 0.0

    loads = F * use
    pre_scale_poly_violation = max(0.0, maximum(loads - d))
    pre_scale_poly_violation == 0.0 && return 1.0, 0.0

    ratios = Float64[]
    for r in eachindex(d)
        if loads[r] > d[r] && loads[r] > 0.0
            push!(ratios, d[r] / loads[r])
        end
    end
    isempty(ratios) && return 1.0, pre_scale_poly_violation

    scale_factor = clamp(minimum(ratios), 0.0, 1.0)
    if 0.0 < scale_factor < 1.0
        scale_factor = prevfloat(scale_factor)
    end
    use .*= scale_factor
    return scale_factor, pre_scale_poly_violation
end

"""Canonicalize an SP/RL point into an exactly sparse feasible scenario."""
function _canonicalize_sparse_lp_solution(
    raw::Vector{Float64},
    u_bar::Vector{Float64},
    F::Matrix{Float64},
    d::Vector{Float64},
    sparsity::Int,
)
    use = hard_threshold(raw, sparsity)
    use .= min.(use, u_bar)
    scale_factor, pre_scale_poly_violation = _scale_to_polyhedron!(use, F, d)
    return use, (
        inactive_spill = 0.0,
        feasibility_scale = scale_factor,
        pre_scale_poly_violation = pre_scale_poly_violation,
        active_binary_count = 0,
    )
end

"""Optimize the uncertainty magnitudes on a selected support."""
function _optimize_fixed_support(
    c::Vector{Float64},
    support::Vector{Int},
    u_bar::Vector{Float64},
    F::Matrix{Float64},
    d::Vector{Float64},
)
    isempty(support) && return zeros(length(c))
    model = _new_model()
    _prefer_lp_simplex!(model)
    @variable(model, 0 <= u[j = 1:length(c)] <= u_bar[j])
    outside = setdiff(eachindex(c), support)
    @constraint(model, [j in outside], u[j] == 0)
    if size(F, 1) > 0
        @constraint(model, [r = 1:size(F, 1)], sum(F[r, j] * u[j] for j in eachindex(c)) <= d[r])
    end
    @objective(model, Max, sum(c[j] * u[j] for j in eachindex(c)))
    optimize!(model)
    termination_status(model) == MOI.OPTIMAL || error("Fixed-support separation failed")
    return Float64.(value.(u))
end

"""Canonicalize a MIP incumbent into an exactly sparse feasible scenario.

MIP solvers may return tiny positive u[j] values when the linked binary z[j]
is numerically zero.  We use z to define the support, clamp only downward to
the box, and (only if needed) apply an O(feasibility-tolerance) common scaling
to remove a residual F*u <= d violation.  Because F >= 0, every correction is
downward and therefore preserves feasibility and sparsity.
"""
function _canonicalize_el_solution(
    raw::Vector{Float64},
    zval::Vector{Float64},
    u_bar::Vector{Float64},
    F::Matrix{Float64},
    d::Vector{Float64},
    sparsity::Int,
)
    active = zval .>= 0.5
    active_count = count(identity, active)
    active_count <= sparsity || error(
        "EL binary incumbent violates sparsity after rounding: " *
        "active=$active_count, s=$sparsity, z=$(zval)",
    )

    use = zeros(length(raw))
    for j in eachindex(raw)
        if active[j]
            use[j] = clamp(raw[j], 0.0, u_bar[j])
        end
    end

    inactive_values = raw[.!active]
    inactive_spill = isempty(inactive_values) ? 0.0 : maximum(abs.(inactive_values))
    scale_factor, pre_scale_poly_violation = _scale_to_polyhedron!(use, F, d)

    return use, (
        inactive_spill = inactive_spill,
        feasibility_scale = scale_factor,
        pre_scale_poly_violation = pre_scale_poly_violation,
        active_binary_count = active_count,
    )
end

function solve_separator(
    c::Vector{Float64},
    ins::AROInstance,
    method::String,
    cfg::AROConfig;
    require_exact_el::Bool = true,
)
    J = length(c)
    scale = max(maximum(abs.(c)), 1.0e-12)
    c_scaled = c ./ scale
    method_name = uppercase(method)
    t0 = time()

    raw = zeros(J)
    use = zeros(J)
    cleanup = (
        inactive_spill = 0.0,
        feasibility_scale = 1.0,
        pre_scale_poly_violation = 0.0,
        active_binary_count = 0,
    )

    if method_name == "SP" || method_name == "RL"
        model = _new_model()
        _prefer_lp_simplex!(model)
        @variable(model, 0 <= u[j = 1:J] <= ins.u_bar[j])
        if size(ins.F, 1) > 0
            @constraint(
                model,
                [r = 1:size(ins.F, 1)],
                sum(ins.F[r, j] * u[j] for j in 1:J) <= ins.d[r],
            )
        end
        if method_name == "RL"
            @constraint(model, sum(u[j] / ins.u_bar[j] for j in 1:J) <= ins.sparsity)
        end
        @objective(model, Max, sum(c_scaled[j] * u[j] for j in 1:J))
        optimize!(model)
        status = termination_status(model)
        if status != MOI.OPTIMAL
            if !(status == MOI.SLOW_PROGRESS && has_values(model))
                error("$method_name separator failed: $status")
            end
        end
        raw = Float64.(value.(u))
        use, cleanup = _canonicalize_sparse_lp_solution(
            raw, ins.u_bar, ins.F, ins.d, ins.sparsity,
        )
        support = findall(>(1.0e-9), use)
        use = _optimize_fixed_support(c_scaled, support, ins.u_bar, ins.F, ins.d)
    elseif method_name == "EL"
        model = _new_model(timelimit = cfg.mip_time_limit_s)
        _set_mip_rel_gap!(model, cfg.mip_rel_gap)
        @variable(model, 0 <= u[j = 1:J] <= ins.u_bar[j])
        @variable(model, z[1:J], Bin)
        @constraint(model, [j = 1:J], u[j] <= ins.u_bar[j] * z[j])
        @constraint(model, sum(z[j] for j in 1:J) <= ins.sparsity)
        if size(ins.F, 1) > 0
            @constraint(
                model,
                [r = 1:size(ins.F, 1)],
                sum(ins.F[r, j] * u[j] for j in 1:J) <= ins.d[r],
            )
        end
        @objective(model, Max, sum(c_scaled[j] * u[j] for j in 1:J))
        optimize!(model)
        status = termination_status(model)
        if require_exact_el && status != MOI.OPTIMAL
            if _is_solver_resource_limit(status)
                throw(SolverTimeLimit("EL separator", string(status)))
            end
            error("EL separator not globally solved: $status")
        end
        has_values(model) || error("EL separator returned no incumbent")
        raw = Float64.(value.(u))
        zval = Float64.(value.(z))
        use, cleanup = _canonicalize_el_solution(
            raw, zval, ins.u_bar, ins.F, ins.d, ins.sparsity,
        )
    else
        error("method must be SP, RL, or EL")
    end

    poly_violation = size(ins.F, 1) > 0 ?
        max(0.0, maximum(ins.F * use - ins.d)) :
        0.0
    support = count(>(1.0e-8), use)
    bound_violation = max(0.0, maximum(use - ins.u_bar), maximum(-use))
    if poly_violation > cfg.violation_tol || support > ins.sparsity || bound_violation > cfg.violation_tol
        error(
            "ARO separator produced an infeasible sparse scenario after canonicalization: " *
            "poly_violation=$poly_violation, support=$support, s=$(ins.sparsity), " *
            "bound_violation=$bound_violation, inactive_spill=$(cleanup.inactive_spill), " *
            "feasibility_scale=$(cleanup.feasibility_scale)",
        )
    end

    stats = (
        time = time() - t0,
        raw_support = count(>(1.0e-8), raw),
        support = support,
        raw_value = dot(c, raw),
        value = dot(c, use),
        inactive_binary_spill = cleanup.inactive_spill,
        feasibility_scale = cleanup.feasibility_scale,
        pre_scale_poly_violation = cleanup.pre_scale_poly_violation,
    )
    return use, stats
end

function href(ins::AROInstance)
    return vec(maximum(ins.h, dims = 2))
end

function deficit(ins::AROInstance, p::Vector{Float64}, u::Vector{Float64})
    h_ref = href(ins)
    K = size(ins.h, 1)
    I = length(p)
    J = length(u)
    out = zeros(K)
    for k in 1:K
        attack = sum(u[j] * ins.g[k, j, i] * p[i] for j in 1:J, i in 1:I)
        out[k] = (
            ins.theta * ins.sigma2[k] - dot(ins.h[k, :], p) + ins.theta * attack
        ) / h_ref[k]
    end
    return out
end

function solve_lambda(ins::AROInstance, p::Vector{Float64}, u::Vector{Float64})
    K = size(ins.h, 1)
    q = deficit(ins, p, u)
    model = _new_model()
    @variable(model, lambda[1:K] >= 0)
    @constraint(
        model,
        [r = 1:size(ins.B, 2)],
        sum(ins.B[k, r] * lambda[k] for k in 1:K) <= ins.mode_cost[r],
    )
    @objective(model, Max, sum(q[k] * lambda[k] for k in 1:K))
    optimize!(model)
    termination_status(model) == MOI.OPTIMAL || error("recourse dual failed")
    return objective_value(model), value.(lambda)
end

function u_coeff(ins::AROInstance, p::Vector{Float64}, lambda::Vector{Float64})
    h_ref = href(ins)
    J = length(ins.u_bar)
    I = length(p)
    K = length(lambda)
    return [
        ins.theta * sum(
            lambda[k] * ins.g[k, j, i] * p[i] / h_ref[k]
            for k in 1:K, i in 1:I
        )
        for j in 1:J
    ]
end

function cut_coeff(ins::AROInstance, u::Vector{Float64}, lambda::Vector{Float64})
    h_ref = href(ins)
    I = length(ins.p_bar)
    K = size(ins.h, 1)
    J = length(u)
    a = ones(I)

    for i in 1:I
        a[i] += sum(
            lambda[k] / h_ref[k] * (
                -ins.h[k, i] +
                ins.theta * sum(u[j] * ins.g[k, j, i] for j in 1:J)
            )
            for k in 1:K
        )
    end

    b = sum(
        lambda[k] * ins.theta * ins.sigma2[k] / h_ref[k]
        for k in 1:K
    )
    return a, b
end

function alternating_separator(
    ins::AROInstance,
    p::Vector{Float64},
    method::String,
    cfg::AROConfig,
    seed::Int,
)
    rng = MersenneTwister(seed)
    J = length(ins.u_bar)
    K = size(ins.h, 1)
    starts = Vector{Vector{Float64}}()
    push!(starts, zeros(J))

    n_deterministic = min(K, max(1, fld(cfg.alternating_starts, 2)))
    for k in 1:n_deterministic
        lambda0 = zeros(K)
        lambda0[k] = 1.0
        c = u_coeff(ins, p, lambda0)
        try
            u, _ = solve_separator(
                c,
                ins,
                method,
                cfg;
                require_exact_el = uppercase(method) == "EL",
            )
            push!(starts, u)
        catch
            # Optional starting point; skip if the separator cannot certify it.
        end
    end

    while length(starts) < cfg.alternating_starts
        support = randperm(rng, J)[1:min(ins.sparsity, J)]
        u = zeros(J)
        vals = (0.2 .+ 0.8 .* rand(rng, length(support))) .* ins.u_bar[support]
        u[support] = vals
        if size(ins.F, 1) > 0
            load = ins.F * u
            ratios = [ins.d[r] / max(load[r], 1.0e-12) for r in 1:length(ins.d)]
            scale = min(1.0, minimum(ratios))
            u .*= max(0.0, 0.999 * scale)
        end
        push!(starts, u)
    end

    best_value = -Inf
    best_u = copy(starts[1])
    best_lambda = zeros(K)

    for u0 in starts
        u = copy(u0)
        previous_value = -Inf
        lambda = zeros(K)

        for _ in 1:cfg.alternating_max_iterations
            _, lambda = solve_lambda(ins, p, u)
            c = u_coeff(ins, p, lambda)
            u_new, _ = solve_separator(
                c,
                ins,
                method,
                cfg;
                require_exact_el = uppercase(method) == "EL",
            )
            recourse_value, lambda_new = solve_lambda(ins, p, u_new)
            value_now = sum(p) + recourse_value

            if abs(value_now - previous_value) <= 1.0e-8 * max(1.0, abs(value_now)) &&
               maximum(abs.(u_new - u)) <= 1.0e-8
                u = u_new
                lambda = lambda_new
                break
            end

            previous_value = value_now
            u = u_new
            lambda = lambda_new
        end

        recourse_value, lambda = solve_lambda(ins, p, u)
        value_now = sum(p) + recourse_value
        if value_now > best_value + 1.0e-12
            best_value = value_now
            best_u = copy(u)
            best_lambda = copy(lambda)
        end
    end

    return best_value, best_u, best_lambda
end

function solve_dual_master(ins::AROInstance, cuts::Vector{Cut})
    I = length(ins.p_bar)
    K = size(ins.h, 1)
    model = _new_model()
    _prefer_lp_simplex!(model)
    @variable(model, 0 <= p[i = 1:I] <= ins.p_bar[i])
    @variable(model, tau >= 0)

    @constraint(
        model,
        [k = 1:K],
        sum(ins.h[k, i] * p[i] for i in 1:I) >= ins.theta * ins.sigma2[k],
    )
    @constraint(model, tau >= sum(p[i] for i in 1:I))

    for cut in cuts
        a, b = cut_coeff(ins, cut.u, cut.lam)
        @constraint(model, sum(a[i] * p[i] for i in 1:I) + b <= tau)
    end

    @objective(model, Min, tau)
    optimize!(model)
    termination_status(model) == MOI.OPTIMAL || error("ARO master failed")
    return value.(p), value(tau)
end

function solve_db(ins::AROInstance, cfg::AROConfig, method::String)
    method_name = uppercase(method)
    t0 = time()
    p = _solve_nominal(ins.h, ins.sigma2, ins.theta, ins.p_bar)
    tau = sum(p)
    cuts = Cut[]
    max_violation = Inf

    for iteration in 1:cfg.max_iterations
        if time() - t0 > cfg.method_time_limit_s
            return AROResult(
                "DB-$method_name",
                tau,
                time() - t0,
                "TIME_LIMIT",
                p,
                iteration - 1,
                length(cuts),
                max_violation,
            )
        end

        separator_value, u, lambda = alternating_separator(
            ins,
            p,
            method_name,
            cfg,
            1000 * cfg.seed + iteration,
        )
        max_violation = separator_value - tau

        if max_violation <= cfg.violation_tol && !isempty(cuts)
            return AROResult(
                "DB-$method_name",
                tau,
                time() - t0,
                "NO_VIOLATION_FOUND",
                p,
                iteration - 1,
                length(cuts),
                max_violation,
            )
        end

        duplicate = any(
            maximum(abs.(cut.u - u)) < 1.0e-9 &&
            maximum(abs.(cut.lam - lambda)) < 1.0e-8
            for cut in cuts
        )
        if !duplicate
            push!(cuts, Cut(copy(u), copy(lambda)))
        end

        p_new, tau_new = solve_dual_master(ins, cuts)
        monotonicity_tolerance = 1.0e-6 * max(1.0, abs(tau))
        if tau_new + monotonicity_tolerance < tau
            error("ARO lower bound decreased")
        end
        tau_new = max(tau_new, tau)
        p = p_new
        tau = tau_new
    end

    return AROResult(
        "DB-$method_name",
        tau,
        time() - t0,
        "MAX_ITERATIONS",
        p,
        cfg.max_iterations,
        length(cuts),
        max_violation,
    )
end


# Small, dependency-free combination generator used only by exact validation.
function _index_combinations(n::Int, r::Int)
    if r < 0 || r > n
        return Vector{Vector{Int}}()
    elseif r == 0
        return [Int[]]
    end

    out = Vector{Vector{Int}}()
    current = Vector{Int}(undef, r)

    function recurse(position::Int, first_value::Int)
        if position > r
            push!(out, copy(current))
            return
        end
        last_value = n - (r - position)
        for value in first_value:last_value
            current[position] = value
            recurse(position + 1, value + 1)
        end
    end

    recurse(1, 1)
    return out
end

function _vertices_on_support(
    ins::AROInstance,
    support::Vector{Int};
    tol::Float64 = 1.0e-9,
)
    J = length(ins.u_bar)
    r = length(support)
    if r == 0
        return [zeros(J)]
    end

    identity = Matrix{Float64}(I, r, r)
    if size(ins.F, 1) > 0
        A = vcat(-identity, identity, ins.F[:, support])
        b = vcat(zeros(r), ins.u_bar[support], ins.d)
    else
        A = vcat(-identity, identity)
        b = vcat(zeros(r), ins.u_bar[support])
    end

    vertices = Vector{Vector{Float64}}()
    for active in _index_combinations(length(b), r)
        M = A[active, :]
        if rank(M) < r
            continue
        end
        x = M \ b[active]
        if maximum(A * x - b) <= tol
            u = zeros(J)
            u[support] = x
            duplicate = any(maximum(abs.(u - v)) < 1.0e-8 for v in vertices)
            if !duplicate
                push!(vertices, u)
            end
        end
    end
    return vertices
end

function enumerate_sparse_vertices(
    ins::AROInstance;
    max_support_sets::Int = 5000,
)
    J = length(ins.u_bar)
    s = ins.sparsity
    n_support_sets = sum(binomial(J, r) for r in 0:s)
    if n_support_sets > max_support_sets
        error(
            "exact validation disabled: $n_support_sets support sets exceed guard $max_support_sets",
        )
    end

    vertices = Vector{Vector{Float64}}()
    for r in 0:s
        for support_tuple in _index_combinations(J, r)
            support = collect(support_tuple)
            for u in _vertices_on_support(ins, support)
                duplicate = any(maximum(abs.(u - v)) < 1.0e-8 for v in vertices)
                if !duplicate
                    push!(vertices, u)
                end
            end
        end
    end
    return vertices
end

function solve_exact_full(ins::AROInstance, cfg::AROConfig, vertices)
    t0 = time()
    I = length(ins.p_bar)
    K = size(ins.h, 1)
    R = size(ins.B, 2)
    S = length(vertices)
    h_ref = href(ins)

    model = _new_model()
    @variable(model, 0 <= p[i = 1:I] <= ins.p_bar[i])
    @variable(model, tau >= 0)
    @variable(model, y[1:S, 1:R] >= 0)

    @constraint(
        model,
        [k = 1:K],
        sum(ins.h[k, i] * p[i] for i in 1:I) >= ins.theta * ins.sigma2[k],
    )

    for scenario_index in 1:S
        u = vertices[scenario_index]
        for k in 1:K
            lhs = (
                ins.theta * ins.sigma2[k] +
                sum(
                    (
                        -ins.h[k, i] +
                        ins.theta * sum(u[j] * ins.g[k, j, i] for j in 1:length(u))
                    ) * p[i]
                    for i in 1:I
                )
            ) / h_ref[k]
            @constraint(
                model,
                lhs <= sum(ins.B[k, r] * y[scenario_index, r] for r in 1:R),
            )
        end
        @constraint(
            model,
            sum(p[i] for i in 1:I) +
            sum(ins.mode_cost[r] * y[scenario_index, r] for r in 1:R) <= tau,
        )
    end

    @objective(model, Min, tau)
    optimize!(model)
    termination_status(model) == MOI.OPTIMAL || error("exact full validation failed")
    return AROResult(
        "EXACT-FULL",
        value(tau),
        time() - t0,
        "CERTIFIED_BY_VERTEX_ENUMERATION",
        value.(p),
        1,
        0,
        0.0,
    )
end

function solve_exact_aadr(ins::AROInstance, cfg::AROConfig, vertices)
    t0 = time()
    I = length(ins.p_bar)
    K = size(ins.h, 1)
    J = length(ins.u_bar)
    R = size(ins.B, 2)
    h_ref = href(ins)

    model = _new_model()
    @variable(model, 0 <= p[i = 1:I] <= ins.p_bar[i])
    @variable(model, y0[1:R])
    @variable(model, Y[1:R, 1:J])
    @variable(model, tau >= 0)

    @constraint(
        model,
        [k = 1:K],
        sum(ins.h[k, i] * p[i] for i in 1:I) >= ins.theta * ins.sigma2[k],
    )

    for u in vertices
        @constraint(
            model,
            [r = 1:R],
            y0[r] + sum(Y[r, j] * u[j] for j in 1:J) >= 0,
        )
        for k in 1:K
            lhs = (
                ins.theta * ins.sigma2[k] +
                sum(
                    (
                        -ins.h[k, i] +
                        ins.theta * sum(u[j] * ins.g[k, j, i] for j in 1:J)
                    ) * p[i]
                    for i in 1:I
                )
            ) / h_ref[k]
            rhs = sum(
                ins.B[k, r] * (y0[r] + sum(Y[r, j] * u[j] for j in 1:J))
                for r in 1:R
            )
            @constraint(model, lhs <= rhs)
        end
        @constraint(
            model,
            sum(p[i] for i in 1:I) +
            sum(
                ins.mode_cost[r] *
                (y0[r] + sum(Y[r, j] * u[j] for j in 1:J))
                for r in 1:R
            ) <= tau,
        )
    end

    @objective(model, Min, tau)
    optimize!(model)
    termination_status(model) == MOI.OPTIMAL || error("exact affine-policy validation failed")
    return AROResult(
        "AADR-EXACT",
        value(tau),
        time() - t0,
        "CERTIFIED_AFFINE_POLICY",
        value.(p),
        1,
        0,
        0.0,
    )
end

"""Solve the direct robust counterpart of the affine policy over U_RL.

`U_RL = {u >= 0: u <= u_bar, F*u <= d, sum(u ./ u_bar) <= s}` is the
continuous lifted relaxation of the sparse uncertainty set.  Since the true
sparse set is contained in `U_RL`, the returned affine policy is feasible for
the original uncertainty set, but its value is a conservative RL-derived
restricted-policy calculation rather than an independent AADR competitor.
"""
function solve_aadr_rl(ins::AROInstance, cfg::AROConfig)
    t0 = time()
    I = length(ins.p_bar)
    K = size(ins.h, 1)
    J = length(ins.u_bar)
    R = size(ins.B, 2)
    h_ref = href(ins)

    all(ins.u_bar .> 0.0) || error("AADR-RL requires strictly positive u_bar")

    # A*u <= b together with u >= 0 is the polyhedral RL uncertainty set.
    box_rows = Matrix{Float64}(LinearAlgebra.I, J, J)
    cardinality_row = permutedims(1.0 ./ ins.u_bar)
    A = vcat(box_rows, ins.F, cardinality_row)
    b = vcat(ins.u_bar, ins.d, [Float64(ins.sparsity)])
    L = length(b)

    model = _new_model(timelimit = cfg.method_time_limit_s)
    _prefer_lp_interior_point!(model)
    @variable(model, 0 <= p[i = 1:I] <= ins.p_bar[i])
    @variable(model, y0[1:R])
    @variable(model, Y[1:R, 1:J])
    @variable(model, tau >= 0)

    @constraint(
        model,
        [k = 1:K],
        sum(ins.h[k, i] * p[i] for i in 1:I) >= ins.theta * ins.sigma2[k],
    )

    # Robust policy nonnegativity:
    # max_{u in U_RL} -Y[r,:]'u <= y0[r].
    @variable(model, nonneg_dual[1:R, 1:L] >= 0)
    @constraint(
        model,
        [r = 1:R, j = 1:J],
        sum(A[l, j] * nonneg_dual[r, l] for l in 1:L) >= -Y[r, j],
    )
    @constraint(
        model,
        [r = 1:R],
        sum(b[l] * nonneg_dual[r, l] for l in 1:L) <= y0[r],
    )

    # Robust service constraints after substituting y(u) = y0 + Y*u.
    @variable(model, service_dual[1:K, 1:L] >= 0)
    @expression(
        model,
        service_coefficient[k = 1:K, j = 1:J],
        ins.theta * sum(ins.g[k, j, i] * p[i] for i in 1:I) / h_ref[k] -
        sum(ins.B[k, r] * Y[r, j] for r in 1:R),
    )
    @constraint(
        model,
        [k = 1:K, j = 1:J],
        sum(A[l, j] * service_dual[k, l] for l in 1:L) >=
        service_coefficient[k, j],
    )
    @constraint(
        model,
        [k = 1:K],
        (ins.theta * ins.sigma2[k] - sum(ins.h[k, i] * p[i] for i in 1:I)) /
        h_ref[k] - sum(ins.B[k, r] * y0[r] for r in 1:R) +
        sum(b[l] * service_dual[k, l] for l in 1:L) <= 0,
    )

    # Robust epigraph constraint for first-stage plus recovery cost.
    @variable(model, cost_dual[1:L] >= 0)
    @expression(
        model,
        cost_coefficient[j = 1:J],
        sum(ins.mode_cost[r] * Y[r, j] for r in 1:R),
    )
    @constraint(
        model,
        [j = 1:J],
        sum(A[l, j] * cost_dual[l] for l in 1:L) >= cost_coefficient[j],
    )
    @constraint(
        model,
        sum(p[i] for i in 1:I) + sum(ins.mode_cost[r] * y0[r] for r in 1:R) - tau +
        sum(b[l] * cost_dual[l] for l in 1:L) <= 0,
    )

    @objective(model, Min, tau)
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
        if _is_solver_resource_limit(status)
            throw(
                SolverTimeLimit(
                    "AADR-RL robust-counterpart LP",
                    "termination_status=$status, raw_status=$(raw_status(model))",
                ),
            )
        end
        error(
            "AADR-RL robust-counterpart LP failed: termination_status=$status, " *
            "primal_status=$(primal_status(model)), raw_status=$(raw_status(model))",
        )
    end

    return AROResult(
        "AADR-RL",
        value(tau),
        time() - t0,
        "CERTIFIED_AADR_RL_POLICY",
        value.(p),
        1,
        R + K + 1,
        0.0,
    )
end

function solve_exact_ro(ins::AROInstance, vertices)
    t0 = time()
    I = length(ins.p_bar)
    K = size(ins.h, 1)
    model = _new_model()
    @variable(model, 0 <= p[i = 1:I] <= ins.p_bar[i])

    for u in vertices
        for k in 1:K
            @constraint(
                model,
                sum(
                    (
                        -ins.h[k, i] +
                        ins.theta * sum(u[j] * ins.g[k, j, i] for j in 1:length(u))
                    ) * p[i]
                    for i in 1:I
                ) <= -ins.theta * ins.sigma2[k],
            )
        end
    end

    @objective(model, Min, sum(p[i] for i in 1:I))
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
        return AROResult(
            "RO-EL",
            Inf,
            time() - t0,
            "INFEASIBLE",
            zeros(I),
            1,
            0,
            0.0,
        )
    end
    return AROResult(
        "RO-EL",
        objective_value(model),
        time() - t0,
        "CERTIFIED_ONE_STAGE",
        value.(p),
        1,
        0,
        0.0,
    )
end

function solve_ro_master(ins::AROInstance, pools)
    I = length(ins.p_bar)
    K = size(ins.h, 1)
    model = _new_model()
    @variable(model, 0 <= p[i = 1:I] <= ins.p_bar[i])
    @objective(model, Min, sum(p[i] for i in 1:I))

    for k in 1:K
        for u in pools[k]
            @constraint(
                model,
                sum(
                    (
                        -ins.h[k, i] +
                        ins.theta * sum(u[j] * ins.g[k, j, i] for j in 1:length(u))
                    ) * p[i]
                    for i in 1:I
                ) <= -ins.theta * ins.sigma2[k],
            )
        end
    end

    optimize!(model)
    termination_status(model) == MOI.OPTIMAL || error("RO master infeasible")
    return value.(p), objective_value(model)
end

function solve_ro_el_cp(ins::AROInstance, cfg::AROConfig)
    t0 = time()
    K = size(ins.h, 1)
    zero_scenario = zeros(length(ins.u_bar))
    pools = [[copy(zero_scenario)] for _ in 1:K]
    p, objective = solve_ro_master(ins, pools)
    max_violation = Inf

    for iteration in 1:cfg.max_iterations
        new_cuts = Tuple{Int,Vector{Float64}}[]
        max_violation = -Inf

        for k in 1:K
            c = [
                ins.theta * sum(ins.g[k, j, i] * p[i] for i in 1:length(p))
                for j in 1:length(ins.u_bar)
            ]
            u, _ = solve_separator(c, ins, "EL", cfg; require_exact_el = true)
            violation =
                -dot(ins.h[k, :], p) + ins.theta * ins.sigma2[k] + dot(c, u)
            max_violation = max(max_violation, violation)
            if violation > cfg.violation_tol
                push!(new_cuts, (k, u))
            end
        end

        if max_violation <= cfg.violation_tol
            n_cuts = sum(length(pool) for pool in pools) - K
            return AROResult(
                "RO-EL",
                objective,
                time() - t0,
                "CERTIFIED_ONE_STAGE",
                p,
                iteration - 1,
                n_cuts,
                max_violation,
            )
        end

        for (k, u) in new_cuts
            push!(pools[k], u)
        end
        p, objective = solve_ro_master(ins, pools)

        if time() - t0 > cfg.method_time_limit_s
            n_cuts = sum(length(pool) for pool in pools) - K
            return AROResult(
                "RO-EL",
                objective,
                time() - t0,
                "TIME_LIMIT",
                p,
                iteration,
                n_cuts,
                max_violation,
            )
        end
    end

    n_cuts = sum(length(pool) for pool in pools) - K
    return AROResult(
        "RO-EL",
        objective,
        time() - t0,
        "MAX_ITERATIONS",
        p,
        cfg.max_iterations,
        n_cuts,
        max_violation,
    )
end

function solve_aadr_master(
    ins::AROInstance,
    scenarios;
    deadline::Float64 = Inf,
)
    I = length(ins.p_bar)
    K = size(ins.h, 1)
    J = length(ins.u_bar)
    R = size(ins.B, 2)
    h_ref = href(ins)

    remaining = isfinite(deadline) ? max(deadline - time(), 1.0e-3) : nothing
    model = _new_model(timelimit = remaining)
    _prefer_lp_interior_point!(model)
    @variable(model, 0 <= p[i = 1:I] <= ins.p_bar[i])
    @variable(model, y0[1:R])
    @variable(model, Y[1:R, 1:J])
    @variable(model, tau >= 0)

    @constraint(
        model,
        [k = 1:K],
        sum(ins.h[k, i] * p[i] for i in 1:I) >= ins.theta * ins.sigma2[k],
    )

    for u in scenarios
        time() > deadline && throw(
            SolverTimeLimit("AADR-RL/EL master construction", "METHOD_DEADLINE"),
        )
        @constraint(
            model,
            [r = 1:R],
            y0[r] + sum(Y[r, j] * u[j] for j in 1:J) >= 0,
        )
        for k in 1:K
            lhs = (
                ins.theta * ins.sigma2[k] +
                sum(
                    (
                        -ins.h[k, i] +
                        ins.theta * sum(u[j] * ins.g[k, j, i] for j in 1:J)
                    ) * p[i]
                    for i in 1:I
                )
            ) / h_ref[k]
            rhs = sum(
                ins.B[k, r] * (y0[r] + sum(Y[r, j] * u[j] for j in 1:J))
                for r in 1:R
            )
            @constraint(model, lhs <= rhs)
        end
        @constraint(
            model,
            sum(p[i] for i in 1:I) +
            sum(
                ins.mode_cost[r] *
                (y0[r] + sum(Y[r, j] * u[j] for j in 1:J))
                for r in 1:R
            ) <= tau,
        )
    end

    @objective(model, Min, tau)
    if isfinite(deadline)
        remaining = deadline - time()
        remaining > 0.0 || throw(
            SolverTimeLimit("AADR-RL/EL master construction", "METHOD_DEADLINE"),
        )
        set_time_limit_sec(model, max(remaining, 1.0e-3))
    end
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
        if _is_solver_resource_limit(status)
            throw(
                SolverTimeLimit(
                    "AADR-RL/EL master LP",
                    "termination_status=$status, raw_status=$(raw_status(model))",
                ),
            )
        end
        error(
            "AADR-RL/EL master LP failed: termination_status=$status, " *
            "primal_status=$(primal_status(model)), raw_status=$(raw_status(model))",
        )
    end
    return value.(p), value.(y0), value.(Y), value(tau)
end

function _aadr_separator_or_timeout(
    c::Vector{Float64},
    ins::AROInstance,
    method::String,
    cfg::AROConfig;
    require_exact_el::Bool,
)
    try
        u, stats = solve_separator(
            c,
            ins,
            method,
            cfg;
            require_exact_el = require_exact_el,
        )
        return u, stats, false
    catch err
        err isa SolverTimeLimit || rethrow()
        return zeros(length(c)), nothing, true
    end
end

function _collect_aadr_candidates(
    ins::AROInstance,
    cfg::AROConfig,
    p::Vector{Float64},
    y0::Vector{Float64},
    Y::Matrix{Float64},
    tau::Float64,
    method::String;
    deadline::Float64 = Inf,
)
    J = length(ins.u_bar)
    R = size(ins.B, 2)
    K = size(ins.h, 1)
    h_ref = href(ins)
    exact_el = uppercase(method) == "EL"
    candidates = Vector{Tuple{Float64,Vector{Float64},String,Int}}()

    for r in 1:R
        time() > deadline && return candidates, true
        c = -vec(Y[r, :])
        u, _, separator_timed_out = _aadr_separator_or_timeout(
            c, ins, method, cfg; require_exact_el = exact_el,
        )
        separator_timed_out && return candidates, true
        violation = -y0[r] + dot(c, u)
        push!(candidates, (violation, copy(u), "nonnegativity", r))
    end

    for k in 1:K
        time() > deadline && return candidates, true
        c = [
            ins.theta * sum(ins.g[k, j, i] * p[i] for i in 1:length(p)) / h_ref[k] -
            sum(ins.B[k, r] * Y[r, j] for r in 1:R)
            for j in 1:J
        ]
        u, _, separator_timed_out = _aadr_separator_or_timeout(
            c, ins, method, cfg; require_exact_el = exact_el,
        )
        separator_timed_out && return candidates, true
        intercept =
            (ins.theta * ins.sigma2[k] - dot(ins.h[k, :], p)) / h_ref[k] -
            dot(ins.B[k, :], y0)
        violation = intercept + dot(c, u)
        push!(candidates, (violation, copy(u), "service", k))
    end

    time() > deadline && return candidates, true
    c = vec(transpose(ins.mode_cost) * Y)
    u, _, separator_timed_out = _aadr_separator_or_timeout(
        c, ins, method, cfg; require_exact_el = exact_el,
    )
    separator_timed_out && return candidates, true
    violation = sum(p) + dot(ins.mode_cost, y0) - tau + dot(c, u)
    push!(candidates, (violation, copy(u), "cost", 1))

    sort!(candidates, by = x -> x[1], rev = true)
    return candidates, false
end

function _append_aadr_scenarios!(
    scenarios::Vector{Vector{Float64}},
    seen::Set,
    candidates,
    cfg::AROConfig,
)
    added = 0
    for candidate in candidates
        violation, u = candidate[1], candidate[2]
        violation > cfg.violation_tol || break
        key = Tuple(round.(u, digits = 10))
        if !(key in seen)
            push!(seen, key)
            push!(scenarios, copy(u))
            added += 1
            added >= cfg.aadr_batch_size && break
        end
    end
    return added
end

function solve_aadr_rl_el_cp(ins::AROInstance, cfg::AROConfig)
    t0 = time()
    J = length(ins.u_bar)
    scenarios = [zeros(J)]
    seen = Set([Tuple(round.(scenarios[1], digits = 10))])
    worst_violation = Inf
    p = zeros(length(ins.p_bar))
    tau = Inf

    for iteration in 1:cfg.max_iterations
        if time() - t0 > cfg.method_time_limit_s
            return AROResult(
                "AADR-RL/EL",
                isfinite(tau) ? tau : NaN,
                time() - t0,
                "TIME_LIMIT",
                p,
                iteration - 1,
                length(scenarios) - 1,
                worst_violation,
            )
        end

        master_result = try
            solve_aadr_master(
                ins,
                scenarios;
                deadline = t0 + cfg.method_time_limit_s,
            )
        catch err
            if err isa SolverTimeLimit
                return AROResult(
                    "AADR-RL/EL",
                    isfinite(tau) ? tau : NaN,
                    time() - t0,
                    _resource_limit_label(err),
                    p,
                    iteration - 1,
                    length(scenarios) - 1,
                    worst_violation,
                )
            end
            rethrow()
        end
        p, y0, Y, tau = master_result

        # Cheap multi-scenario screening first.  RL scenarios are always feasible
        # for the true sparse set, so any positive violation is a valid cut.  A
        # clean RL scan is not a certificate; it only triggers the exact EL scan.
        if cfg.aadr_screen_with_rl
            screen_candidates, screen_timed_out = _collect_aadr_candidates(
                ins, cfg, p, y0, Y, tau, "RL";
                deadline = t0 + cfg.method_time_limit_s,
            )
            if screen_timed_out
                return AROResult(
                    "AADR-RL/EL", tau, time() - t0, "TIME_LIMIT", p, iteration - 1,
                    length(scenarios) - 1, worst_violation,
                )
            end
            screen_worst = isempty(screen_candidates) ? -Inf : screen_candidates[1][1]
            worst_violation = screen_worst
            if screen_worst > cfg.violation_tol
                added = _append_aadr_scenarios!(scenarios, seen, screen_candidates, cfg)
                if added > 0
                    continue
                end
            end
        end

        if time() - t0 > cfg.method_time_limit_s
            return AROResult(
                "AADR-RL/EL",
                tau,
                time() - t0,
                "TIME_LIMIT",
                p,
                iteration - 1,
                length(scenarios) - 1,
                worst_violation,
            )
        end

        # Exact EL separation is the certification step for the affine policy.
        exact_candidates, exact_timed_out = _collect_aadr_candidates(
            ins, cfg, p, y0, Y, tau, "EL";
            deadline = t0 + cfg.method_time_limit_s,
        )
        if exact_timed_out
            return AROResult(
                "AADR-RL/EL", tau, time() - t0, "TIME_LIMIT", p, iteration - 1,
                length(scenarios) - 1, worst_violation,
            )
        end
        worst_violation = isempty(exact_candidates) ? -Inf : exact_candidates[1][1]

        if worst_violation <= cfg.violation_tol
            return AROResult(
                "AADR-RL/EL",
                tau,
                time() - t0,
                "CERTIFIED_AFFINE_POLICY",
                p,
                iteration - 1,
                length(scenarios) - 1,
                worst_violation,
            )
        end

        added = _append_aadr_scenarios!(scenarios, seen, exact_candidates, cfg)
        if added == 0
            return AROResult(
                "AADR-RL/EL",
                tau,
                time() - t0,
                "STALLED",
                p,
                iteration - 1,
                length(scenarios) - 1,
                worst_violation,
            )
        end

        if time() - t0 > cfg.method_time_limit_s
            return AROResult(
                "AADR-RL/EL",
                tau,
                time() - t0,
                "TIME_LIMIT",
                p,
                iteration,
                length(scenarios) - 1,
                worst_violation,
            )
        end
    end

    return AROResult(
        "AADR-RL/EL",
        tau,
        time() - t0,
        "MAX_ITERATIONS",
        p,
        cfg.max_iterations,
        length(scenarios) - 1,
        worst_violation,
    )
end

# Backward-compatible entry point.  The result label is deliberately explicit
# about the RL screening and EL certification used by the scalable procedure.
solve_aadr_el_cp(ins::AROInstance, cfg::AROConfig) = solve_aadr_rl_el_cp(ins, cfg)

function run_one_aro(
    cfg::AROConfig;
    methods = ["SP", "RL", "EL"],
    run_aadr = nothing,
    run_aadr_rl_el = false,
    run_aadr_rl = true,
    run_ro = true,
    exact_validation = false,
)
    ins = generate_instance(cfg)
    rows = NamedTuple[]
    results = Dict{String,AROResult}()

    if exact_validation
        vertices = enumerate_sparse_vertices(ins)

        result = solve_exact_full(ins, cfg, vertices)
        results[result.method] = result
        push!(rows, _result_row(result))

        result = solve_exact_aadr(ins, cfg, vertices)
        results[result.method] = result
        push!(rows, _result_row(result))

        result = solve_exact_ro(ins, vertices)
        results[result.method] = result
        push!(rows, _result_row(result))
    else
        # `run_aadr` is retained only for compatibility with earlier callers
        # callers and enables/disables both newly distinguished AADR variants.
        # New callers should use the two explicit flags.
        if run_aadr !== nothing
            run_aadr_rl_el = Bool(run_aadr)
            run_aadr_rl = Bool(run_aadr)
        end

        if run_aadr_rl
            method_started = time()
            try
                result = solve_aadr_rl(ins, cfg)
                results[result.method] = result
                push!(rows, _result_row(result))
            catch err
                if err isa SolverTimeLimit
                    push!(rows, _limit_row(
                        "AADR-RL", time() - method_started,
                        _resource_limit_label(err),
                    ))
                else
                    @warn "AADR-RL policy comparison failed" exception = (err, catch_backtrace())
                    push!(rows, _failure_row("AADR-RL", time() - method_started))
                end
            end
        end

        if run_aadr_rl_el
            method_started = time()
            try
                result = solve_aadr_rl_el_cp(ins, cfg)
                results[result.method] = result
                push!(rows, _result_row(result))
            catch err
                if err isa SolverTimeLimit
                    push!(rows, _limit_row(
                        "AADR-RL/EL", time() - method_started,
                        _resource_limit_label(err),
                    ))
                else
                    @warn "AADR-RL/EL policy procedure failed" exception = (err, catch_backtrace())
                    push!(rows, _failure_row("AADR-RL/EL", time() - method_started))
                end
            end
        end

        if run_ro
            try
                result = solve_ro_el_cp(ins, cfg)
                results[result.method] = result
                push!(rows, _result_row(result))
            catch err
                @warn "RO-EL failed or timed out" exception = (err, catch_backtrace())
                push!(rows, _failure_row("RO-EL", cfg.method_time_limit_s))
            end
        end
    end

    for method in methods
        method_name = uppercase(method)
        try
            result = solve_db(ins, cfg, method_name)
            results[result.method] = result
            push!(rows, _result_row(result))
        catch err
            @warn "DB method failed or timed out" method = method_name exception = (err, catch_backtrace())
            push!(rows, _failure_row("DB-$method_name", cfg.method_time_limit_s))
        end
    end

    return ins, results, rows
end

function _result_row(result::AROResult)
    return (
        method = result.method,
        objective = result.objective,
        wall_time_s = result.wall_time_s,
        status = result.status,
        iterations = result.iterations,
        cuts = result.cuts,
        max_violation = result.max_violation,
    )
end

function _failure_row(method::String, time_limit::Float64)
    return (
        method = method,
        objective = NaN,
        wall_time_s = time_limit,
        status = "FAILED_OR_TIMEOUT",
        iterations = 0,
        cuts = 0,
        max_violation = NaN,
    )
end

function _limit_row(method::String, elapsed::Float64, status::String)
    return (
        method = method,
        objective = NaN,
        wall_time_s = elapsed,
        status = status,
        iterations = 0,
        cuts = 0,
        max_violation = NaN,
    )
end

end # module WirelessSparseARO
