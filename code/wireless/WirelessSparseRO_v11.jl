module WirelessSparseRO

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

export ROConfig,
       ROInstance,
       ROResult,
       recommended_q_groups,
       generate_instance,
       load_instance_arrays,
       solve_separator,
       run_cutting_plane,
       exact_validate,
       run_one_ro,
       set_optimizer!

Base.@kwdef struct ROConfig
    seed::Int = 2
    instance_design::Symbol = :physical_overlap
    data_path::String = joinpath(@__DIR__, "..", "..", "data", "wireless_source", "dichasus-d036-32subbands.npz")
    n_tx::Int = 10
    n_j::Int = 350
    m_ue::Int = 35
    q_groups::Int = 120
    sparsity::Int = 15
    radius_m::Float64 = 1000.0
    inter_site_distance_m::Float64 = 500.0
    theta::Float64 = 0.35
    p_max::Float64 = 80.0
    memberships_per_jammer::Int = 4
    min_group_size::Int = 6
    budget_multiplier_low::Float64 = 1.5
    budget_multiplier_high::Float64 = 2.4
    jammer_stress_fraction::Float64 = 0.45
    resource_capacity::Float64 = 0.95
    cut_batch::Int = 3
    violation_tol::Float64 = 1.0e-6
    max_iterations::Int = 60
    mip_time_limit_s::Float64 = 120.0
    mip_rel_gap::Float64 = 1.0e-6
    method_time_limit_s::Float64 = 1000.0
end

struct ROInstance
    h::Matrix{Float64}
    g::Array{Float64,3}
    sigma2::Vector{Float64}
    u_bar::Vector{Float64}
    F::Matrix{Float64}
    d::Vector{Float64}
    p_bar::Vector{Float64}
    theta::Float64
    sparsity::Int
    diagnostics::Dict{String,Float64}
end

struct ROResult
    method::String
    objective::Float64
    wall_time_s::Float64
    iterations::Int
    cuts::Int
    max_violation::Float64
    converged::Bool
    p::Vector{Float64}
    history::Vector{NamedTuple}
end

function recommended_q_groups(
    n_j::Int;
    memberships_per_jammer::Int = 4,
    target_mean_group_size::Float64 = 12.0,
)
    return max(
        memberships_per_jammer + 1,
        round(Int, memberships_per_jammer * n_j / target_mean_group_size),
    )
end

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

function _generate_physical_overlap_instance(cfg::ROConfig)
    cfg.q_groups >= cfg.memberships_per_jammer + 1 ||
        error("q_groups must be at least memberships_per_jammer + 1")
    1 <= cfg.sparsity <= cfg.n_j || error("invalid sparsity")

    rng = MersenneTwister(cfg.seed)
    I = cfg.n_tx
    J = cfg.n_j
    K = cfg.m_ue
    Q = cfg.q_groups

    bs = _hexagonal_sites(I, cfg.inter_site_distance_m)
    ue = _points_in_disk(rng, K, 0.90 * cfg.radius_m)
    jm = _points_in_disk(rng, J, 0.90 * cfg.radius_m)

    # 3D link lengths use the 25 m UMa base-station height and a 1.5 m
    # terminal height. Potential jammers are placed at terminal height.
    d_bu = [hypot(norm(ue[k, :] - bs[i, :]), 25.0 - 1.5) for k in 1:K, i in 1:I]
    d_bj = [hypot(norm(bs[i, :] - jm[j, :]), 25.0 - 1.5) for i in 1:I, j in 1:J]
    d_ju = [max(norm(ue[k, :] - jm[j, :]), 1.0) for k in 1:K, j in 1:J]

    h = _large_scale_gain(d_bu, 3.5, 8.0, rng)
    gain_bj = _large_scale_gain(d_bj, 2.2, 3.0, rng)
    gain_ju = _large_scale_gain(d_ju, 2.0, 3.0, rng)

    raw_g = zeros(K, J, I)
    @inbounds for k in 1:K
        for j in 1:J
            for i in 1:I
                raw_g[k, j, i] = gain_ju[k, j] * gain_bj[i, j]
            end
        end
    end

    u_bar = 0.8 .+ 0.4 .* rand(rng, J)

    raw_c_equal_power = zeros(K, J)
    for k in 1:K
        for j in 1:J
            raw_c_equal_power[k, j] = sum(raw_g[k, j, i] for i in 1:I)
        end
    end

    top_s_box = zeros(K)
    for k in 1:K
        vals = raw_c_equal_power[k, :] .* u_bar
        sorted_vals = sort(collect(vals), rev = true)
        top_s_box[k] = sum(sorted_vals[1:cfg.sparsity])
    end

    useful_equal_power = vec(sum(h, dims = 2))
    scale_candidates = useful_equal_power ./ (cfg.theta .* top_s_box .+ 1.0e-14)
    jammer_scale = cfg.jammer_stress_fraction * minimum(scale_candidates)
    g = jammer_scale .* raw_g

    sigma2 = fill(0.003 * median(useful_equal_power), K)

    # Spatial resource pools: each jammer draws from its four nearest pools.
    # The construction yields overlapping local power/control limits.
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

    group_sizes = vec(sum(F .> 0.0, dims = 2))
    jammer_memberships = vec(sum(F .> 0.0, dims = 1))
    diagnostics = Dict(
        "jammer_scale" => Float64(jammer_scale),
        "mean_group_size" => Float64(mean(group_sizes)),
        "min_group_size" => Float64(minimum(group_sizes)),
        "max_group_size" => Float64(maximum(group_sizes)),
        "mean_jammer_memberships" => Float64(mean(jammer_memberships)),
        "inter_site_distance_m" => Float64(cfg.inter_site_distance_m),
    )

    p_bar = fill(cfg.p_max, I)
    return ROInstance(
        h,
        g,
        sigma2,
        u_bar,
        F,
        d,
        p_bar,
        cfg.theta,
        cfg.sparsity,
        diagnostics,
    )
end

function generate_instance(cfg::ROConfig)
    if cfg.instance_design == :physical_overlap
        return _generate_physical_overlap_instance(cfg)
    elseif cfg.instance_design != :dichasus
        error("instance_design must be :physical_overlap or :dichasus")
    end

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
    return ROInstance(
        arrays.h,
        arrays.g,
        arrays.sigma2,
        arrays.u_bar,
        arrays.F,
        arrays.d,
        arrays.p_bar,
        cfg.theta,
        cfg.sparsity,
        arrays.diagnostics,
    )
end

function load_instance_arrays(h, g, sigma2, u_bar, F, d, p_bar, theta, sparsity)
    return ROInstance(
        Float64.(h),
        Float64.(g),
        vec(Float64.(sigma2)),
        vec(Float64.(u_bar)),
        Float64.(F),
        vec(Float64.(d)),
        vec(Float64.(p_bar)),
        Float64(theta),
        Int(sparsity),
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
    @constraint(model, [r = 1:size(F, 1)], sum(F[r, j] * u[j] for j in eachindex(c)) <= d[r])
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
    ins::ROInstance,
    method::String,
    cfg::ROConfig;
    require_exact_el::Bool = true,
    time_limit_s::Float64 = cfg.mip_time_limit_s,
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
        @constraint(
            model,
            [r = 1:size(ins.F, 1)],
            sum(ins.F[r, j] * u[j] for j in 1:J) <= ins.d[r],
        )
        if method_name == "RL"
            @constraint(model, sum(u[j] / ins.u_bar[j] for j in 1:J) <= ins.sparsity)
        end
        @objective(model, Max, sum(c_scaled[j] * u[j] for j in 1:J))
        optimize!(model)
        status = termination_status(model)
        if status != MOI.OPTIMAL
            # SP/RL are heuristic scenario generators. MOSEK documents SLOW_PROGRESS
            # as a numerical termination for which a usable solution may still be
            # available; accept it only when JuMP exposes a primal solution.
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
        model = _new_model(timelimit = time_limit_s)
        _set_mip_rel_gap!(model, cfg.mip_rel_gap)
        @variable(model, 0 <= u[j = 1:J] <= ins.u_bar[j])
        @variable(model, z[1:J], Bin)
        @constraint(model, [j = 1:J], u[j] <= ins.u_bar[j] * z[j])
        @constraint(model, sum(z[j] for j in 1:J) <= ins.sparsity)
        @constraint(
            model,
            [r = 1:size(ins.F, 1)],
            sum(ins.F[r, j] * u[j] for j in 1:J) <= ins.d[r],
        )
        @objective(model, Max, sum(c_scaled[j] * u[j] for j in 1:J))
        optimize!(model)
        status = termination_status(model)
        if require_exact_el && status != MOI.OPTIMAL
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

    poly_violation = max(0.0, maximum(ins.F * use - ins.d))
    support = count(>(1.0e-8), use)
    bound_violation = max(0.0, maximum(use - ins.u_bar), maximum(-use))
    if poly_violation > cfg.violation_tol || support > ins.sparsity || bound_violation > cfg.violation_tol
        error(
            "separator produced an infeasible sparse scenario after canonicalization: " *
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

function robust_row(ins::ROInstance, k::Int, u::Vector{Float64})
    jammer_coeff_by_bs = vec(transpose(ins.g[k, :, :]) * u)
    return -vec(ins.h[k, :]) + ins.theta .* jammer_coeff_by_bs
end

function constraint_violation(ins::ROInstance, p::Vector{Float64}, k::Int, u::Vector{Float64})
    return dot(robust_row(ins, k, u), p) + ins.theta * ins.sigma2[k]
end

function solve_master(ins::ROInstance, pools)
    K, I = size(ins.h)
    model = _new_model()
    _prefer_lp_simplex!(model)
    @variable(model, 0 <= p[i = 1:I] <= ins.p_bar[i])
    @objective(model, Min, sum(p[i] for i in 1:I))

    zero_scenario = zeros(length(ins.u_bar))
    for k in 1:K
        nominal_row = robust_row(ins, k, zero_scenario)
        @constraint(
            model,
            sum(nominal_row[i] * p[i] for i in 1:I) <= -ins.theta * ins.sigma2[k],
        )
        for u in pools[k]
            row = robust_row(ins, k, u)
            @constraint(
                model,
                sum(row[i] * p[i] for i in 1:I) <= -ins.theta * ins.sigma2[k],
            )
        end
    end

    optimize!(model)
    termination_status(model) == MOI.OPTIMAL ||
        error("RO master failed: $(termination_status(model))")
    return value.(p), objective_value(model)
end

function run_cutting_plane(
    ins::ROInstance,
    method::String,
    cfg::ROConfig;
    verbose::Bool = false,
)
    method_name = uppercase(method)
    K = size(ins.h, 1)
    pools = [Vector{Vector{Float64}}() for _ in 1:K]
    p, objective = solve_master(ins, pools)

    history = NamedTuple[]
    cuts = 0
    converged = false
    t0 = time()
    max_violation = Inf

    for iteration in 1:cfg.max_iterations
        elapsed = time() - t0
        if elapsed > cfg.method_time_limit_s
            break
        end

        scenarios = Vector{Vector{Float64}}(undef, K)
        violations = zeros(K)
        raw_supports = zeros(K)
        kept_supports = zeros(K)

        for k in 1:K
            elapsed = time() - t0
            remaining = cfg.method_time_limit_s - elapsed
            if remaining <= 0.0
                break
            end
            separator_limit = min(cfg.mip_time_limit_s, max(0.01, remaining))
            c = vec(ins.g[k, :, :] * p)
            u, stats = solve_separator(
                c,
                ins,
                method_name,
                cfg;
                require_exact_el = true,
                time_limit_s = separator_limit,
            )
            scenarios[k] = u
            violations[k] = constraint_violation(ins, p, k, u)
            raw_supports[k] = stats.raw_support
            kept_supports[k] = stats.support
        end

        max_violation = max(0.0, maximum(violations))
        n_violated = count(>(cfg.violation_tol), violations)
        push!(
            history,
            (
                iteration = iteration,
                objective = objective,
                max_violation = max_violation,
                violated_ues = n_violated,
                cuts = cuts,
                mean_raw_support = mean(raw_supports),
                mean_projected_support = mean(kept_supports),
            ),
        )

        if verbose
            println(
                method_name,
                " iter ",
                iteration,
                " | LB=",
                objective,
                " | max violation=",
                max_violation,
                " | cuts=",
                cuts,
            )
        end

        if max_violation <= cfg.violation_tol
            converged = true
            break
        end

        violated_idx = findall(>(cfg.violation_tol), violations)
        sort!(violated_idx, by = k -> violations[k], rev = true)
        n_selected = min(cfg.cut_batch, length(violated_idx))
        for pos in 1:n_selected
            k = violated_idx[pos]
            push!(pools[k], scenarios[k])
            cuts += 1
        end

        p_new, objective_new = solve_master(ins, pools)
        if objective_new + 1.0e-9 < objective
            error("RO master lower bound decreased")
        end
        p = p_new
        objective = objective_new
    end

    return ROResult(
        method_name,
        objective,
        time() - t0,
        length(history),
        cuts,
        max_violation,
        converged,
        p,
        history,
    )
end

function exact_validate(ins::ROInstance, p::Vector{Float64}, cfg::ROConfig)
    violations = Float64[]
    t0 = time()
    for k in 1:size(ins.h, 1)
        c = vec(ins.g[k, :, :] * p)
        u, _ = solve_separator(c, ins, "EL", cfg; require_exact_el = true)
        push!(violations, constraint_violation(ins, p, k, u))
    end
    return (max_violation = max(0.0, maximum(violations)), time_s = time() - t0)
end

function run_one_ro(
    cfg::ROConfig;
    methods = ["SP", "RL", "EL"],
    validate = true,
    verbose = false,
)
    ins = generate_instance(cfg)
    rows = NamedTuple[]
    results = Dict{String,ROResult}()

    for method in methods
        method_name = uppercase(method)
        method_t0 = time()
        try
            result = run_cutting_plane(ins, method_name, cfg; verbose = verbose)
            results[method_name] = result
            validation = validate ?
                exact_validate(ins, result.p, cfg) :
                (max_violation = NaN, time_s = 0.0)
            push!(
                rows,
                (
                    method = method_name,
                    objective = result.objective,
                    wall_time_s = result.wall_time_s,
                    iterations = result.iterations,
                    cuts = result.cuts,
                    max_method_violation = result.max_violation,
                    exact_max_violation = validation.max_violation,
                    status = result.converged ? "CONVERGED" : "TIME_OR_ITER_LIMIT",
                ),
            )
        catch err
            @warn "RO method failed or timed out" method = method_name exception = (err, catch_backtrace())
            push!(
                rows,
                (
                    method = method_name,
                    objective = NaN,
                    wall_time_s = time() - method_t0,
                    iterations = 0,
                    cuts = 0,
                    max_method_violation = NaN,
                    exact_max_violation = NaN,
                    status = "FAILED_OR_TIMEOUT",
                ),
            )
        end
    end

    return ins, results, rows
end

end # module WirelessSparseRO
