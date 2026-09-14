using CSV
using DataFrames
using JuMP
using LinearAlgebra
using MathOptInterface
using MosekTools
using Printf
using Random
using SparseArrays
using Statistics

const MOI = MathOptInterface
const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = joinpath(ROOT, "data", "netlib_large")

function argvalue(flag::String, default::String)
    index = findfirst(==(flag), ARGS)
    return index === nothing ? default : ARGS[index + 1]
end

const RESULT_DIR = abspath(argvalue("--outdir", joinpath(ROOT, "reproduced", "implementation_error")))
mkpath(RESULT_DIR)
const INSTANCES = String.(split(argvalue("--instances", "SCFXM1,SCFXM3,25FV47,WOODW,FIT2D"), ','))
const SPARSITY_LEVELS = parse.(Int, split(argvalue("--s", "10,20"), ','))
const METHODS = String.(split(argvalue("--methods", "EL,RL"), ','))
const ERROR_SCALE = 0.01
const FEASIBILITY_TOL = 1.0e-7
const MAX_ITERATIONS = 100

"""Complete a deterministic index list to the requested number of distinct entries."""
function complete_distinct_indices(candidates, target::Int, modulus::Int, start::Int)
    target <= modulus || error("cannot select $target distinct indices from $modulus entries")
    indices = unique(Int.(candidates))
    offset = 0
    while length(indices) < target
        candidate = 1 + mod(start - 1 + offset, modulus)
        candidate in indices || push!(indices, candidate)
        offset += 1
    end
    return indices[1:target]
end

const SOURCE_COUNT = Dict(
    "SCFXM1" => 240,
    "SCFXM2" => 360,
    "SCFXM3" => 480,
    "SCAGR25" => 300,
    "BNL1" => 450,
    "25FV47" => 600,
    "PILOT87" => 900,
    "80BAU3B" => 1200,
    "WOODW" => 1200,
    "FIT2D" => 1500,
)

"""Construct deterministic overlapping implementation-resource limits.

The public LP supplies the nominal model. The extra rows describe shared
calibration or installation resources for coefficient implementations. Each
source belongs to four overlapping groups with heterogeneous loads.
"""
function coupling_matrix(n_sources::Int)
    group_count = max(12, ceil(Int, n_sources / 8))
    rng = MersenneTwister(10_000 + n_sources)
    F = zeros(group_count, n_sources)
    for i in 1:n_sources
        candidates = [
            1 + mod(i - 1, group_count),
            1 + mod(3 * i + 1, group_count),
            1 + mod(7 * i + 3, group_count),
            1 + mod(11 * i + 5, group_count),
        ]
        groups = complete_distinct_indices(candidates, 4, group_count, 13i + 7)
        for group in groups
            F[group, i] = 0.50 + rand(rng)
        end
    end
    # Each resource pool can sustain roughly two fully active sources.
    d = fill(2.0, group_count)
    return F, d
end

"""Map public LP variables to overlapping sources of coefficient error.

Each row is nonnegative and sums to one, so a source realization in [-1,1]
produces a relative implementation error bounded by ERROR_SCALE for every variable.
"""
function calibration_loading(n_variables::Int, n_sources::Int)
    rng = MersenneTwister(20_000 + 13n_variables + n_sources)
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    for i in 1:n_variables
        candidates = complete_distinct_indices([
            1 + mod(i - 1, n_sources),
            1 + mod(5i + 1, n_sources),
            1 + mod(11i + 3, n_sources),
            1 + mod(17i + 5, n_sources),
            1 + mod(29i + 7, n_sources),
            1 + mod(43i + 11, n_sources),
        ], 6, n_sources, 19i + 9)
        weights = 0.5 .+ rand(rng, length(candidates))
        weights ./= sum(weights)
        for (source, weight) in zip(candidates, weights)
            push!(rows, i)
            push!(cols, source)
            push!(vals, weight)
        end
    end
    return sparse(rows, cols, vals, n_variables, n_sources)
end

struct SeparationOracle
    model::Model
    v::Vector{VariableRef}
    F::Matrix{Float64}
    d::Vector{Float64}
end

function build_separation_oracle(
    F::Matrix{Float64},
    d::Vector{Float64},
    s::Int;
    binary::Union{Bool,Nothing},
)
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, 0 <= v[1:size(F, 2)] <= 1)
    if binary !== nothing
        if binary
            @variable(model, z[1:size(F, 2)], Bin)
        else
            @variable(model, 0 <= z[1:size(F, 2)] <= 1)
        end
        @constraint(model, [i in axes(F, 2)], v[i] <= z[i])
        @constraint(model, sum(z) <= s)
    end
    @constraint(model, F * v .<= d)
    @objective(model, Max, 0.0)
    return SeparationOracle(model, collect(v), F, d)
end

function solve_separation!(
    oracle::SeparationOracle,
    q::Vector{Float64};
    support::Union{Nothing,Vector{Int}} = nothing,
)
    length(q) == length(oracle.v) || error("objective and oracle dimensions differ")
    if support === nothing
        for variable in oracle.v
            set_upper_bound(variable, 1.0)
        end
    else
        selected = falses(length(q))
        selected[support] .= true
        for i in eachindex(oracle.v)
            set_upper_bound(oracle.v[i], selected[i] ? 1.0 : 0.0)
        end
    end
    for i in eachindex(q)
        set_objective_coefficient(oracle.model, oracle.v[i], ERROR_SCALE * abs(q[i]))
    end
    optimize!(oracle.model)
    status = termination_status(oracle.model)
    status in (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) || error("Separation ended with $status")
    v_value = value.(oracle.v)
    u_value = sign.(q) .* v_value
    active_resources = count(oracle.F * v_value .>= oracle.d .- 1.0e-7)
    return (
        value = objective_value(oracle.model),
        u = u_value,
        v = v_value,
        active_resources = active_resources,
    )
end

"""Return the hard-thresholding support, with smaller indices winning ties."""
function largest_magnitude_support(u::Vector{Float64}, s::Int)
    0 <= s <= length(u) || error("invalid support size")
    order = sortperm(collect(eachindex(u)); by = i -> (-abs(u[i]), i))
    return order[1:s]
end

"""Relax the indicators, select a support, and optimize on that fixed support."""
function relaxed_candidate!(
    relaxed_oracle::SeparationOracle,
    fixed_support_oracle::SeparationOracle,
    q::Vector{Float64},
    s::Int,
)
    relaxed = solve_separation!(relaxed_oracle, q)
    support = largest_magnitude_support(relaxed.u, min(s, length(q)))
    fixed = solve_separation!(fixed_support_oracle, q; support = support)
    return (
        value = fixed.value,
        u = fixed.u,
        v = fixed.v,
        active_resources = fixed.active_resources,
        relaxation_upper_bound = relaxed.value,
    )
end

function affine_coefficients(func::AffExpr, variables::Vector{VariableRef})
    position = Dict(index(v) => i for (i, v) in enumerate(variables))
    a = zeros(length(variables))
    for (coef, variable) in linear_terms(func)
        a[position[index(variable)]] += coef
    end
    return a
end

function normalized_row(func::AffExpr, bound::Float64, variables, multiplier::Float64)
    normalized = multiplier * func
    return (
        func = normalized,
        upper = multiplier * bound,
        a = affine_coefficients(normalized, variables),
    )
end

function load_instance(instance::String)
    paths = (joinpath(DATA_DIR, instance * ".SIF"), joinpath(DATA_DIR, instance * ".mps"))
    path_index = findfirst(isfile, paths)
    path_index === nothing && error("No SIF or MPS file found for $instance in $DATA_DIR")
    path = paths[path_index]
    model = read_from_file(path; format = MOI.FileFormats.FORMAT_MPS)
    variables = all_variables(model)
    original_rows = NamedTuple[]
    for constraint in all_constraints(model, AffExpr, MOI.LessThan{Float64})
        object = constraint_object(constraint)
        push!(original_rows, normalized_row(copy(object.func), object.set.upper, variables, 1.0))
    end
    for constraint in all_constraints(model, AffExpr, MOI.GreaterThan{Float64})
        object = constraint_object(constraint)
        push!(original_rows, normalized_row(copy(object.func), object.set.lower, variables, -1.0))
    end
    return model, variables, original_rows
end

function solve_master(instance::String, s::Int, method::String)
    model, variables, rows = load_instance(instance)
    n_sources = SOURCE_COUNT[instance]
    D = calibration_loading(length(variables), n_sources)
    F, d = coupling_matrix(n_sources)
    exact_oracle = build_separation_oracle(F, d, s; binary = true)
    relaxed_oracle = build_separation_oracle(F, d, s; binary = false)
    fixed_support_oracle = build_separation_oracle(F, d, s; binary = nothing)
    set_optimizer(model, Mosek.Optimizer)
    set_silent(model)
    optimize!(model)
    termination_status(model) == MOI.OPTIMAL || error("Nominal $instance is not optimal")
    nominal_objective = objective_value(model)

    start_time = time()
    cuts = 0
    iterations = 0
    for iteration in 1:MAX_ITERATIONS
        iterations = iteration
        x_value = value.(variables)
        new_cuts = 0
        for row in rows
            q = Vector(transpose(D) * (row.a .* x_value))
            separated = method == "EL" ?
                solve_separation!(exact_oracle, q) :
                relaxed_candidate!(relaxed_oracle, fixed_support_oracle, q, s)
            nominal_value = row.func.constant + dot(row.a, x_value)
            if nominal_value + separated.value - row.upper > FEASIBILITY_TOL
                relative_error = ERROR_SCALE .* Vector(D * separated.u)
                expression = row.func.constant + sum(
                    row.a[i] * (1.0 + relative_error[i]) * variables[i]
                    for i in eachindex(variables) if row.a[i] != 0.0
                )
                @constraint(model, expression <= row.upper)
                new_cuts += 1
                cuts += 1
            end
        end
        new_cuts == 0 && break
        optimize!(model)
        termination_status(model) == MOI.OPTIMAL ||
            error("$method master for $instance and s=$s ended with $(termination_status(model))")
    end
    solve_time = time() - start_time
    x_value = value.(variables)

    exact_violations = Float64[]
    relaxation_gaps = Float64[]
    active_counts = Int[]
    for row in rows
        q = Vector(transpose(D) * (row.a .* x_value))
        exact = solve_separation!(exact_oracle, q)
        relaxed = solve_separation!(relaxed_oracle, q)
        nominal_value = row.func.constant + dot(row.a, x_value)
        push!(exact_violations, nominal_value + exact.value - row.upper)
        if exact.value > 1.0e-10
            push!(relaxation_gaps, 100.0 * (relaxed.value - exact.value) / exact.value)
        end
        push!(active_counts, exact.active_resources)
    end

    return (
        instance = instance,
        variables = length(variables),
        inequalities = length(rows),
        sources = n_sources,
        resource_groups = size(F, 1),
        s = s,
        error_scale = ERROR_SCALE,
        method = method,
        nominal_objective = nominal_objective,
        objective = objective_value(model),
        time_s = solve_time,
        iterations = iterations,
        cuts = cuts,
        max_exact_violation = maximum(exact_violations),
        mean_relaxation_gap_pct = isempty(relaxation_gaps) ? 0.0 : mean(relaxation_gaps),
        mean_active_resources = mean(active_counts),
        status = string(termination_status(model)),
    )
end

raw = DataFrame()
for instance in INSTANCES, s in SPARSITY_LEVELS
    for method in METHODS
        method in ("EL", "RL") || error("Unknown method $method; use EL or RL")
        row = solve_master(instance, s, method)
        push!(raw, row)
        @printf(
            "%-6s s=%d %-2s obj=%12.6f time=%7.3fs exact_violation=%9.2e\n",
            instance,
            s,
            method,
            row.objective,
            row.time_s,
            row.max_exact_violation,
        )
    end
end

exact_objective = Dict(
    (row.instance, row.s) => row.objective
    for row in eachrow(raw) if row.method == "EL"
)
raw.gap_to_el_pct = [
    haskey(exact_objective, (row.instance, row.s)) ?
        100.0 * (exact_objective[(row.instance, row.s)] - row.objective) /
        max(abs(exact_objective[(row.instance, row.s)]), 1.0e-8) : missing
    for row in eachrow(raw)
]
raw.objective_change_pct = [
    100.0 * (row.objective - row.nominal_objective) /
    max(abs(row.nominal_objective), 1.0e-8)
    for row in eachrow(raw)
]
CSV.write(joinpath(RESULT_DIR, "implementation_error_coupled_raw.csv"), raw)

summary = combine(
    groupby(raw, [:instance, :variables, :inequalities, :sources, :resource_groups, :s, :method]),
    :objective => mean => :objective_mean,
    :objective_change_pct => mean => :objective_change_mean,
    :time_s => mean => :time_mean,
    :gap_to_el_pct => mean => :gap_to_el_mean,
    :max_exact_violation => maximum => :max_exact_violation,
    :mean_relaxation_gap_pct => mean => :relaxation_gap_mean,
    :mean_active_resources => mean => :active_resources_mean,
)
CSV.write(joinpath(RESULT_DIR, "implementation_error_coupled_summary.csv"), summary)
