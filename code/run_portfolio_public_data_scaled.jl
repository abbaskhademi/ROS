using CSV
using DataFrames
using JuMP
using LinearAlgebra
using MathOptInterface
using MosekTools
using Printf
using Random
using Serialization
using Statistics

include(joinpath(@__DIR__, "SparseAGP.jl"))
using .SparseAGP: agp_maximize, hard_threshold

const MOI = MathOptInterface
const ROOT = normpath(joinpath(@__DIR__, ".."))
const MONTHLY_FILE = joinpath(ROOT, "data", "fama_french", "49_Industry_Portfolios.csv")

function argument_value(flag::String, default::String)
    index = findfirst(==(flag), ARGS)
    return index === nothing ? default : ARGS[index + 1]
end

const OUTPUT_DIR = abspath(argument_value("--outdir", joinpath(ROOT, "reproduced", "portfolio")))
const QUICK_RUN = "--quick" in ARGS
const APPROXIMATE_ONLY = "--approx-only" in ARGS
const EXACT_ONLY = "--exact-only" in ARGS
const EXACT_WORKER_INPUT = argument_value("--exact-worker-input", "")
const EXACT_WORKER_OUTPUT = argument_value("--exact-worker-output", "")
const EXACT_WORKER_STARTED = argument_value("--exact-worker-started", "")
const EXACT_TIME_LIMIT_S = parse(Float64, argument_value("--exact-limit", QUICK_RUN ? "120" : "1000"))
const EXACT_VALIDATION_MAX_DIMENSION = parse(Int, argument_value("--validation-max-dimension", "108"))
const METHOD_TIME_LIMIT_S = 1000.0
const END_YEAR = parse(Int, argument_value("--end-year", "2025"))
const WINDOW_YEARS = parse(Int, argument_value("--window-years", "30"))
const RISK_AVERSION = parse(Float64, argument_value("--risk-aversion", "0.5"))
const DIAGONAL_LOADING = parse(Float64, argument_value(
    "--diagonal-loading", argument_value("--diagonal-shrinkage", "0.01")))
const MONTHLY_TAIL_FRACTION = parse(Float64, argument_value("--monthly-tail-fraction", "-1.0"))
const MONTHLY_SPARSITY_FRACTION = parse(Float64, argument_value("--sparsity-fraction", "0.3333333333333333"))
const FIXED_MONTHLY_SPARSITY = parse(Int, argument_value("--sparsity-budget", "3"))
const CUT_TOLERANCE = 1.0e-5
const EXACT_MIP_RELATIVE_GAP = 1.0e-6
const APPROXIMATE_OUTER_LIMIT = 100
const EXACT_OUTER_LIMIT = 10_000
const INNER_ITERATION_LIMIT = 3000
const INNER_TOLERANCE = 1.0e-5
const START_COUNT = parse(Int, argument_value("--start-count", "15"))
const AGP_GAMMA_INITIAL = parse(Float64, argument_value("--agp-gamma-initial", "0.05"))
const AGP_GAMMA_MINIMUM = parse(Float64, argument_value("--agp-gamma-minimum", "1.0e-4"))
const MONTHLY_PROFILE_COUNTS = MONTHLY_TAIL_FRACTION > 0.0 ?
    [round(Int, 12 * WINDOW_YEARS * MONTHLY_TAIL_FRACTION)] :
    parse.(Int, split(argument_value(
        "--profile-counts", QUICK_RUN ? "24" : "24,36,54,90,108"), ','))

mkpath(OUTPUT_DIR)

function read_french_table(path::String, marker_text::String, date_digits::Int)
    lines = readlines(path)
    marker = findfirst(line -> occursin(marker_text, line), lines)
    isnothing(marker) && error("Requested return table was not found in $path")
    names = strip.(split(lines[marker + 1], ','))[2:end]
    dates = Int[]
    observations = Vector{Vector{Float64}}()
    date_pattern = date_digits == 6 ? r"^\s*\d{6}," : r"^\s*\d{8},"
    for line in lines[(marker + 2):end]
        occursin(date_pattern, line) || break
        fields = strip.(split(line, ','))
        values = parse.(Float64, fields[2:end])
        any(value -> value <= -99.0, values) && continue
        push!(dates, parse(Int, fields[1]))
        push!(observations, values ./ 100.0)
    end
    return dates, names, reduce(vcat, permutedims.(observations))
end

function empirical_penalty(profiles::Matrix{Float64})
    gram = profiles * transpose(profiles) / size(profiles, 2)
    loading = max.(DIAGONAL_LOADING .* diag(gram), 1.0e-10)
    return Matrix(Symmetric(gram + Diagonal(loading))), loading
end

function monthly_instance(dates, returns, window_years::Int, n_stress::Int)
    start_date = 100 * (END_YEAR - window_years + 1) + 1
    end_date = 100 * END_YEAR + 12
    rows = findall(date -> start_date <= date <= end_date, dates)
    length(rows) == 12 * window_years || error("Incomplete monthly estimation window")
    sample = returns[rows, :]
    center = vec(mean(sample; dims = 1))
    mu = 12.0 .* center
    Sigma = Matrix(Symmetric(12.0 .* cov(sample; dims = 1) + 1.0e-8I))
    market_return = vec(mean(sample; dims = 2))
    1 <= n_stress <= length(rows) || error("Invalid number of historical profiles")
    stress_rows = partialsortperm(market_return, 1:n_stress)
    profiles = max.(permutedims(center) .- sample[stress_rows, :], 0.0)
    Q, loading = empirical_penalty(profiles)
    return mu, Sigma, profiles, Q, loading, dates[rows[stress_rows]]
end

stress_value(u, w, Q) = 2.0 * dot(u, w) - dot(u, Q * u)

function optimize_fixed_support(u, w, Q, s::Int)
    support = partialsortperm(abs.(u), 1:min(s, length(u)); rev = true)
    optimized = zeros(length(u))
    optimized[support] .= cholesky(Symmetric(Q[support, support])) \ w[support]
    return stress_value(optimized, w, Q), optimized
end

function common_starts(w, Q, s::Int)
    starts = [zeros(length(w))]
    rng = MersenneTwister(31_000 + length(w) + s)
    for _ in 2:START_COUNT
        support = sort(randperm(rng, length(w))[1:s])
        start = zeros(length(w))
        start[support] .= cholesky(Symmetric(Q[support, support])) \ w[support]
        push!(starts, start)
    end
    return starts
end

function approximate_separation(w, Q, s::Int, method::String)
    best_value = -Inf
    best_point = zeros(length(w))
    points_by_support = Dict{Tuple{Vararg{Int}},Vector{Float64}}()
    total_inner_iterations = 0
    iht_step = 1.0 / (2.0 * eigmax(Symmetric(Q)))

    for start in common_starts(w, Q, s)
        if method == "AGP"
            value_function(u) = stress_value(u, w, Q)
            gradient_function(u) = 2.0 .* (w - Q * u)
            result = agp_maximize(value_function, gradient_function, s, start,
                INNER_ITERATION_LIMIT;
                gamma_initial = AGP_GAMMA_INITIAL, gamma_minimum = AGP_GAMMA_MINIMUM,
                delta = 1.0e-10, beta = 2.0,
                epsilon = INNER_TOLERANCE, initial_seed = 23,
                adapt_gamma = true)
            candidate_value, candidate_point = optimize_fixed_support(result.point, w, Q, s)
            total_inner_iterations += result.iterations
        elseif method == "IHT"
            u = hard_threshold(start, s)
            used_iterations = 0
            for iteration in 1:INNER_ITERATION_LIMIT
                used_iterations = iteration
                candidate = hard_threshold(u + 2.0 .* iht_step .* (w - Q * u), s)
                if norm(candidate - u) <= INNER_TOLERANCE
                    u = candidate
                    break
                end
                u = candidate
            end
            candidate_value, candidate_point = optimize_fixed_support(u, w, Q, s)
            total_inner_iterations += used_iterations
        else
            error("Unknown sparse method $method")
        end

        if candidate_value > best_value
            best_value = candidate_value
            best_point = candidate_point
        end
        support = Tuple(findall(value -> abs(value) > 1.0e-8, candidate_point))
        points_by_support[support] = candidate_point
    end
    return best_value, best_point, collect(values(points_by_support)), total_inner_iterations
end

function exact_separation(w, Q, s::Int; time_limit_s::Float64, start = nothing)
    n = length(w)
    radius = 2.0 * norm(w) / eigmin(Symmetric(Q))
    model = Model(Mosek.Optimizer)
    set_silent(model)
    separator_limit = max(time_limit_s, 1.0e-3)
    set_time_limit_sec(model, separator_limit)
    set_optimizer_attribute(model, "MSK_DPAR_MIO_MAX_TIME", separator_limit)
    set_optimizer_attribute(model, "MSK_DPAR_MIO_TOL_REL_GAP", EXACT_MIP_RELATIVE_GAP)
    @variable(model, z[1:n], Bin)
    @variable(model, -radius <= u[1:n] <= radius)
    @constraint(model, sum(z) <= s)
    @constraint(model, [radius; u] in SecondOrderCone())
    @constraint(model, [i = 1:n], u[i] <= radius * z[i])
    @constraint(model, [i = 1:n], u[i] >= -radius * z[i])
    @objective(model, Max,
        2.0 * dot(w, u) - sum(Q[i, j] * u[i] * u[j] for i in 1:n, j in 1:n))
    if start !== nothing
        for i in 1:n
            set_start_value(u[i], start[i])
            set_start_value(z[i], abs(start[i]) > 1.0e-8 ? 1.0 : 0.0)
        end
    end
    elapsed = @elapsed optimize!(model)
    has_solution = has_values(model)
    return (
        status = string(termination_status(model)),
        elapsed_s = elapsed,
        incumbent = has_solution ? objective_value(model) : NaN,
        bound = objective_bound(model),
        relative_gap = relative_gap(model),
        nodes = node_count(model),
        point = has_solution ? value.(u) : zeros(n),
    )
end

"""Exact separation for a fixed decision when the sparsity budget is three.

For a support `S`, strict positive definiteness of `Q[S,S]` gives the unique
maximizer `u[S] = Q[S,S] \\ w[S]` and value `w[S]'u[S]`. Enumerating all
three-element supports also covers solutions with fewer than three nonzeros,
because every smaller coordinate subspace is contained in a three-coordinate
subspace.
"""
function exact_separation_by_supports(w, Q, s::Int)
    s == 3 || error("support enumeration is implemented for s=3")
    n = length(w)
    n >= s || error("the vector dimension must be at least s")
    start_time = time()
    best_value = -Inf
    best_point = zeros(n)
    support_count = 0
    for i in 1:(n - 2), j in (i + 1):(n - 1), k in (j + 1):n
        support = [i, j, k]
        point_on_support = cholesky(Symmetric(Q[support, support])) \ w[support]
        candidate_value = dot(w[support], point_on_support)
        support_count += 1
        if candidate_value > best_value
            best_value = candidate_value
            fill!(best_point, 0.0)
            best_point[support] .= point_on_support
        end
    end
    return (
        status = "OPTIMAL",
        elapsed_s = time() - start_time,
        incumbent = best_value,
        bound = best_value,
        relative_gap = 0.0,
        nodes = support_count,
        point = best_point,
    )
end

function portfolio_master(mu, Sigma)
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, x[1:length(mu)] >= 0.0)
    @variable(model, t >= 0.0)
    @constraint(model, sum(x) == 1.0)
    @objective(model, Max,
        dot(mu, x) - RISK_AVERSION *
        sum(Sigma[i, j] * x[i] * x[j]
            for i in eachindex(mu), j in eachindex(mu)) - t)
    return model, x, t
end

function solve_exact_portfolio(mu, Sigma, profiles, Q, s::Int)
    model, x, t = portfolio_master(mu, Sigma)
    start_time = time()
    cuts = 0
    last_point = nothing
    last_separator = nothing
    for iteration in 1:EXACT_OUTER_LIMIT
        optimize!(model)
        termination_status(model) == MOI.OPTIMAL || error("Exact portfolio master failed")
        remaining = EXACT_TIME_LIMIT_S - (time() - start_time)
        if remaining <= 0.0
            return (status = "TIME_LIMIT", time_s = time() - start_time,
                iterations = iteration - 1, cuts = cuts, objective = NaN,
                separator_incumbent = NaN, separator_bound = NaN, separator_gap = NaN)
        end
        x_value = value.(x)
        separator = exact_separation(profiles * x_value, Q, s;
            time_limit_s = remaining, start = last_point)
        last_separator = separator
        if separator.status != "OPTIMAL"
            return (status = "TIME_LIMIT", time_s = time() - start_time,
                iterations = iteration, cuts = cuts, objective = NaN,
                separator_incumbent = separator.incumbent,
                separator_bound = separator.bound,
                separator_gap = separator.relative_gap)
        end
        last_point = separator.point
        current_t = value(t)
        if separator.incumbent <= current_t + CUT_TOLERANCE
            return (status = "OPTIMAL", time_s = time() - start_time,
                iterations = iteration, cuts = cuts, objective = objective_value(model),
                separator_incumbent = separator.incumbent,
                separator_bound = separator.bound,
                separator_gap = separator.relative_gap)
        end
        @constraint(model,
            t >= dot(2.0 .* (transpose(last_point) * profiles)[:], x) -
                 dot(last_point, Q * last_point))
        cuts += 1
    end
    return (status = "ITERATION_LIMIT", time_s = time() - start_time,
        iterations = EXACT_OUTER_LIMIT, cuts = cuts, objective = NaN,
        separator_incumbent = last_separator === nothing ? NaN : last_separator.incumbent,
        separator_bound = last_separator === nothing ? NaN : last_separator.bound,
        separator_gap = last_separator === nothing ? NaN : last_separator.relative_gap)
end

function solve_exact_portfolio_with_hard_limit(mu, Sigma, profiles, Q, s::Int)
    worker_directory = mktempdir()
    input_path = joinpath(worker_directory, "input.jls")
    output_path = joinpath(worker_directory, "output.jls")
    started_path = joinpath(worker_directory, "started")
    serialize(input_path, (mu = mu, Sigma = Sigma, profiles = profiles, Q = Q, s = s))

    project_directory = dirname(Base.active_project())
    command = `$(Base.julia_cmd()) --project=$project_directory $(@__FILE__) --outdir $worker_directory --exact-limit $EXACT_TIME_LIMIT_S --exact-worker-input $input_path --exact-worker-output $output_path --exact-worker-started $started_path`
    process = run(command; wait = false)

    try
        startup_deadline = time() + 120.0
        while !isfile(started_path) && !process_exited(process) && time() < startup_deadline
            sleep(0.05)
        end
        if !isfile(started_path)
            !process_exited(process) && Base.kill(process, Base.SIGKILL)
            wait(process)
            error("The exact portfolio worker did not start correctly")
        end

        hard_deadline = time() + EXACT_TIME_LIMIT_S + 2.0
        while !process_exited(process) && time() < hard_deadline
            sleep(0.10)
        end
        if !process_exited(process)
            Base.kill(process, Base.SIGKILL)
            wait(process)
            return (
                status = "TIME_LIMIT", time_s = EXACT_TIME_LIMIT_S,
                iterations = 0, cuts = 0, objective = NaN,
                separator_incumbent = NaN, separator_bound = NaN,
                separator_gap = NaN,
            )
        end

        wait(process)
        success(process) || error("The exact portfolio worker failed")
        isfile(output_path) || error("The exact portfolio worker returned no result")
        return deserialize(output_path)
    finally
        !process_exited(process) && Base.kill(process, Base.SIGKILL)
        !process_exited(process) && wait(process)
        rm(worker_directory; recursive = true, force = true)
    end
end

function solve_approximate_portfolio(mu, Sigma, profiles, Q, s::Int,
    method::String; validate::Bool)
    model, x, t = portfolio_master(mu, Sigma)
    start_time = time()
    cuts = 0
    status = "ITERATION_LIMIT"
    outer_iterations = 0
    total_inner_iterations = 0
    final_violation = NaN
    for iteration in 1:APPROXIMATE_OUTER_LIMIT
        outer_iterations = iteration
        if time() - start_time >= METHOD_TIME_LIMIT_S
            status = "TIME_LIMIT"
            break
        end
        optimize!(model)
        termination_status(model) == MOI.OPTIMAL || error("Approximate portfolio master failed")
        x_value = value.(x)
        w = profiles * x_value
        stress, point, scenario_points, inner_iterations =
            approximate_separation(w, Q, s, method)
        total_inner_iterations += inner_iterations
        current_t = value(t)
        final_violation = stress - current_t
        if final_violation <= CUT_TOLERANCE
            status = "CONVERGED"
            break
        end
        for scenario in scenario_points
            stress_value(scenario, w, Q) <= current_t + CUT_TOLERANCE && continue
            @constraint(model,
                t >= dot(2.0 .* (transpose(scenario) * profiles)[:], x) -
                     dot(scenario, Q * scenario))
            cuts += 1
        end
    end
    optimize!(model)
    termination_status(model) == MOI.OPTIMAL || error("Final approximate portfolio master failed")
    method_time = time() - start_time
    x_value = value.(x)
    w = profiles * x_value
    reported_stress, reported_point, _, inner_iterations =
        approximate_separation(w, Q, s, method)
    total_inner_iterations += inner_iterations
    risk = dot(x_value, Sigma * x_value)
    nominal_part = dot(mu, x_value) - RISK_AVERSION * risk
    local_objective = nominal_part - reported_stress

    validation = validate ? (s == 3 ? exact_separation_by_supports(w, Q, s) :
        exact_separation(w, Q, s;
            time_limit_s = EXACT_TIME_LIMIT_S, start = reported_point)) : nothing
    validated_objective = validate && validation.status == "OPTIMAL" ?
        nominal_part - validation.incumbent : NaN
    shortfall = validate && validation.status == "OPTIMAL" ?
        100.0 * (validation.incumbent - reported_stress) /
        max(abs(validation.incumbent), 1.0e-12) : NaN
    return (
        status = status,
        time_s = method_time,
        iterations = outer_iterations,
        cuts = cuts,
        total_inner_iterations = total_inner_iterations,
        final_violation = final_violation,
        local_objective = local_objective,
        validated_objective = validated_objective,
        reported_stress = reported_stress,
        exact_stress = validate ? validation.incumbent : NaN,
        separation_shortfall_pct = shortfall,
        validation_status = validate ? validation.status : "NOT_RUN",
        validation_time_s = validate ? validation.elapsed_s : NaN,
        validation_bound = validate ? validation.bound : NaN,
        validation_gap = validate ? validation.relative_gap : NaN,
        decision = x_value,
        nominal_part = nominal_part,
    )
end

function warmup_paths()
    Q = [2.0 0.2; 0.2 1.0]
    w = [0.4, 0.2]
    approximate_separation(w, Q, 1, "AGP")
    approximate_separation(w, Q, 1, "IHT")
    model = Model(Mosek.Optimizer)
    set_silent(model)
    @variable(model, x >= 0.0)
    @objective(model, Min, (x - 1.0)^2)
    optimize!(model)
end

if !isempty(EXACT_WORKER_INPUT)
    isempty(EXACT_WORKER_OUTPUT) && error("Missing exact worker output path")
    isempty(EXACT_WORKER_STARTED) && error("Missing exact worker start path")
    worker_data = deserialize(EXACT_WORKER_INPUT)
    open(EXACT_WORKER_STARTED, "w") do stream
        write(stream, "started")
    end
    worker_result = solve_exact_portfolio(
        worker_data.mu, worker_data.Sigma, worker_data.profiles, worker_data.Q,
        worker_data.s)
    serialize(EXACT_WORKER_OUTPUT, worker_result)
    exit()
end

monthly_dates, industry_names, monthly_returns = read_french_table(
    MONTHLY_FILE, "Average Value Weighted Returns -- Monthly", 6)
length(industry_names) == 49 || error("Expected 49 industry portfolios")
warmup_paths()

outer_rows = NamedTuple[]
for n_stress_requested in MONTHLY_PROFILE_COUNTS
    window_years = WINDOW_YEARS
    mu, Sigma, profiles, Q, loading, stress_dates =
        monthly_instance(monthly_dates, monthly_returns, window_years, n_stress_requested)
    n_stress = size(profiles, 1)
    s = FIXED_MONTHLY_SPARSITY > 0 ? FIXED_MONTHLY_SPARSITY :
        max(1, round(Int, MONTHLY_SPARSITY_FRACTION * n_stress))
    if EXACT_ONLY
        exact = solve_exact_portfolio_with_hard_limit(mu, Sigma, profiles, Q, s)
        push!(outer_rows, (
            window_years = window_years,
            tail_fraction = n_stress / (12.0 * window_years),
            n_stress = n_stress,
            s = s,
            method = "Exact",
            objective = exact.objective,
            local_objective = exact.objective,
            gap_to_exact_pct = exact.status == "OPTIMAL" ? 0.0 : NaN,
            time_s = exact.time_s,
            requested_time_limit_s = EXACT_TIME_LIMIT_S,
            inner_iteration_limit = 0,
            status = exact.status,
            iterations = exact.iterations,
            cuts = exact.cuts,
            reported_stress = exact.separator_incumbent,
            exact_stress = exact.separator_incumbent,
            separation_shortfall_pct = 0.0,
            validation_status = exact.status,
            validation_time_s = NaN,
            validation_bound = exact.separator_bound,
            validation_gap = exact.separator_gap,
            total_inner_iterations = 0,
            stress_dates = join(stress_dates, ";"),
        ))
        CSV.write(joinpath(OUTPUT_DIR, "portfolio_public_data_checkpoint.csv"),
            DataFrame(outer_rows))
        continue
    end
    method_results = Dict{String,Any}()
    for method in ("AGP", "IHT")
        method_results[method] = solve_approximate_portfolio(
            mu, Sigma, profiles, Q, s, method;
            validate = n_stress <= EXACT_VALIDATION_MAX_DIMENSION)
    end
    postchecks = Dict{String,Any}()
    for method in ("AGP", "IHT")
        result = method_results[method]
        if n_stress <= EXACT_VALIDATION_MAX_DIMENSION
            postchecks[method] = (
                objective = result.validated_objective,
                stress = result.exact_stress,
                shortfall_pct = result.separation_shortfall_pct,
                status = result.validation_status,
                time_s = result.validation_time_s,
                bound = result.validation_bound,
                gap = result.validation_gap,
            )
        else
            w = profiles * result.decision
            check_start = time()
            agp_stress, _, _, _ = approximate_separation(w, Q, s, "AGP")
            iht_stress, _, _, _ = approximate_separation(w, Q, s, "IHT")
            checked_stress = max(agp_stress, iht_stress)
            postchecks[method] = (
                objective = result.nominal_part - checked_stress,
                stress = checked_stress,
                shortfall_pct = 100.0 * (checked_stress - result.reported_stress) /
                    max(abs(checked_stress), 1.0e-12),
                status = "AGP_IHT_CROSS_CHECK",
                time_s = time() - check_start,
                bound = NaN,
                gap = NaN,
            )
        end
    end
    exact = APPROXIMATE_ONLY ? (
        status = "NOT_RUN", time_s = 0.0, iterations = 0, cuts = 0,
        objective = NaN, separator_incumbent = NaN,
        separator_bound = NaN, separator_gap = NaN,
    ) : solve_exact_portfolio_with_hard_limit(mu, Sigma, profiles, Q, s)
    method_results["Exact"] = exact
    exact_objective = exact.status == "OPTIMAL" ? exact.objective : NaN
    for method in ("Exact", "AGP", "IHT")
        result = method_results[method]
        check = method == "Exact" ? nothing : postchecks[method]
        objective = method == "Exact" ? result.objective :
            (isfinite(check.objective) ? check.objective : result.local_objective)
        gap = isfinite(exact_objective) && isfinite(objective) ?
            100.0 * (exact_objective - objective) /
            max(abs(exact_objective), 1.0e-12) : NaN
        push!(outer_rows, (
            window_years = window_years,
            tail_fraction = n_stress / (12.0 * window_years),
            n_stress = n_stress,
            s = s,
            method = method,
            objective = objective,
            local_objective = method == "Exact" ? result.objective : result.local_objective,
            gap_to_exact_pct = gap,
            time_s = result.time_s,
            requested_time_limit_s = method == "Exact" ?
                EXACT_TIME_LIMIT_S : METHOD_TIME_LIMIT_S,
            inner_iteration_limit = method == "Exact" ? 0 : INNER_ITERATION_LIMIT,
            status = result.status,
            iterations = result.iterations,
            cuts = result.cuts,
            reported_stress = method == "Exact" ?
                result.separator_incumbent : result.reported_stress,
            exact_stress = method == "Exact" ?
                result.separator_incumbent : check.stress,
            separation_shortfall_pct = method == "Exact" ?
                0.0 : check.shortfall_pct,
            validation_status = method == "Exact" ?
                result.status : check.status,
            validation_time_s = method == "Exact" ?
                NaN : check.time_s,
            validation_bound = method == "Exact" ?
                result.separator_bound : check.bound,
            validation_gap = method == "Exact" ?
                result.separator_gap : check.gap,
            total_inner_iterations = method == "Exact" ?
                0 : result.total_inner_iterations,
            stress_dates = join(stress_dates, ";"),
        ))
        @printf(
            "monthly n=%d s=%d %-5s status=%-10s obj=% .6f gap=%7.3f time=%8.3f\n",
            n_stress, s, method, result.status, objective, gap, result.time_s)
    end
    CSV.write(
        joinpath(OUTPUT_DIR, "portfolio_public_data_checkpoint.csv"),
        DataFrame(outer_rows),
    )
end

outer = DataFrame(outer_rows)
outer.agp_improvement_vs_iht_pct = fill(NaN, nrow(outer))
if !EXACT_ONLY
    for q in unique(outer.n_stress)
        block = outer[outer.n_stress .== q, :]
        agp_objective = only(block[block.method .== "AGP", :objective])
        iht_objective = only(block[block.method .== "IHT", :objective])
        improvement = 100.0 * (agp_objective - iht_objective) /
                      max(abs(iht_objective), 1.0e-12)
        outer[outer.n_stress .== q, :agp_improvement_vs_iht_pct] .= improvement
    end
end
CSV.write(joinpath(OUTPUT_DIR, "portfolio_public_data_raw.csv"), outer)

summary = select(outer,
    :window_years, :tail_fraction, :n_stress, :s, :method,
    :objective, :local_objective, :gap_to_exact_pct, :time_s,
    :requested_time_limit_s, :inner_iteration_limit,
    :status, :iterations, :cuts, :separation_shortfall_pct,
    :validation_status, :validation_gap, :agp_improvement_vs_iht_pct)
CSV.write(joinpath(OUTPUT_DIR, "portfolio_public_data_summary.csv"), summary)

metadata = DataFrame(
    sample_start = fill(@sprintf("%04d-01", END_YEAR - WINDOW_YEARS + 1),
        length(MONTHLY_PROFILE_COUNTS)),
    sample_end = fill(@sprintf("%04d-12", END_YEAR), length(MONTHLY_PROFILE_COUNTS)),
    observations = fill(12 * WINDOW_YEARS, length(MONTHLY_PROFILE_COUNTS)),
    assets = fill(length(industry_names), length(MONTHLY_PROFILE_COUNTS)),
    n_stress = MONTHLY_PROFILE_COUNTS,
    tail_fraction = MONTHLY_PROFILE_COUNTS ./ (12.0 * WINDOW_YEARS),
    sparsity_budget = fill(FIXED_MONTHLY_SPARSITY, length(MONTHLY_PROFILE_COUNTS)),
    risk_aversion = fill(RISK_AVERSION, length(MONTHLY_PROFILE_COUNTS)),
    diagonal_loading = fill(DIAGONAL_LOADING, length(MONTHLY_PROFILE_COUNTS)),
    common_starts = fill(START_COUNT, length(MONTHLY_PROFILE_COUNTS)),
    agp_gamma_initial = fill(AGP_GAMMA_INITIAL, length(MONTHLY_PROFILE_COUNTS)),
    agp_gamma_minimum = fill(AGP_GAMMA_MINIMUM, length(MONTHLY_PROFILE_COUNTS)),
    inner_iteration_limit = fill(INNER_ITERATION_LIMIT, length(MONTHLY_PROFILE_COUNTS)),
    tolerance = fill(INNER_TOLERANCE, length(MONTHLY_PROFILE_COUNTS)),
    requested_time_limit_s = fill(EXACT_TIME_LIMIT_S, length(MONTHLY_PROFILE_COUNTS)),
    fixed_decision_evaluation = fill(
        FIXED_MONTHLY_SPARSITY == 3 ? "support_enumeration" : "mixed_integer_separation",
        length(MONTHLY_PROFILE_COUNTS)),
)
CSV.write(joinpath(OUTPUT_DIR, "portfolio_instance_metadata.csv"), metadata)

println("Portfolio results written to $OUTPUT_DIR")
