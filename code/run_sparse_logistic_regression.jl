# Run with the package environment supplied in code/Project.toml.
using LIBSVMdata
using LinearAlgebra
using SparseArrays
using Arpack
using Printf
using DataFrames
using CSV
using Statistics
using Random

function argument_value(flag::String, default::String)
    position = findfirst(==(flag), ARGS)
    return position === nothing ? default : ARGS[position + 1]
end
const OUTPUT_DIR = abspath(argument_value("--outdir", joinpath(@__DIR__, "..", "reproduced", "sparse_logistic_regression")))
const DATA_DIR = abspath(argument_value("--data-dir", joinpath(@__DIR__, "..", "data", "libsvm")))
const QUICK_RUN = "--quick" in ARGS
const RESUME_RUN = "--resume" in ARGS
mkpath(OUTPUT_DIR)
mkpath(DATA_DIR)
ENV["LIBSVMDATA_HOME"] = DATA_DIR
output_path(name::String) = joinpath(OUTPUT_DIR, name)

# IHT function
function IHT(f, g, s, L, x0, N; epsilon=1e-5)
    start_time = time()
    n = length(x0)
    x = copy(x0)
    d = count(!iszero, x)
    if d > s
        perm = sortperm(abs.(x), by=abs)
        x_new = zeros(n)
        x_new[perm[(n-s+1):n]] = x[perm[(n-s+1):n]]
        x = x_new
    end
    iters = N
    for i in 1:N
        x_prev = copy(x)
        x = x .- (1/L) * g(x)
        perm = sortperm(abs.(x), by=abs)
        x_new = zeros(n)
        x_new[perm[(n-s+1):n]] = x[perm[(n-s+1):n]]
        x = x_new
        fx = f(x)
        if i % 500 == 0
            @printf("IHT: iter = %5d, f(x) = %5.4f\n", i, fx)
        end
        if norm(x - x_prev) <= epsilon
            @printf("Stopped IHT early at iter %d: ||x_%d - x_%d|| = %.2e <= %.2e\n",
                    i, i, i-1, norm(x - x_prev), epsilon)
            iters = i
            @printf("IHT: iter = %5d, f(x) = %5.4f\n", i, fx)
            break
        end
    end
    fx = f(x)
    elapsed_time = time() - start_time
    println("  Elapsed time: $(round(elapsed_time, digits=4)) seconds")
    return x, fx, elapsed_time, iters
end

# Adaptive IHT function
function adaptive_IHT(f, g, s, x0, N; gamma=1/4, delta=1e-10, beta=2.0, epsilon=1e-5)
    start_time = time()
    n = length(x0)
    x_curr = copy(x0)
    fx = f(x_curr)
    current_grad = g(x_curr)
    iterations = 0
    fx_values = Float64[]
    push!(fx_values, fx)
    L_ks = Float64[]
    backtrack_counts = Int[]
    gamma_history = [gamma]
    recent_backtracks = Int[]
    L_k = 0.0

    d = count(!iszero, x_curr)
    if d > s
        perm = sortperm(abs.(x_curr), rev=true)
        x_new = zeros(n)
        x_new[perm[1:s]] = x_curr[perm[1:s]]
        x_curr = x_new
        fx = f(x_curr)
        current_grad = g(x_curr)
        fx_values[1] = fx
    end

    x_prev = copy(x_curr)
    prev_grad = copy(current_grad)

    for k in 1:N
        if k == 1
            rng = Xoshiro(23)
            d0 = randn(rng, n)
            dx = 1e-3 * d0
            grad_perturbed = g(x_curr + dx)
            grad_diff_norm = norm(current_grad - grad_perturbed)
            dx_norm = norm(dx)
            L_k = gamma * (grad_diff_norm / dx_norm + delta)
        else
            grad_diff_norm = norm(current_grad - prev_grad)
            x_diff_norm = norm(x_curr - x_prev)
            if x_diff_norm < 1e-10
                break
            end
            L_k = gamma * (grad_diff_norm / x_diff_norm + delta)
        end
        push!(L_ks, L_k)

        descent_ok = false
        backtrack_count = 0
        x_next = copy(x_curr)
        while !descent_ok
            x_candidate = x_curr .- (1 / L_k) * current_grad
            perm = sortperm(abs.(x_candidate), rev=true)
            x_new = zeros(n)
            x_new[perm[1:s]] = x_candidate[perm[1:s]]
            x_next = x_new

            f_next = f(x_next)
            dx_step = x_next - x_curr
            grad_inner = dot(current_grad, dx_step)
            residual = (L_k / 2) * norm(dx_step)^2

            if f_next <= fx + grad_inner + residual
                descent_ok = true
            else
                L_k *= beta
                backtrack_count += 1
            end
        end

        x_prev = copy(x_curr)
        prev_grad = copy(current_grad)
        x_curr = x_next
        fx = f(x_next)
        current_grad = g(x_curr)
        iterations = k
        push!(fx_values, fx)
        push!(backtrack_counts, backtrack_count)
        push!(recent_backtracks, backtrack_count)

        if k % 10 == 0
            total_backtracks = sum(recent_backtracks)
            if total_backtracks == 0
                gamma = max(1e-4, gamma * 0.9)
            elseif total_backtracks > 10
                gamma = min(1, gamma * 1.1)
            end
            push!(gamma_history, gamma)
            recent_backtracks = Int[]
        end

        if k % 500 == 0
            @printf("adaptive_IHT: iter = %5d, f(x) = %5.4f\n", k, fx)
        end

        if norm(x_curr - x_prev) <= epsilon
            @printf("Adaptive IHT stopped early: ||x_%d - x_%d|| = %.2e <= %.2e\n",
                    k, k-1, norm(x_curr - x_prev), epsilon)
            iterations = k
            @printf("adaptive_IHT: iter = %5d, f(x) = %5.4f\n", k, fx)
            break
        end
    end

    elapsed_time = time() - start_time
    println("  Elapsed time: $(round(elapsed_time, digits=4)) seconds")
    return x_curr, fx, elapsed_time, iterations, fx_values, L_ks, backtrack_counts, gamma_history
end

softplus(value::Float64) =
    value > 0.0 ? value + log1p(exp(-value)) : log1p(exp(value))

function logistic_value(scores::Vector{Float64})
    return mean(softplus(score) for score in scores)
end

function logistic_sigmoid(value::Float64)
    if value >= 0.0
        inverse = exp(-value)
        return 1.0 / (1.0 + inverse)
    end
    exponential = exp(value)
    return exponential / (1.0 + exponential)
end

"""Globally minimize a convex logistic loss along one coordinate.

The derivative is monotone. We first bracket its zero and then use bisection,
which solves the true one-dimensional logistic problem used by PSS.
"""
function logistic_line_minimum(
    scores::Vector{Float64},
    direction::Vector{Float64};
    derivative_tolerance::Float64 = 1.0e-10,
    interval_tolerance::Float64 = 1.0e-10,
)
    length(scores) == length(direction) || error("line-search vectors must have equal length")

    function derivative(step::Float64)
        total = 0.0
        @inbounds for row in eachindex(scores)
            total += direction[row] *
                     logistic_sigmoid(scores[row] + step * direction[row])
        end
        return total / length(scores)
    end

    derivative_at_zero = derivative(0.0)
    if abs(derivative_at_zero) <= derivative_tolerance
        return 0.0, logistic_value(scores)
    end

    lower = 0.0
    upper = 0.0
    bracketed = false
    if derivative_at_zero > 0.0
        lower = -1.0
        upper = 0.0
        for _ in 1:60
            if derivative(lower) <= 0.0
                bracketed = true
                break
            end
            lower *= 2.0
        end
    else
        lower = 0.0
        upper = 1.0
        for _ in 1:60
            if derivative(upper) >= 0.0
                bracketed = true
                break
            end
            upper *= 2.0
        end
    end
    bracketed || error("the logistic coordinate problem has no finite minimizer")

    step = 0.5 * (lower + upper)
    for _ in 1:100
        step = 0.5 * (lower + upper)
        derivative_at_step = derivative(step)
        if abs(derivative_at_step) <= derivative_tolerance ||
           upper - lower <= interval_tolerance * max(1.0, abs(step))
            break
        elseif derivative_at_step > 0.0
            upper = step
        else
            lower = step
        end
    end
    candidate_scores = scores .+ step .* direction
    return step, logistic_value(candidate_scores)
end

function logistic_coordinate_minimum(A, y, scores, coordinate::Int)
    direction = -y .* Vector(A[:, coordinate])
    step, value = logistic_line_minimum(scores, direction)
    return step, value, direction
end

function partial_sparse_simplex_logistic(A, y, f, grad_f, s, N, x0; epsilon=1e-5)
    start_time = time()
    n = length(x0)
    1 <= s <= n || error("s must lie between one and the vector dimension")

    order = sortperm(collect(1:n); by = i -> (-abs(x0[i]), i))
    x = zeros(Float64, n)
    x[order[1:s]] .= x0[order[1:s]]
    scores = -y .* Vector(A * x)
    fx = f(x)
    iters = N

    for k in 1:N
        x_prev = copy(x)
        previous_value = fx
        supp = findall(!iszero, x)
        nonsupp = setdiff(1:n, supp)

        if length(supp) < s
            best_value = Inf
            best_step = 0.0
            best_coordinate = 0
            best_direction = zeros(length(y))
            for i in 1:n
                step, candidate_value, direction =
                    logistic_coordinate_minimum(A, y, scores, i)
                if candidate_value < best_value
                    best_value = candidate_value
                    best_step = step
                    best_coordinate = i
                    best_direction = direction
                end
            end
            if best_value < fx
                x[best_coordinate] += best_step
                scores .+= best_step .* best_direction
                fx = best_value
            else
                iters = k
                break
            end
        else
            # The intended PSS rule selects the support coordinate giving the
            # largest decrease, consistent with the paper's prose and proof.
            D1 = Inf
            T1 = 0.0
            i1 = 0
            direction1 = zeros(length(y))
            for i in supp
                step, candidate_value, direction =
                    logistic_coordinate_minimum(A, y, scores, i)
                if candidate_value < D1
                    D1 = candidate_value
                    T1 = step
                    i1 = i
                    direction1 = direction
                end
            end

            if isempty(nonsupp)
                x[i1] += T1
                scores .+= T1 .* direction1
                fx = D1
            else
                gradient = grad_f(x)
                m_idx = supp[argmin(abs.(x[supp]))]
                i2_local_idx = argmax(abs.(gradient[nonsupp]))
                i2 = nonsupp[i2_local_idx]
                removal_direction = -y .* Vector(A[:, m_idx])
                swap_scores = scores .- x[m_idx] .* removal_direction
                T2, D2, direction2 =
                    logistic_coordinate_minimum(A, y, swap_scores, i2)

                if D1 < D2
                    x[i1] += T1
                    scores .+= T1 .* direction1
                    fx = D1
                else
                    x[m_idx] = 0.0
                    x[i2] += T2
                    scores = swap_scores .+ T2 .* direction2
                    fx = D2
                end
            end
        end

        evaluated_value = f(x)
        isapprox(evaluated_value, fx; atol = 1.0e-9, rtol = 1.0e-9) ||
            error("PSS line value and evaluated logistic loss disagree")
        fx = evaluated_value
        fx <= previous_value + 1.0e-9 * max(1.0, abs(previous_value)) ||
            error("PSS increased the objective from $previous_value to $fx")

        if k % 500 == 0
            @printf("PSS: iter = %5d, f(x) = %5.4f\n", k, fx)
        end

        if norm(x - x_prev) <= epsilon
            @printf("Stopped PSS early at iter %d: ||x_%d - x_%d|| = %.2e <= %.2e\n",
                    k, k, k-1, norm(x - x_prev), epsilon)
            iters = k
            break
        end
    end

    elapsed_time = time() - start_time
    println("  Elapsed time: $(round(elapsed_time, digits=4)) seconds")
    return x, fx, elapsed_time, iters
end

# Problem Generation
function load_logistic_instance(dataset_name; seed=nothing)
    rng = isnothing(seed) ? Random.default_rng() : Xoshiro(seed)
    println("Loading dataset: $dataset_name")
    dataset_path = joinpath(DATA_DIR, dataset_name)
    isfile(dataset_path) || error("Missing supplied LIBSVM input: $dataset_path")
    AA, y_raw = load_dataset(dataset_name, dense=false, replace=false, verbose=false)
    A = AA
    m, n = size(A)
    println("Loaded data: m = $m samples, n = $n features")
    if length(y_raw) != m
        error("Dataset has $m rows but $(length(y_raw)) labels")
    end
    unique_labels = unique(y_raw)
    if length(unique_labels) == 2
        y = map(l -> l == unique_labels[1] ? -1.0 : 1.0, y_raw)
        y = Float64.(y)
    else
        error("Expected exactly two unique labels, found: $unique_labels")
    end
    @assert all(ℓ -> ℓ == -1.0 || ℓ == 1.0, y) "Labels must be ±1"
    @assert length(y) == m
    L = opnorm(Matrix(A))^2 / (4 * m)
    x0 = rand(rng, n)
    return A, y, x0, L
end

# Experiment Configuration
groups = [
    (dataset="a5a", s=5),
    (dataset="a5a", s=10),
    (dataset="a5a", s=20),
    (dataset="a6a", s=5),
    (dataset="a6a", s=10),
    (dataset="a6a", s=20),
    (dataset="a7a", s=5),
    (dataset="a7a", s=10),
    (dataset="a7a", s=20),
    (dataset="a8a", s=5),
    (dataset="a8a", s=10),
    (dataset="a8a", s=20),
]

seeds = [23, 24, 25, 26, 27]
N = 3000
epsilon = 1e-5
if QUICK_RUN
    groups = groups[1:1]
    seeds = seeds[1:1]
    N = 50
end

checkpoint_path = output_path("sparse_logistic_checkpoint.csv")
all_results = NamedTuple[]
if RESUME_RUN && isfile(checkpoint_path)
    checkpoint = CSV.read(checkpoint_path, DataFrame)
    for row in eachrow(checkpoint)
        push!(all_results, NamedTuple(row))
    end
    println("Resuming from $(nrow(checkpoint)) completed method rows.")
end
completed_cases = Set((row.dataset, row.s, row.seed) for row in all_results)

println("\n" * "^"^70)
println("STARTING Sparse Logistic Regression BENCHMARK ($(length(groups)) group(s) × $(length(seeds)) seed(s))")
println("^"^70)

for (gid, (dataset, s)) in enumerate(groups)
    println("\n" * "#" * "="^68 * "#")
    println("### GROUP $gid: (dataset=$dataset, s=$s) ###")
    println("#" * "="^68 * "#")

    for seed in seeds
        if (dataset, s, seed) in completed_cases
            println("Skipping completed case: dataset=$dataset, s=$s, seed=$seed")
            continue
        end
        println("\nSeed = $seed")
        A, y, x0, L = load_logistic_instance(dataset; seed=seed)
        m, n = size(A)

        # Define problem-specific f and grad_f
        f(x) = logistic_value(-y .* Vector(A * x))
        function grad_f(x)
            scores = -y .* Vector(A * x)
            weights = -y .* logistic_sigmoid.(scores)
            return Vector(transpose(A) * weights) ./ m
        end

        # IHT
        println("-------------------- Method = IHT ----------------------------")
        try
            _, fx, elapsed_time, _ = IHT(f, grad_f, s, L, x0, N; epsilon=epsilon)
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="IHT", UB=fx, Time=elapsed_time))
            println("IHT:                     UB = $(round(fx, digits=4)), Time = $(round(elapsed_time, digits=2))s")
        catch e
            @warn "IHT failed: $e"
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="IHT", UB=Inf, Time=0.0))
        end

        # Adaptive IHT (AGP)
        println("-------------------- Method = AGP ----------------------------")
        try
            _, fx, elapsed_time, _, _, _, _, _ = adaptive_IHT(
                f, grad_f, s, x0, N; gamma=1/4, delta=1e-10, beta=2.0, epsilon=epsilon)
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="AGP", UB=fx, Time=elapsed_time))
            println("AGP:                     UB = $(round(fx, digits=4)), Time = $(round(elapsed_time, digits=2))s")
        catch e
            @warn "AGP failed: $e"
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="AGP", UB=Inf, Time=0.0))
        end

        # Partial Sparse Simplex
        println("-------------------- Method = PSS ----------------------------")
        try
            _, fx, elapsed_time, _ =
                partial_sparse_simplex_logistic(A, y, f, grad_f, s, N, x0; epsilon=epsilon)
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="PSS", UB=fx, Time=elapsed_time))
            println("PSS:                     UB = $(round(fx, digits=4)), Time = $(round(elapsed_time, digits=2))s")
        catch e
            @warn "PSS failed: $e"
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="PSS", UB=Inf, Time=0.0))
        end
        CSV.write(checkpoint_path, DataFrame(all_results))
        push!(completed_cases, (dataset, s, seed))
    end
end

# Save Raw Results
CSV.write(output_path("sparse_logistic_results_all.csv"), DataFrame(all_results))
println("\nSaved all results to: sparse_logistic_results_all.csv")

# Compute SGap (%)
df = CSV.read(output_path("sparse_logistic_results_all.csv"), DataFrame)
rename!(df, :Time => Symbol("Time (s)"))
sort!(df, [:dataset, :s, :seed, :method])
gdf = groupby(df, [:dataset, :s, :seed])
transform!(gdf, :UB => (x -> minimum(x)) => :UB_best)
df[!, Symbol("SGap (%)")] = (df.UB .- df.UB_best) ./ (abs.(df.UB) .+ 1e-4) .* 100
select!(df, Not(:UB_best))
CSV.write(output_path("sparse_logistic_detailed_with_sgap.csv"), df)
println("Saved detailed results with SGap to: sparse_logistic_detailed_with_sgap.csv")

# Summary Statistics
summary_df = combine(
    groupby(df, [:dataset, :s, :method]),
    Symbol("Time (s)") => (x -> round(mean(x), digits=2)) => Symbol("Avg Time"),
    Symbol("Time (s)") => (x -> round(std(x), digits=2)) => Symbol("Std Time"),
    Symbol("SGap (%)") => (x -> round(mean(x), digits=2)) => Symbol("Avg SGap"),
    Symbol("SGap (%)") => (x -> round(std(x), digits=2)) => Symbol("Std SGap")
)
sort!(summary_df, [:dataset, :s, :method])

println("\n" * "="^90)
println("SUMMARY: Avg ± Std over seeds (per dataset, s, method)")
println("="^90)
println("dataset\ts\tmethod\t\t\tAvg Time\tStd Time\tAvg SGap\tStd SGap")
for row in eachrow(summary_df)
    @printf("%-8s\t%d\t%-20s\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n",
        row.dataset, row.s, row.method,
        row[Symbol("Avg Time")], row[Symbol("Std Time")],
        row[Symbol("Avg SGap")], row[Symbol("Std SGap")]
    )
end
CSV.write(output_path("sparse_logistic_summary_with_sgap.csv"), summary_df)
println("\nFinal summary saved to: sparse_logistic_summary_with_sgap.csv")
println("\nAll experiments completed.")
