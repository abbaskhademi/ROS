# Load required packages
required_packages = ["LinearAlgebra", "Statistics", "Random", "Printf", "LIBSVMdata", "Arpack", "SparseArrays", "Optim", "Zygote", "JuMP", "MosekTools"]
for pkg in required_packages
    try
        eval(Meta.parse("using $pkg"))
    catch e
        @warn "$pkg not found, installing..."
        import Pkg
        Pkg.add(pkg)
        eval(Meta.parse("using $pkg"))
    end
end

# Main Algorithm Functions
using LIBSVMdata
using LinearAlgebra
using SparseArrays
using Arpack
using Printf
using DataFrames
using CSV
using Statistics
using Zygote
using Optim
using Random
using JuMP
using MosekTools

# IHT Function (Unchanged)
function IHT(f, g, s, L, x0, N; epsilon=1e-5)
    start_time = time()
    n = length(x0)
    x = copy(x0)
    fold = Inf
    iter_stuck = 0
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
            @printf("🛑 Stopped IHT early at iter %d: ||x_%d - x_%d|| = %.2e <= %.2e\n",
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

# Adaptive IHT Function (Unchanged)
function adaptive_IHT(f, g, s, x0, N; gamma=1/4, delta=1e-10, beta=2.0, epsilon=1e-5)
    start_time = time()
    n = length(x0)
    x_curr = copy(x0)
    fx = f(x_curr)
    current_grad = g(x_curr)
    fold = Inf
    iter_stuck = 0
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
            Random.seed!(23)
            d0 = randn(n)
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
                gamma = gamma * 0.9
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
            @printf("🛑 Adaptive IHT stopped early: ||x_%d - x_%d|| = %.2e <= %.2e\n",
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

# Partial Sparse Simplex with MOSEK (Updated for Infinite Bounds)
function partial_sparse_simplex_mosek(f, grad_f, s, N, x0, L; epsilon=1e-5)
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
    fx = f(x)
    iters = N

    # Helper function for one-dimensional optimization using MOSEK
    function optimize_1d_mosek(obj, lower_bound, upper_bound, x, direction, grad_f)
        model = Model(Mosek.Optimizer)
        set_silent(model)
        # Handle infinite bounds
        if isinf(lower_bound) && isinf(upper_bound)
            @variable(model, t)
        else
            @variable(model, t, lower_bound=lower_bound, upper_bound=upper_bound)
        end
        f_t0 = obj(0.0)
        grad_t0 = dot(grad_f(x), direction)  # Directional derivative
        @objective(model, Min, f_t0 + grad_t0 * t + 0.5 * L * t^2)
        optimize!(model)
        if termination_status(model) == MOI.OPTIMAL
            t_opt = value(t)
            # Check for large t_opt to avoid numerical issues
            if abs(t_opt) > 1e6
                @warn "MOSEK returned large t_opt = $t_opt, falling back to grid search"
                t_values = range(-1000.0, 1000.0, length=100)
                f_values = [obj(t_val) for t_val in t_values]
                min_idx = argmin(f_values)
                return t_values[min_idx], f_values[min_idx]
            end
            # Evaluate true objective, handle potential numerical issues
            try
                f_val = obj(t_opt)
                return t_opt, f_val
            catch e
                @warn "Objective evaluation failed at t_opt = $t_opt: $e, falling back to grid search"
                t_values = range(-1000.0, 1000.0, length=100)
                f_values = [obj(t_val) for t_val in t_values]
                min_idx = argmin(f_values)
                return t_values[min_idx], f_values[min_idx]
            end
        else
            @warn "MOSEK failed, falling back to grid search"
            t_values = range(-1000.0, 1000.0, length=100)
            f_values = [obj(t_val) for t_val in t_values]
            min_idx = argmin(f_values)
            return t_values[min_idx], f_values[min_idx]
        end
    end

    for k in 1:N
        x_prev = copy(x)
        g = grad_f(x)
        supp = findall(!iszero, x)
        nonsupp = setdiff(1:n, supp)
        f_best = fx
        x_best = copy(x)

        # Case 1: Support is not full
        if length(supp) < s
            for i in 1:n
                e_i = zeros(n); e_i[i] = 1.0
                obj(t) = f(x + t * e_i)
                t_opt, f_new = optimize_1d_mosek(obj, -Inf, Inf, x, e_i, grad_f)
                if f_new < f_best - 1e-8
                    f_best = f_new
                    x_best = x + t_opt * e_i
                end
            end

        # Case 2: Support is full
        else
            # Step 1: Minimize along each support coordinate
            for i in supp
                e_i = zeros(n); e_i[i] = 1.0
                obj(t) = f(x + t * e_i)
                t_opt, f_new = optimize_1d_mosek(obj, -Inf, Inf, x, e_i, grad_f)
                if f_new < f_best - 1e-8
                    f_best = f_new
                    x_best = x + t_opt * e_i
                end
            end

            # Step 2: Swap the worst support feature with the best non-support feature
            if !isempty(nonsupp)
                m_idx = supp[argmin(abs.(x[supp]))]
                i2_local_idx = argmax(abs.(g[nonsupp]))
                i2 = nonsupp[i2_local_idx]
                e_m = zeros(n); e_m[m_idx] = -x[m_idx]
                e_i2 = zeros(n); e_i2[i2] = 1.0
                obj2(t) = f(x + e_m + t * e_i2)
                t_opt, f_new = optimize_1d_mosek(obj2, -Inf, Inf, x + e_m, e_i2, grad_f)
                if f_new < f_best - 1e-8
                    f_best = f_new
                    x_best = x + e_m + t_opt * e_i2
                end
            end
        end

        x = x_best
        fx = f_best

        # Logging
        if k % 500 == 0
            @printf("PSS-MOSEK: iter = %5d, f(x) = %5.4f\n", k, fx)
        end

        # Early stopping
        if norm(x - x_prev) <= epsilon
            @printf("🛑 Stopped PSS-MOSEK early at iter %d: ||x_%d - x_%d|| = %.2e <= %.2e\n",
                    k, k, k-1, norm(x - x_prev), epsilon)
            iters = k
            @printf("PSS-MOSEK: iter = %5d, f(x) = %5.4f\n", k, fx)
            break
        end
    end

    elapsed_time = time() - start_time
    println("  Elapsed time: $(round(elapsed_time, digits=4)) seconds")
    return x, fx, elapsed_time, iters
end

# Problem Generation
function load_logistic_instance(dataset_name; seed=nothing)
    if !isnothing(seed)
        Random.seed!(seed)
    end
    println("📥 Loading dataset: $dataset_name")
    AA, y_raw = load_dataset(dataset_name, dense=false, replace=true, verbose=false)
    A = AA
    m, n = size(A)
    println("📏 Loaded data: m = $m samples, n = $n features")
    if length(y_raw) != m
        error("FATAL: A has $m rows, but y has $(length(y_raw)) labels.")
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
    x0 = rand(n)
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

all_results = NamedTuple[]

println("\n" * "^"^70)
println("🚀 STARTING Sparse Logistic Regression BENCHMARK ($(length(groups)) GROUPS × $(length(seeds)) seeds)")
println("^"^70)

for (gid, (dataset, s)) in enumerate(groups)
    println("\n" * "#" * "="^68 * "#")
    println("### 🌐 GROUP $gid: (dataset=$dataset, s=$s) ###")
    println("#" * "="^68 * "#")

    for seed in seeds
        println("\n🌱 Seed = $seed")
        A, y, x0, L = load_logistic_instance(dataset; seed=seed)
        m, n = size(A)

        # Define problem-specific f and grad_f
        f(x) = mean(log.(1 .+ exp.(-y .* (A * x))))
        grad_f(x) = Zygote.gradient(f, x)[1]

        # IHT
        println("-------------------- Method = IHT ----------------------------")
        try
            x, fx, time, iters = IHT(f, grad_f, s, L, x0, N; epsilon=epsilon)
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="IHT", UB=fx, Time=time))
            println("✅ IHT:                     UB = $(round(fx, digits=4)), Time = $(round(time, digits=2))s")
        catch e
            @warn "❌ IHT failed: $e"
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="IHT", UB=Inf, Time=0.0))
        end

        # Adaptive IHT (AGP)
        println("-------------------- Method = AGP ----------------------------")
        try
            x, fx, time, iters, _, _, backtrack_counts, _ = adaptive_IHT(
                f, grad_f, s, x0, N; gamma=1/4, delta=1e-10, beta=2.0, epsilon=epsilon)
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="AGP", UB=fx, Time=time))
            println("✅ AGP:                     UB = $(round(fx, digits=4)), Time = $(round(time, digits=2))s")
        catch e
            @warn "❌ AGP failed: $e"
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="AGP", UB=Inf, Time=0.0))
        end

        # Partial Sparse Simplex with MOSEK
        println("-------------------- Method = PSS-MOSEK ----------------------------")
        try
            x, fx, time, iters = partial_sparse_simplex_mosek(f, grad_f, s, N, x0, L; epsilon=epsilon)
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="PSS-MOSEK", UB=fx, Time=time))
            println("✅ PSS-MOSEK:               UB = $(round(fx, digits=4)), Time = $(round(time, digits=2))s")
        catch e
            @warn "❌ PSS-MOSEK failed: $e"
            push!(all_results, (group_id=gid, dataset=dataset, m=m, n=n, s=s, seed=seed, method="PSS-MOSEK", UB=Inf, Time=0.0))
        end
    end
end

# Save Raw Results
CSV.write("sparse_logistic_results_all.csv", DataFrame(all_results))
println("\n💾 Saved all results to: sparse_logistic_results_all.csv")

# Compute SGap (%)
df = CSV.read("sparse_logistic_results_all.csv", DataFrame)
rename!(df, :Time => Symbol("Time (s)"))
sort!(df, [:dataset, :s, :seed, :method])
gdf = groupby(df, [:dataset, :s, :seed])
transform!(gdf, :UB => (x -> minimum(x)) => :UB_best)
df[!, Symbol("SGap (%)")] = (df.UB .- df.UB_best) ./ (abs.(df.UB_best) .+ 1e-4) .* 100
select!(df, Not(:UB_best))
CSV.write("sparse_logistic_detailed_with_sgap.csv", df)
println("💾 Saved detailed results with SGap to: sparse_logistic_detailed_with_sgap.csv")

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
println("📊 SUMMARY: Avg ± Std over seeds (per dataset, s, method)")
println("="^90)
println("dataset\ts\tmethod\t\t\tAvg Time\tStd Time\tAvg SGap\tStd SGap")
for row in eachrow(summary_df)
    @printf("%-8s\t%d\t%-20s\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n",
        row.dataset, row.s, row.method,
        row[Symbol("Avg Time")], row[Symbol("Std Time")],
        row[Symbol("Avg SGap")], row[Symbol("Std SGap")]
    )
end
CSV.write("sparse_logistic_summary_with_sgap.csv", summary_df)
println("\n✅ Final summary saved to: sparse_logistic_summary_with_sgap.csv")
println("\n🎉 All experiments completed!")