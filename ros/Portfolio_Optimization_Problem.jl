#                    Main code

using JuMP, MosekTools, CSV, DataFrames, Distributions, LinearAlgebra, Printf, Random, PrettyTables

# --- 1. Generate Synthetic Data (Matches Goldfarb & Iyengar (2003) Section 7) ---
function generate_portfolio_data(n_assets, n_factors; seed=2024)
    Random.seed!(seed)
    n, k = n_assets, n_factors

    # Generate expected returns and factor model parameters
    mu = rand(Uniform(0.05, 0.15), n)
    B = randn(n, k) * 0.1
    F = Diagonal(rand(Uniform(0.02, 0.05), k))
    D = Diagonal(rand(Uniform(0.01, 0.03), n))
    Σ = B * F * B' + D
    Σ = (Σ + Σ') / 2  # Ensure symmetry

    # Compute matrix square root of Σ (via eigendecomposition under the hood)
    Σ_half = nothing
    try
        Σ_half = sqrt(Hermitian(Σ))
    catch e
        @warn "Matrix square root failed — adding small jitter for numerical stability"
        jitter = 1e-8 * I(n)
        Σ_stable = Hermitian(Σ + jitter)
        Σ_half = sqrt(Σ_stable)
    end

    # Final safety check
    if isnothing(Σ_half)
        error("Σ_half was not computed — matrix square root failed even after jitter.")
    end

    # Export to CSV (Includes seed in filename)
    CSV.write("mu_seed$(seed).csv", DataFrame(mu=mu))
    CSV.write("Sigma_half_seed$(seed).csv", DataFrame(Matrix(Σ_half), :auto))

    return mu, Σ_half
end


# --- 2. Sparse Projection onto {‖ζ‖₀ ≤ s, ‖ζ‖_∞ ≤ 1} ---
function sparse_project(z_tilde, s)
    n = length(z_tilde)
    # Get indices of the 's' largest absolute values
    idx = sortperm(abs.(z_tilde), rev=true)[1:s]
    ζ = zeros(n)
    # Apply L-infinity constraint and sparsity
    for i in idx
        ζ[i] = clamp(z_tilde[i], -1, 1)
    end
    return ζ
end

# --- 3. Worst-Case Subproblem (All Variants) ---
function worst_case_subproblem(x, Σ_half, s; method="heuristic", optimizer=Mosek.Optimizer)
    n = length(x)
    if size(Σ_half, 2) != n
        error("Dimension mismatch: Σ_half has $(size(Σ_half, 2)) columns, but x has length $n")
    end
    c = Σ_half' * x

    model = Model(optimizer)
    set_silent(model)
    @variable(model, -1 <= ζ[1:n] <= 1)

    if method == "heuristic"
        # Solves the relaxed problem and projects the solution
        @objective(model, Min, ζ' * c)
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            @warn "Heuristic subproblem not optimal."
            return zeros(n)
        end
        ζ_tilde = value.(ζ)
        return sparse_project(ζ_tilde, s)

    elseif method == "mip"
        # Solves the exact non-convex problem using binary variables
        @variable(model, z[1:n], Bin)
        @constraint(model, [i=1:n], ζ[i] <= z[i])
        @constraint(model, [i=1:n], ζ[i] >= -z[i])
        @constraint(model, sum(z) <= s)
        @objective(model, Min, ζ' * c)
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            @warn "MIP subproblem not optimal."
            return zeros(n)
        end
        return value.(ζ)

    elseif method == "relaxed"
        # Solves the continuous relaxation (convex) and projects
        @variable(model, 0 <= z[1:n] <= 1)
        @constraint(model, [i=1:n], ζ[i] <= z[i])
        @constraint(model, [i=1:n], ζ[i] >= -z[i])
        @constraint(model, sum(z) <= s)
        @objective(model, Min, ζ' * c)
        optimize!(model)
        if termination_status(model) != MOI.OPTIMAL
            @warn "Relaxed subproblem not optimal."
            return zeros(n)
        end
        ζ_tilde = value.(ζ)
        return sparse_project(ζ_tilde, s)

    else
        error("Unknown method: $method. Use 'heuristic', 'mip', or 'relaxed'.")
    end
end

# --- 4. Solve Master Problem (Finds optimal portfolio x given current scenarios S) ---
function solve_master_problem(mu, Σ_half, scenarios; optimizer=Mosek.Optimizer)
    n = length(mu)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, x[1:n] >= 0)
    @variable(model, τ) # Lower Bound on worst-case return
    @objective(model, Max, τ)
    @constraint(model, sum(x) == 1)

    # Add constraints for all current scenarios
    for ζ in scenarios
        r = Σ_half * ζ + mu
        @constraint(model, τ <= r' * x)
    end

    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL
        @warn "Master problem not optimal."
        return zeros(n), -Inf
    end
    return value.(x), objective_value(model)
end

# --- 5. Solve Nominal Problem (Initial solution: x maximizing mu'x) ---
function solve_nominal_problem(mu; optimizer=Mosek.Optimizer)
    n = length(mu)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, x[1:n] >= 0)
    @objective(model, Max, mu' * x)
    @constraint(model, sum(x) == 1)
    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL
        @warn "Nominal problem not optimal. Returning equal weights."
        return ones(n) / n
    end
    return value.(x)
end

# --- 6. Cutting Plane Algorithm ---
function cutting_plane_algorithm(mu, Σ_half, s;
    method="heuristic",
    max_iter=5000,
    tol=1e-4,
    limit_time=1000.0,
    verbose=true,
    optimizer=Mosek.Optimizer,
    x_init=nothing,
    seed::Int # 👈 New required argument
)
    n = length(mu)
    S = Vector{Vector{Float64}}()
    tau_history = Vector{Tuple{Int, Float64}}()

    # Initialize from provided x_init (Nominal solution computed once per seed)
    if isnothing(x_init)
        # Should not happen if main loop is structured correctly
        error("x_init must be provided (nominal solution).")
    end

    x_current = x_init
    total_algorithm_time = 0.0
    if verbose
        println("Intialized from provided nominal solution (Seed $seed | $method).")
    end

    τ_prev = -Inf

    for ℓ in 1:max_iter
        # Check time limit
        if total_algorithm_time > limit_time
            if verbose
                println("⏰ Time limit ($limit_time seconds) exceeded at iteration $ℓ ($method | Seed $seed)")
            end
            break
        end

        # Step 1 & 2: Solve worst-case subproblem
        iter_time = @elapsed ζ = worst_case_subproblem(x_current, Σ_half, s; method=method, optimizer=optimizer)

        # Step 3: Add to scenarios (S)
        push!(S, ζ)

        # Step 4: Solve master problem
        t_master = @elapsed x_opt, τ_current = solve_master_problem(mu, Σ_half, S; optimizer=optimizer)
        iter_time += t_master
        total_algorithm_time += iter_time

        # Store τ for this iteration
        push!(tau_history, (ℓ, τ_current))

        x_current = x_opt

        # Compute Δτ for output
        Δτ = ℓ > 1 ? abs(τ_current - τ_prev) : "N/A"

        # Check convergence
        if ℓ > 1 && abs(τ_current - τ_prev) <= tol
            if verbose
                println("✅ Converged at iteration $ℓ: Δτ = $(abs(τ_current - τ_prev)) ($method | Seed $seed)")
            end
            break
        end

        τ_prev = τ_current

        if verbose && ℓ % 10 == 0
            println("Iteration $ℓ ($method | Seed $seed): τ = $(round(τ_current, digits=6)) | Δτ = $(Δτ isa String ? Δτ : round(Δτ, digits=6)) | #Scenarios = $(length(S)) | Iter time = $(round(iter_time, digits=4))s")
        end
    end

    # Save τ history to CSV (Includes seed in filename)
    df_tau = DataFrame(iteration=[t[1] for t in tau_history], tau=[t[2] for t in tau_history])
    CSV.write("tau_history_$(method)_s$(s)_seed$(seed).csv", df_tau)
    if verbose
        println("💾 Saved τ history: tau_history_$(method)_s$(s)_seed$(seed).csv")
    end

    if verbose
        println("\n⏱️ TOTAL CUTTING PLANE TIME ($method | Seed $seed): $(round(total_algorithm_time, digits=4)) seconds")
    end

    return x_current, τ_prev, S, total_algorithm_time
end

println("✅ All Julia functions defined (heuristic, mip, relaxed).")




# 1000
# --- RUN EXPERIMENT FOR 5 SEEDS ----------------------------------------------
# -----------------------------------------------------------------------------
n_assets = 1000      # n
n_factors = 25
SEEDS = [23, 24, 25, 26, 27] # 5 different seeds
methods = ["heuristic", "mip", "relaxed"]
sparsities = [5, 10, 20]  #s
all_results = [] # Collect all results here

println("\n" * "^"^60)
println("🚀 STARTING CUTTING PLANE ALGORITHM (ALL METHODS, 5 SEEDS)")
println("📏 Problem Size: n_assets = $n_assets, n_factors = $n_factors")
println("^"^60)

for seed in SEEDS
    println("\n" * "#" * "="^58 * "#")
    println("### 🌱 Starting Experiment for SEED = $seed ###")
    println("#" * "="^58 * "#")

    # --- 1. Data Generation (Seed specific) ---
    mu, Σ_half = generate_portfolio_data(n_assets, n_factors; seed=seed)
    println("✅ Generated and saved synthetic data (seed=$seed).")

    # --- 2. Compute Nominal Solution (Seed specific) ---
    println("🧠 Computing nominal solution (seed=$seed)...")
    nominal_time_shared = @elapsed x_nominal = solve_nominal_problem(mu; optimizer=Mosek.Optimizer)
    nominal_return = mu' * x_nominal
    println("✅ Nominal solution computed in $(round(nominal_time_shared, digits=4))s")
    println("📈 Nominal expected return (μ'x): $(round(nominal_return, digits=6))")

    # --- 3. Run Experiments ---
    for method in methods
        for s in sparsities
            println("\n" * "-"^50)
            println("▶️ Solving for sparsity s = $s | Method = $method | Seed = $seed")
            println("-"^50)

            total_elapsed = @elapsed begin
                x_opt, τ_opt, scenarios, algo_time = cutting_plane_algorithm(
                    mu, Σ_half, s;
                    method=method,
                    max_iter=5000,
                    tol=1e-4,
                    limit_time=1000.0,
                    x_init=x_nominal,
                    seed=seed
                )
            end

            push!(all_results, (
                seed = seed,
                method = method,
                sparsity = s,
                tau = τ_opt,
                scenarios = length(scenarios),
                algo_time = algo_time,
                nominal_return = nominal_return
            ))

            println("🏁 $method (Seed $seed): FINAL: τ = $(round(τ_opt, digits=6)) | #Scenarios = $(length(scenarios)) | CP Time = $(round(algo_time, digits=4))s")

            df = DataFrame(index=1:length(x_opt), weight=x_opt)
            CSV.write("x_opt_$(method)_s$(s)_seed$(seed).csv", df)
            println("💾 Saved: x_opt_$(method)_s$(s)_seed$(seed).csv")
        end
    end
end

# --- Create Final Aggregate Table with SGap Calculation ---
println("\n📊 FINAL AGGREGATE REPORT (Across all 5 Seeds) — Problem Size: n = $n_assets assets")

df_final = DataFrame(all_results)
rename!(df_final, :sparsity => :s, :tau => :UB, :scenarios => Symbol("#Scenarios"), :algo_time => Symbol("Time (s)"))

# Sort for better display
sort!(df_final, [:seed, :s, :method])

# --- COMPUTE AND ADD SGAP COLUMN ---
# Group the DataFrame by seed and sparsity (s)
gdf = groupby(df_final, [:seed, :s])

# Use `transform` to compute the best UB and then the SGap for each group
transform!(gdf, :UB => (UB -> minimum(UB)) => :UB_best, renamecols=false)
transform!(gdf, [:UB, :UB_best] => ((UB, UB_best) -> (UB .- UB_best) ./ (abs.(UB) .+ 1e-4) .* 100) => Symbol("SGap (%)"), renamecols=false)

# Drop the temporary column
select!(df_final, Not(:UB_best))

#println("\n--- Detailed Results (All Seeds) ---")
#pretty_table(df_final, title="Detailed Performance by Seed", limit_printing=false)

# Aggregate (Summary Statistics)
df_summary = combine(
    groupby(df_final, [:s, :method]),
    Symbol("Time (s)") => (x -> round(mean(x), digits=2)) => Symbol("Avg Time"),
    Symbol("Time (s)") => (x -> round(std(x), digits=2)) => Symbol("Std Time"),
    Symbol("SGap (%)") => (x -> round(mean(x), digits=2)) => Symbol("Avg SGap"),
    Symbol("SGap (%)") => (x -> round(std(x), digits=2)) => Symbol("Std SGap")
)
sort!(df_summary, [:s, :method])

println("\n--- Summary Statistics (Averaged Across 5 Seeds) ---")
pretty_table(df_summary, title="Average Performance (5 Seeds)", limit_printing=false)

CSV.write("final_report_aggregate_n$(n_assets).csv", df_final)
CSV.write("final_report_summary_n$(n_assets).csv", df_summary)
println("💾 Saved detailed report: final_report_aggregate_n$(n_assets).csv")
println("💾 Saved summary report: final_report_summary_n$(n_assets).csv")

println("\n🎉 Experiment completed for all 5 seeds.")





# 3000
# --- RUN EXPERIMENT FOR 5 SEEDS ----------------------------------------------
# -----------------------------------------------------------------------------
n_assets = 3000      # n
n_factors = 25
SEEDS = [23, 24, 25, 26, 27] # 5 different seeds
methods = ["heuristic", "mip", "relaxed"]
sparsities = [5, 10, 20]  #s
all_results = [] # Collect all results here

println("\n" * "^"^60)
println("🚀 STARTING CUTTING PLANE ALGORITHM (ALL METHODS, 5 SEEDS)")
println("📏 Problem Size: n_assets = $n_assets, n_factors = $n_factors")
println("^"^60)

for seed in SEEDS
    println("\n" * "#" * "="^58 * "#")
    println("### 🌱 Starting Experiment for SEED = $seed ###")
    println("#" * "="^58 * "#")

    # --- 1. Data Generation (Seed specific) ---
    mu, Σ_half = generate_portfolio_data(n_assets, n_factors; seed=seed)
    println("✅ Generated and saved synthetic data (seed=$seed).")

    # --- 2. Compute Nominal Solution (Seed specific) ---
    println("🧠 Computing nominal solution (seed=$seed)...")
    nominal_time_shared = @elapsed x_nominal = solve_nominal_problem(mu; optimizer=Mosek.Optimizer)
    nominal_return = mu' * x_nominal
    println("✅ Nominal solution computed in $(round(nominal_time_shared, digits=4))s")
    println("📈 Nominal expected return (μ'x): $(round(nominal_return, digits=6))")

    # --- 3. Run Experiments ---
    for method in methods
        for s in sparsities
            println("\n" * "-"^50)
            println("▶️ Solving for sparsity s = $s | Method = $method | Seed = $seed")
            println("-"^50)

            total_elapsed = @elapsed begin
                x_opt, τ_opt, scenarios, algo_time = cutting_plane_algorithm(
                    mu, Σ_half, s;
                    method=method,
                    max_iter=5000,
                    tol=1e-4,
                    limit_time=1000.0,
                    x_init=x_nominal,
                    seed=seed
                )
            end

            push!(all_results, (
                seed = seed,
                method = method,
                sparsity = s,
                tau = τ_opt,
                scenarios = length(scenarios),
                algo_time = algo_time,
                nominal_return = nominal_return
            ))

            println("🏁 $method (Seed $seed): FINAL: τ = $(round(τ_opt, digits=6)) | #Scenarios = $(length(scenarios)) | CP Time = $(round(algo_time, digits=4))s")

            df = DataFrame(index=1:length(x_opt), weight=x_opt)
            CSV.write("x_opt_$(method)_s$(s)_seed$(seed).csv", df)
            println("💾 Saved: x_opt_$(method)_s$(s)_seed$(seed).csv")
        end
    end
end

# --- Create Final Aggregate Table with SGap Calculation ---
println("\n📊 FINAL AGGREGATE REPORT (Across all 5 Seeds) — Problem Size: n = $n_assets assets")

df_final = DataFrame(all_results)
rename!(df_final, :sparsity => :s, :tau => :UB, :scenarios => Symbol("#Scenarios"), :algo_time => Symbol("Time (s)"))

# Sort for better display
sort!(df_final, [:seed, :s, :method])

# --- COMPUTE AND ADD SGAP COLUMN ---
# Group the DataFrame by seed and sparsity (s)
gdf = groupby(df_final, [:seed, :s])

# Use `transform` to compute the best UB and then the SGap for each group
transform!(gdf, :UB => (UB -> minimum(UB)) => :UB_best, renamecols=false)
transform!(gdf, [:UB, :UB_best] => ((UB, UB_best) -> (UB .- UB_best) ./ (abs.(UB) .+ 1e-4) .* 100) => Symbol("SGap (%)"), renamecols=false)

# Drop the temporary column
select!(df_final, Not(:UB_best))

#println("\n--- Detailed Results (All Seeds) ---")
#pretty_table(df_final, title="Detailed Performance by Seed", limit_printing=false)

# Aggregate (Summary Statistics)
df_summary = combine(
    groupby(df_final, [:s, :method]),
    Symbol("Time (s)") => (x -> round(mean(x), digits=2)) => Symbol("Avg Time (s)"),
    Symbol("Time (s)") => (x -> round(std(x), digits=2)) => Symbol("Std Time (s)"),
    Symbol("SGap (%)") => (x -> round(mean(x), digits=2)) => Symbol("Avg SGap (%)"),
    Symbol("SGap (%)") => (x -> round(std(x), digits=2)) => Symbol("Std SGap (%)")
)
sort!(df_summary, [:s, :method])

println("\n--- Summary Statistics (Averaged Across 5 Seeds) ---")
pretty_table(df_summary, title="Average Performance (5 Seeds)", limit_printing=false)

CSV.write("final_report_aggregate_n$(n_assets).csv", df_final)
CSV.write("final_report_summary_n$(n_assets).csv", df_summary)
println("💾 Saved detailed report: final_report_aggregate_n$(n_assets).csv")
println("💾 Saved summary report: final_report_summary_n$(n_assets).csv")

println("\n🎉 Experiment completed for all 5 seeds.")





# 5000
# --- RUN EXPERIMENT FOR 5 SEEDS ----------------------------------------------
# -----------------------------------------------------------------------------
n_assets = 5000      # n
n_factors = 25
SEEDS = [23, 24, 25, 26, 27] # 5 different seeds
methods = ["heuristic", "mip", "relaxed"]
sparsities = [5, 10, 20]  #s
all_results = [] # Collect all results here

println("\n" * "^"^60)
println("🚀 STARTING CUTTING PLANE ALGORITHM (ALL METHODS, 5 SEEDS)")
println("📏 Problem Size: n_assets = $n_assets, n_factors = $n_factors")
println("^"^60)

for seed in SEEDS
    println("\n" * "#" * "="^58 * "#")
    println("### 🌱 Starting Experiment for SEED = $seed ###")
    println("#" * "="^58 * "#")

    # --- 1. Data Generation (Seed specific) ---
    mu, Σ_half = generate_portfolio_data(n_assets, n_factors; seed=seed)
    println("✅ Generated and saved synthetic data (seed=$seed).")

    # --- 2. Compute Nominal Solution (Seed specific) ---
    println("🧠 Computing nominal solution (seed=$seed)...")
    nominal_time_shared = @elapsed x_nominal = solve_nominal_problem(mu; optimizer=Mosek.Optimizer)
    nominal_return = mu' * x_nominal
    println("✅ Nominal solution computed in $(round(nominal_time_shared, digits=4))s")
    println("📈 Nominal expected return (μ'x): $(round(nominal_return, digits=6))")

    # --- 3. Run Experiments ---
    for method in methods
        for s in sparsities
            println("\n" * "-"^50)
            println("▶️ Solving for sparsity s = $s | Method = $method | Seed = $seed")
            println("-"^50)

            total_elapsed = @elapsed begin
                x_opt, τ_opt, scenarios, algo_time = cutting_plane_algorithm(
                    mu, Σ_half, s;
                    method=method,
                    max_iter=5000,
                    tol=1e-4,
                    limit_time=1000.0,
                    x_init=x_nominal,
                    seed=seed
                )
            end

            push!(all_results, (
                seed = seed,
                method = method,
                sparsity = s,
                tau = τ_opt,
                scenarios = length(scenarios),
                algo_time = algo_time,
                nominal_return = nominal_return
            ))

            println("🏁 $method (Seed $seed): FINAL: τ = $(round(τ_opt, digits=6)) | #Scenarios = $(length(scenarios)) | CP Time = $(round(algo_time, digits=4))s")

            df = DataFrame(index=1:length(x_opt), weight=x_opt)
            CSV.write("x_opt_$(method)_s$(s)_seed$(seed).csv", df)
            println("💾 Saved: x_opt_$(method)_s$(s)_seed$(seed).csv")
        end
    end
end

# --- Create Final Aggregate Table with SGap Calculation ---
println("\n📊 FINAL AGGREGATE REPORT (Across all 5 Seeds) — Problem Size: n = $n_assets assets")

df_final = DataFrame(all_results)
rename!(df_final, :sparsity => :s, :tau => :UB, :scenarios => Symbol("#Scenarios"), :algo_time => Symbol("Time (s)"))

# Sort for better display
sort!(df_final, [:seed, :s, :method])

# --- COMPUTE AND ADD SGAP COLUMN ---
# Group the DataFrame by seed and sparsity (s)
gdf = groupby(df_final, [:seed, :s])

# Use `transform` to compute the best UB and then the SGap for each group
transform!(gdf, :UB => (UB -> minimum(UB)) => :UB_best, renamecols=false)
transform!(gdf, [:UB, :UB_best] => ((UB, UB_best) -> (UB .- UB_best) ./ (abs.(UB) .+ 1e-4) .* 100) => Symbol("SGap (%)"), renamecols=false)

# Drop the temporary column
select!(df_final, Not(:UB_best))

#println("\n--- Detailed Results (All Seeds) ---")
#pretty_table(df_final, title="Detailed Performance by Seed", limit_printing=false)

# Aggregate (Summary Statistics)
df_summary = combine(
    groupby(df_final, [:s, :method]),
    Symbol("Time (s)") => (x -> round(mean(x), digits=2)) => Symbol("Avg Time (s)"),
    Symbol("Time (s)") => (x -> round(std(x), digits=2)) => Symbol("Std Time (s)"),
    Symbol("SGap (%)") => (x -> round(mean(x), digits=2)) => Symbol("Avg SGap (%)"),
    Symbol("SGap (%)") => (x -> round(std(x), digits=2)) => Symbol("Std SGap (%)")
)
sort!(df_summary, [:s, :method])

println("\n--- Summary Statistics (Averaged Across 5 Seeds) ---")
pretty_table(df_summary, title="Average Performance (5 Seeds)", limit_printing=false)

CSV.write("final_report_aggregate_n$(n_assets).csv", df_final)
CSV.write("final_report_summary_n$(n_assets).csv", df_summary)
println("💾 Saved detailed report: final_report_aggregate_n$(n_assets).csv")
println("💾 Saved summary report: final_report_summary_n$(n_assets).csv")

println("\n🎉 Experiment completed for all 5 seeds.")



# 7000
# --- RUN EXPERIMENT FOR 5 SEEDS ----------------------------------------------
# -----------------------------------------------------------------------------
n_assets = 7000      # n
n_factors = 25
SEEDS = [23, 24, 25, 26, 27] # 5 different seeds
methods = ["heuristic", "mip", "relaxed"]
sparsities = [5, 10, 20]  #s
all_results = [] # Collect all results here

println("\n" * "^"^60)
println("🚀 STARTING CUTTING PLANE ALGORITHM (ALL METHODS, 5 SEEDS)")
println("📏 Problem Size: n_assets = $n_assets, n_factors = $n_factors")
println("^"^60)

for seed in SEEDS
    println("\n" * "#" * "="^58 * "#")
    println("### 🌱 Starting Experiment for SEED = $seed ###")
    println("#" * "="^58 * "#")

    # --- 1. Data Generation (Seed specific) ---
    mu, Σ_half = generate_portfolio_data(n_assets, n_factors; seed=seed)
    println("✅ Generated and saved synthetic data (seed=$seed).")

    # --- 2. Compute Nominal Solution (Seed specific) ---
    println("🧠 Computing nominal solution (seed=$seed)...")
    nominal_time_shared = @elapsed x_nominal = solve_nominal_problem(mu; optimizer=Mosek.Optimizer)
    nominal_return = mu' * x_nominal
    println("✅ Nominal solution computed in $(round(nominal_time_shared, digits=4))s")
    println("📈 Nominal expected return (μ'x): $(round(nominal_return, digits=6))")

    # --- 3. Run Experiments ---
    for method in methods
        for s in sparsities
            println("\n" * "-"^50)
            println("▶️ Solving for sparsity s = $s | Method = $method | Seed = $seed")
            println("-"^50)

            total_elapsed = @elapsed begin
                x_opt, τ_opt, scenarios, algo_time = cutting_plane_algorithm(
                    mu, Σ_half, s;
                    method=method,
                    max_iter=5000,
                    tol=1e-4,
                    limit_time=1000.0,
                    x_init=x_nominal,
                    seed=seed
                )
            end

            push!(all_results, (
                seed = seed,
                method = method,
                sparsity = s,
                tau = τ_opt,
                scenarios = length(scenarios),
                algo_time = algo_time,
                nominal_return = nominal_return
            ))

            println("🏁 $method (Seed $seed): FINAL: τ = $(round(τ_opt, digits=6)) | #Scenarios = $(length(scenarios)) | CP Time = $(round(algo_time, digits=4))s")

            df = DataFrame(index=1:length(x_opt), weight=x_opt)
            CSV.write("x_opt_$(method)_s$(s)_seed$(seed).csv", df)
            println("💾 Saved: x_opt_$(method)_s$(s)_seed$(seed).csv")
        end
    end
end

# --- Create Final Aggregate Table with SGap Calculation ---
println("\n📊 FINAL AGGREGATE REPORT (Across all 5 Seeds) — Problem Size: n = $n_assets assets")

df_final = DataFrame(all_results)
rename!(df_final, :sparsity => :s, :tau => :UB, :scenarios => Symbol("#Scenarios"), :algo_time => Symbol("Time (s)"))

# Sort for better display
sort!(df_final, [:seed, :s, :method])

# --- COMPUTE AND ADD SGAP COLUMN ---
# Group the DataFrame by seed and sparsity (s)
gdf = groupby(df_final, [:seed, :s])

# Use `transform` to compute the best UB and then the SGap for each group
transform!(gdf, :UB => (UB -> minimum(UB)) => :UB_best, renamecols=false)
transform!(gdf, [:UB, :UB_best] => ((UB, UB_best) -> (UB .- UB_best) ./ (abs.(UB) .+ 1e-4) .* 100) => Symbol("SGap (%)"), renamecols=false)

# Drop the temporary column
select!(df_final, Not(:UB_best))

#println("\n--- Detailed Results (All Seeds) ---")
#pretty_table(df_final, title="Detailed Performance by Seed", limit_printing=false)

# Aggregate (Summary Statistics)
df_summary = combine(
    groupby(df_final, [:s, :method]),
    Symbol("Time") => (x -> round(mean(x), digits=2)) => Symbol("Avg Time"),
    Symbol("Time") => (x -> round(std(x), digits=2)) => Symbol("Std Time"),
    Symbol("SGap") => (x -> round(mean(x), digits=2)) => Symbol("Avg SGap"),
    Symbol("SGap") => (x -> round(std(x), digits=2)) => Symbol("Std SGap")
)
sort!(df_summary, [:s, :method])

println("\n--- Summary Statistics (Averaged Across 5 Seeds) ---")
pretty_table(df_summary, title="Average Performance (5 Seeds)", limit_printing=false)

CSV.write("final_report_aggregate_n$(n_assets).csv", df_final)
CSV.write("final_report_summary_n$(n_assets).csv", df_summary)
println("💾 Saved detailed report: final_report_aggregate_n$(n_assets).csv")
println("💾 Saved summary report: final_report_summary_n$(n_assets).csv")

println("\n🎉 Experiment completed for all 5 seeds.")