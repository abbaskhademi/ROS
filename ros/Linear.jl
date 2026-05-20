# =============================================================================
# Linear Optimization with Implementation Error           Part 1
# Complete Implementation with Cutting-Plane Algorithm
# =============================================================================

using JuMP, MosekTools, CSV, DataFrames, Distributions, LinearAlgebra, Printf, Random, PrettyTables

# =============================================================================
# 1. Generate Synthetic Data
# =============================================================================
function generate_lp_data(n_x, m; seed=23)
    Random.seed!(seed)
    A = rand(Uniform(0.0, 1.0), m, n_x)
    b = rand(Uniform(10.0, 20.0), m)
    c = rand(Uniform(-1.0, 1.0), n_x)
    return A, b, c
end

# =============================================================================
# 2. Sparse Projection onto {‖u‖₀ ≤ s}
# =============================================================================
function sparse_project(u_tilde, s)
    n = length(u_tilde)
    if s >= n
        return u_tilde
    end
    idx = sortperm(abs.(u_tilde), rev=true)[1:s]
    u_proj = zeros(n)
    for i in idx
        u_proj[i] = u_tilde[i]
    end
    return u_proj
end

# =============================================================================
# 3. Worst-Case Subproblem
# =============================================================================
function worst_case_subproblem(gamma, s, x_current; method="heuristic", optimizer=Mosek.Optimizer)
    n_x = length(x_current)
    model = Model(optimizer)
    set_silent(model)

    if method == "heuristic"
        norm_x = norm(x_current, 2)
        if norm_x > 1e-10
            u_tilde = gamma * x_current / norm_x
        else
            u_tilde = zeros(n_x)
        end
        return sparse_project(u_tilde, s)

    elseif method == "mip"
        @variable(model, u[1:n_x])
        @variable(model, z[1:n_x], Bin)
        @constraint(model, sum(z) <= s)
        for i in 1:n_x
            @constraint(model, u[i] <= gamma * z[i])
            @constraint(model, u[i] >= -gamma * z[i])
        end
        @constraint(model, sum(u[i]^2 for i in 1:n_x) <= gamma^2)
        @objective(model, Max, sum(u[i] * x_current[i] for i in 1:n_x))
        optimize!(model)
        if termination_status(model) != OPTIMAL
            @warn "MIP subproblem not optimal."
            return zeros(n_x)
        end
        return value.(u)

    elseif method == "relaxed"
        @variable(model, u[1:n_x])
        @variable(model, 0 <= z[1:n_x] <= 1)
        @constraint(model, sum(z) <= s)
        for i in 1:n_x
            @constraint(model, u[i] <= gamma * z[i])
            @constraint(model, u[i] >= -gamma * z[i])
        end
        @constraint(model, sum(u[i]^2 for i in 1:n_x) <= gamma^2)
        @objective(model, Max, sum(u[i] * x_current[i] for i in 1:n_x))
        optimize!(model)
        if termination_status(model) != OPTIMAL
            @warn "Relaxed subproblem not optimal."
            return zeros(n_x)
        end
        u_tilde = value.(u)
        return sparse_project(u_tilde, s)

    else
        error("Unknown method: $method. Use 'heuristic', 'mip', or 'relaxed'.")
    end
end

# =============================================================================
# 4. Solve Master Problem
# =============================================================================
function solve_master_problem(c, A, b, scenarios; optimizer=Mosek.Optimizer)
    m, n_x = size(A)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, x[1:n_x] >= 0)
    @objective(model, Min, c' * x)

    for j in 1:m
        for u in scenarios
            @constraint(model, A[j, :]' * x + u' * x <= b[j])
        end
    end

    optimize!(model)
    if termination_status(model) != OPTIMAL
        @warn "Master problem not optimal."
        return zeros(n_x), Inf
    end
    return value.(x), objective_value(model)
end

# =============================================================================
# 5. Solve Nominal Problem (No uncertainty)
# =============================================================================
function solve_nominal_problem(c, A, b; optimizer=Mosek.Optimizer)
    m, n_x = size(A)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, x[1:n_x] >= 0)
    @objective(model, Min, c' * x)
    @constraint(model, A * x .<= b)
    optimize!(model)
    if termination_status(model) != OPTIMAL
        @warn "Nominal problem not optimal. Returning zeros."
        return zeros(n_x)
    end
    return value.(x)
end

# =============================================================================
# 6. Cutting Plane Algorithm
# =============================================================================
function cutting_plane_algorithm(c, A, b, s, gamma;
    method="heuristic",
    max_iter=5000,
    tol=1e-4,
    limit_time=1000.0,
    verbose=true,
    optimizer=Mosek.Optimizer,
    x_init=nothing,
    seed::Int=23,
    n_x::Int,
    m::Int
)
    S = Vector{Vector{Float64}}()
    tau_history = Vector{Tuple{Int, Float64}}()

    if isnothing(x_init)
        nominal_time_local = @elapsed x_current = solve_nominal_problem(c, A, b; optimizer=optimizer)
        total_algorithm_time = nominal_time_local
        if verbose
            println("(Initialized from nominal solution (Seed $seed | n_x=$n_x, m=$m | $method). Nominal time: $(round(nominal_time_local, digits=4))s")
        end
    else
        x_current = x_init
        total_algorithm_time = 0.0
        if verbose
            println("(Initialized from provided solution (Seed $seed | n_x=$n_x, m=$m | $method).")
        end
    end

    τ_prev = Inf

    for ℓ in 1:max_iter
        if total_algorithm_time > limit_time
            if verbose
                println("⏰ Time limit ($limit_time seconds) exceeded at iteration $ℓ ($method | Seed $seed | n_x=$n_x, m=$m)")
            end
            break
        end

        iter_time = 0.0
        new_scenarios = Vector{Vector{Float64}}()

        t_sub = @elapsed u = worst_case_subproblem(gamma, s, x_current; method=method, optimizer=optimizer)
        iter_time += t_sub
        push!(new_scenarios, u)

        append!(S, new_scenarios)

        t_master = @elapsed x_opt, τ_current = solve_master_problem(c, A, b, S; optimizer=optimizer)
        iter_time += t_master
        total_algorithm_time += iter_time

        push!(tau_history, (ℓ, τ_current))
        x_current = x_opt

        Δτ = ℓ > 1 ? abs(τ_current - τ_prev) : "N/A"

        if ℓ > 1 && isfinite(τ_current) && isfinite(τ_prev) && abs(τ_current - τ_prev) <= tol
            if verbose
                println("✅ Converged at iteration $ℓ: Δτ = $(abs(τ_current - τ_prev)) ($method | Seed $seed | n_x=$n_x, m=$m)")
            end
            break
        end

        τ_prev = τ_current

        if verbose && (ℓ == 1 || ℓ % 10 == 0)
            τ_str = isfinite(τ_current) ? round(τ_current, digits=6) : "Inf"
            delta_str = Δτ isa String ? Δτ : (isfinite(Δτ) ? round(Δτ, digits=6) : "Inf")
            println("Iteration $ℓ ($method | Seed $seed | n_x=$n_x, m=$m): τ = $τ_str | Δτ = $delta_str | #Scenarios = $(length(S)) | Iter time = $(round(iter_time, digits=4))s")
        end
    end

    # Save τ history with n_x and m in filename
    CSV.write("tau_history_$(method)_s$(s)_seed$(seed)_nx$(n_x)_m$(m).csv",
              DataFrame(iteration=[t[1] for t in tau_history], tau=[t[2] for t in tau_history]))
    if verbose
        println("💾 Saved τ history: tau_history_$(method)_s$(s)_seed$(seed)_nx$(n_x)_m$(m).csv")
    end

    if verbose
        final_τ = isfinite(τ_prev) ? round(τ_prev, digits=6) : "Inf"
        println("\n⏱️ TOTAL CUTTING PLANE TIME ($method | Seed $seed | n_x=$n_x, m=$m): $(round(total_algorithm_time, digits=4)) seconds")
        println("📊 Final objective (τ): $final_τ")
    end

    return x_current, τ_prev, S, total_algorithm_time
end

println("✅ All Julia functions defined for linear optimization with implementation error.")

# =============================================================================
# 7. RUN EXPERIMENT FOR 5 SEEDS
# =============================================================================
problem_configs = [
    (n_x=100, m=30),
    (n_x=100, m=20)
]

SEEDS = [23, 24, 25, 26, 27]
methods = ["heuristic", "mip", "relaxed"]
sparsities = [2,5,10]
gamma = 1.0

all_results = []

println("\n" * "^"^60)
println("🚀 STARTING CUTTING PLANE ALGORITHM FOR LINEAR OPTIMIZATION (ALL METHODS, 5 SEEDS)")
println("📏 Testing problem configurations: $(problem_configs)")
println("^"^60)

for (config_idx, config) in enumerate(problem_configs)
    n_x, m = config.n_x, config.m

    println("\n" * "="^60)
    println("📏 PROBLEM CONFIGURATION $config_idx: n_x = $n_x, m = $m")
    println("="^60)

    for seed in SEEDS
        println("\n" * "#" * "="^58 * "#")
        println("### 🌱 Starting Experiment for SEED = $seed | n_x = $n_x, m = $m ###")
        println("#" * "="^58 * "#")

        # Generate and save data with n_x and m in filenames
        A, b, c = generate_lp_data(n_x, m; seed=seed)
        CSV.write("A_nx$(n_x)_m$(m)_seed$(seed).csv", DataFrame(A, :auto))
        CSV.write("b_nx$(n_x)_m$(m)_seed$(seed).csv", DataFrame(b=b))
        CSV.write("c_nx$(n_x)_m$(m)_seed$(seed).csv", DataFrame(c=c))
        println("✅ Generated and saved synthetic data (seed=$seed, n_x=$n_x, m=$m).")

        # Solve nominal problem
        println("🧠 Computing nominal solution (seed=$seed, n_x=$n_x, m=$m)...")
        nominal_time_shared = @elapsed x_nominal = solve_nominal_problem(c, A, b; optimizer=Mosek.Optimizer)
        nominal_obj = c' * x_nominal
        println("✅ Nominal solution computed in $(round(nominal_time_shared, digits=4))s")
        println("📈 Nominal objective (c'x): $(round(nominal_obj, digits=6))")

        # Run experiments
        for method in methods
            for s in sparsities
                println("\n" * "-"^50)
                println("▶️ Solving for sparsity s = $s | Method = $method | Seed = $seed | n_x = $n_x, m = $m")
                println("-"^50)

                total_elapsed = @elapsed begin
                    x_opt, τ_opt, scenarios, algo_time = cutting_plane_algorithm(
                        c, A, b, s, gamma;
                        method=method,
                        max_iter=5000,
                        tol=1e-4,
                        limit_time=1000.0,
                        x_init=x_nominal,
                        seed=seed,
                        n_x=n_x,
                        m=m
                    )
                end

                push!(all_results, (
                    seed = seed,
                    n_x = n_x,
                    m = m,
                    method = method,
                    sparsity = s,
                    objective = τ_opt,
                    scenarios = length(scenarios),
                    algo_time = algo_time,
                    nominal_obj = nominal_obj
                ))

                final_τ_str = isfinite(τ_opt) ? round(τ_opt, digits=6) : "Inf"
                println("🏁 $method (Seed $seed, n_x=$n_x, m=$m): FINAL: τ = $final_τ_str | #Scenarios = $(length(scenarios)) | CP Time = $(round(algo_time, digits=4))s")

                # Save x_opt with n_x and m in filename
                df_x = DataFrame(index=1:length(x_opt), x=x_opt)
                CSV.write("x_opt_$(method)_s$(s)_seed$(seed)_nx$(n_x)_m$(m).csv", df_x)
                println("💾 Saved: x_opt_$(method)_s$(s)_seed$(seed)_nx$(n_x)_m$(m).csv")
            end
        end
    end
end

# =============================================================================
# 8. RESULTS ANALYSIS
# =============================================================================
println("\n📊 FINAL AGGREGATE REPORT (Across all 5 Seeds and Configurations)")

if isempty(all_results)
    println("❌ No results collected.")
else
    df_final = DataFrame(all_results)
    rename!(df_final, :sparsity => :s, :objective => :LB, :scenarios => Symbol("#Scenarios"), :algo_time => Symbol("Time (s)"))
    sort!(df_final, [:n_x, :m, :seed, :s, :method])

    # --- COMPUTE AND ADD SGAP COLUMN ---
    # Group by n_x, m, seed, and sparsity (s)
    gdf = groupby(df_final, [:n_x, :m, :seed, :s])
    # Compute the best LB for each group
    transform!(gdf, :LB => (LB -> maximum(LB)) => :LB_best, renamecols=false)
    # Compute SGap (%) = (LB_best - LB) / (|LB| + 1e-4) * 100
    transform!(gdf, [:LB_best, :LB] => ((LB_best, LB) -> (LB_best .- LB) ./ (abs.(LB) .+ 1e-4) .* 100) => Symbol("SGap (%)"), renamecols=false)
    # Drop the temporary LB_best column
    select!(df_final, Not(:LB_best))

    # --- Save Detailed Results ---
    for (n_x, m) in problem_configs
        df_final_subset = filter(row -> row.n_x == n_x && row.m == m, df_final)
        CSV.write("final_report_aggregate_nx$(n_x)_m$(m).csv", df_final_subset)
        println("💾 Saved detailed report: final_report_aggregate_nx$(n_x)_m$(m).csv")
    end

    # --- Aggregate Summary Statistics ---
    for (n_x, m) in problem_configs
        df_subset = filter(row -> row.n_x == n_x && row.m == m, df_final)
        df_summary = combine(
            groupby(df_subset, [:s, :method]),
            :LB => (x -> round(mean(filter(isfinite, x)), digits=4)) => Symbol("Avg LB"),
            Symbol("Time (s)") => (x -> round(mean(x), digits=2)) => Symbol("Avg Time (s)"),
            Symbol("Time (s)") => (x -> round(std(x), digits=2)) => Symbol("Std Time (s)"),
            Symbol("SGap (%)") => (x -> round(mean(x), digits=2)) => Symbol("Avg SGap (%)"),
            Symbol("SGap (%)") => (x -> round(std(x), digits=2)) => Symbol("Std SGap (%)")
        )
        sort!(df_summary, [:s, :method])

        # Display summary table
        header = ["s", "Method", "Avg LB", "Avg Time (s)", "Std Time (s)", "Avg SGap (%)", "Std SGap (%)"]
        println("\n--- Summary Statistics (Averaged Across 5 Seeds) — Problem Size: n_x = $n_x, m = $m ---")
        try
            pretty_table(
                df_summary,
                header = header,
                title = "Average Performance (5 Seeds, n_x = $n_x, m = $m)",
                alignment = :l,
                tf = tf_unicode
            )
        catch e
            println("⚠️ Error in pretty_table: $e")
            println("📋 Falling back to manual table printing:")
            println(join(header, " | "))
            println("-"^80)
            for row in eachrow(df_summary)
                println(join([row[:s], row[:method], row[Symbol("Avg LB")], row[Symbol("Avg Time (s)")], row[Symbol("Std Time (s)")], row[Symbol("Avg SGap (%)")], row[Symbol("Std SGap (%)")]], " | "))
            end
        end

        # Save summary report
        CSV.write("final_report_summary_nx$(n_x)_m$(m).csv", df_summary)
        println("💾 Saved summary report: final_report_summary_nx$(n_x)_m$(m).csv")
    end

    println("\n🎉 Experiment completed for all 5 seeds and all problem configurations.")
end





#Part 2
 println("\n--- Part 2 ---")
# =============================================================================
# Linear Optimization with Implementation Error
# Complete Implementation with Cutting-Plane Algorithm
# =============================================================================

using JuMP, MosekTools, CSV, DataFrames, Distributions, LinearAlgebra, Printf, Random, PrettyTables

# =============================================================================
# 1. Generate Synthetic Data
# =============================================================================
function generate_lp_data(n_x, m; seed=23)
    Random.seed!(seed)
    A = rand(Uniform(0.0, 1.0), m, n_x)
    b = rand(Uniform(10.0, 20.0), m)
    c = rand(Uniform(-1.0, 1.0), n_x)
    return A, b, c
end

# =============================================================================
# 2. Sparse Projection onto {‖u‖₀ ≤ s}
# =============================================================================
function sparse_project(u_tilde, s)
    n = length(u_tilde)
    if s >= n
        return u_tilde
    end
    idx = sortperm(abs.(u_tilde), rev=true)[1:s]
    u_proj = zeros(n)
    for i in idx
        u_proj[i] = u_tilde[i]
    end
    return u_proj
end

# =============================================================================
# 3. Worst-Case Subproblem
# =============================================================================
function worst_case_subproblem(gamma, s, x_current; method="heuristic", optimizer=Mosek.Optimizer)
    n_x = length(x_current)
    model = Model(optimizer)
    set_silent(model)

    if method == "heuristic"
        norm_x = norm(x_current, 2)
        if norm_x > 1e-10
            u_tilde = gamma * x_current / norm_x
        else
            u_tilde = zeros(n_x)
        end
        return sparse_project(u_tilde, s)

    elseif method == "mip"
        @variable(model, u[1:n_x])
        @variable(model, z[1:n_x], Bin)
        @constraint(model, sum(z) <= s)
        for i in 1:n_x
            @constraint(model, u[i] <= gamma * z[i])
            @constraint(model, u[i] >= -gamma * z[i])
        end
        @constraint(model, sum(u[i]^2 for i in 1:n_x) <= gamma^2)
        @objective(model, Max, sum(u[i] * x_current[i] for i in 1:n_x))
        optimize!(model)
        if termination_status(model) != OPTIMAL
            @warn "MIP subproblem not optimal."
            return zeros(n_x)
        end
        return value.(u)

    elseif method == "relaxed"
        @variable(model, u[1:n_x])
        @variable(model, 0 <= z[1:n_x] <= 1)
        @constraint(model, sum(z) <= s)
        for i in 1:n_x
            @constraint(model, u[i] <= gamma * z[i])
            @constraint(model, u[i] >= -gamma * z[i])
        end
        @constraint(model, sum(u[i]^2 for i in 1:n_x) <= gamma^2)
        @objective(model, Max, sum(u[i] * x_current[i] for i in 1:n_x))
        optimize!(model)
        if termination_status(model) != OPTIMAL
            @warn "Relaxed subproblem not optimal."
            return zeros(n_x)
        end
        u_tilde = value.(u)
        return sparse_project(u_tilde, s)

    else
        error("Unknown method: $method. Use 'heuristic', 'mip', or 'relaxed'.")
    end
end

# =============================================================================
# 4. Solve Master Problem
# =============================================================================
function solve_master_problem(c, A, b, scenarios; optimizer=Mosek.Optimizer)
    m, n_x = size(A)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, x[1:n_x] >= 0)
    @objective(model, Min, c' * x)

    for j in 1:m
        for u in scenarios
            @constraint(model, A[j, :]' * x + u' * x <= b[j])
        end
    end

    optimize!(model)
    if termination_status(model) != OPTIMAL
        @warn "Master problem not optimal."
        return zeros(n_x), Inf
    end
    return value.(x), objective_value(model)
end

# =============================================================================
# 5. Solve Nominal Problem (No uncertainty)
# =============================================================================
function solve_nominal_problem(c, A, b; optimizer=Mosek.Optimizer)
    m, n_x = size(A)
    model = Model(optimizer)
    set_silent(model)
    @variable(model, x[1:n_x] >= 0)
    @objective(model, Min, c' * x)
    @constraint(model, A * x .<= b)
    optimize!(model)
    if termination_status(model) != OPTIMAL
        @warn "Nominal problem not optimal. Returning zeros."
        return zeros(n_x)
    end
    return value.(x)
end

# =============================================================================
# 6. Cutting Plane Algorithm
# =============================================================================
function cutting_plane_algorithm(c, A, b, s, gamma;
    method="heuristic",
    max_iter=5000,
    tol=1e-4,
    limit_time=1000.0,
    verbose=true,
    optimizer=Mosek.Optimizer,
    x_init=nothing,
    seed::Int=23,
    n_x::Int,
    m::Int
)
    S = Vector{Vector{Float64}}()
    tau_history = Vector{Tuple{Int, Float64}}()

    if isnothing(x_init)
        nominal_time_local = @elapsed x_current = solve_nominal_problem(c, A, b; optimizer=optimizer)
        total_algorithm_time = nominal_time_local
        if verbose
            println("(Initialized from nominal solution (Seed $seed | n_x=$n_x, m=$m | $method). Nominal time: $(round(nominal_time_local, digits=4))s")
        end
    else
        x_current = x_init
        total_algorithm_time = 0.0
        if verbose
            println("(Initialized from provided solution (Seed $seed | n_x=$n_x, m=$m | $method).")
        end
    end

    τ_prev = Inf

    for ℓ in 1:max_iter
        if total_algorithm_time > limit_time
            if verbose
                println("⏰ Time limit ($limit_time seconds) exceeded at iteration $ℓ ($method | Seed $seed | n_x=$n_x, m=$m)")
            end
            break
        end

        iter_time = 0.0
        new_scenarios = Vector{Vector{Float64}}()

        t_sub = @elapsed u = worst_case_subproblem(gamma, s, x_current; method=method, optimizer=optimizer)
        iter_time += t_sub
        push!(new_scenarios, u)

        append!(S, new_scenarios)

        t_master = @elapsed x_opt, τ_current = solve_master_problem(c, A, b, S; optimizer=optimizer)
        iter_time += t_master
        total_algorithm_time += iter_time

        push!(tau_history, (ℓ, τ_current))
        x_current = x_opt

        Δτ = ℓ > 1 ? abs(τ_current - τ_prev) : "N/A"

        if ℓ > 1 && isfinite(τ_current) && isfinite(τ_prev) && abs(τ_current - τ_prev) <= tol
            if verbose
                println("✅ Converged at iteration $ℓ: Δτ = $(abs(τ_current - τ_prev)) ($method | Seed $seed | n_x=$n_x, m=$m)")
            end
            break
        end

        τ_prev = τ_current

        if verbose && (ℓ == 1 || ℓ % 10 == 0)
            τ_str = isfinite(τ_current) ? round(τ_current, digits=6) : "Inf"
            delta_str = Δτ isa String ? Δτ : (isfinite(Δτ) ? round(Δτ, digits=6) : "Inf")
            println("Iteration $ℓ ($method | Seed $seed | n_x=$n_x, m=$m): τ = $τ_str | Δτ = $delta_str | #Scenarios = $(length(S)) | Iter time = $(round(iter_time, digits=4))s")
        end
    end

    # Save τ history with n_x and m in filename
    CSV.write("tau_history_$(method)_s$(s)_seed$(seed)_nx$(n_x)_m$(m).csv",
              DataFrame(iteration=[t[1] for t in tau_history], tau=[t[2] for t in tau_history]))
    if verbose
        println("💾 Saved τ history: tau_history_$(method)_s$(s)_seed$(seed)_nx$(n_x)_m$(m).csv")
    end

    if verbose
        final_τ = isfinite(τ_prev) ? round(τ_prev, digits=6) : "Inf"
        println("\n⏱️ TOTAL CUTTING PLANE TIME ($method | Seed $seed | n_x=$n_x, m=$m): $(round(total_algorithm_time, digits=4)) seconds")
        println("📊 Final objective (τ): $final_τ")
    end

    return x_current, τ_prev, S, total_algorithm_time
end

println("✅ All Julia functions defined for linear optimization with implementation error.")

# =============================================================================
# 7. RUN EXPERIMENT FOR 5 SEEDS
# =============================================================================
problem_configs = [
    (n_x=300, m=30),
    (n_x=300, m=20)
]

SEEDS = [23, 24, 25, 26, 27]
methods = ["heuristic", "mip", "relaxed"]
sparsities = [2,5,10]
gamma = 1.0

all_results = []

println("\n" * "^"^60)
println("🚀 STARTING CUTTING PLANE ALGORITHM FOR LINEAR OPTIMIZATION (ALL METHODS, 5 SEEDS)")
println("📏 Testing problem configurations: $(problem_configs)")
println("^"^60)

for (config_idx, config) in enumerate(problem_configs)
    n_x, m = config.n_x, config.m

    println("\n" * "="^60)
    println("📏 PROBLEM CONFIGURATION $config_idx: n_x = $n_x, m = $m")
    println("="^60)

    for seed in SEEDS
        println("\n" * "#" * "="^58 * "#")
        println("### 🌱 Starting Experiment for SEED = $seed | n_x = $n_x, m = $m ###")
        println("#" * "="^58 * "#")

        # Generate and save data with n_x and m in filenames
        A, b, c = generate_lp_data(n_x, m; seed=seed)
        CSV.write("A_nx$(n_x)_m$(m)_seed$(seed).csv", DataFrame(A, :auto))
        CSV.write("b_nx$(n_x)_m$(m)_seed$(seed).csv", DataFrame(b=b))
        CSV.write("c_nx$(n_x)_m$(m)_seed$(seed).csv", DataFrame(c=c))
        println("✅ Generated and saved synthetic data (seed=$seed, n_x=$n_x, m=$m).")

        # Solve nominal problem
        println("🧠 Computing nominal solution (seed=$seed, n_x=$n_x, m=$m)...")
        nominal_time_shared = @elapsed x_nominal = solve_nominal_problem(c, A, b; optimizer=Mosek.Optimizer)
        nominal_obj = c' * x_nominal
        println("✅ Nominal solution computed in $(round(nominal_time_shared, digits=4))s")
        println("📈 Nominal objective (c'x): $(round(nominal_obj, digits=6))")

        # Run experiments
        for method in methods
            for s in sparsities
                println("\n" * "-"^50)
                println("▶️ Solving for sparsity s = $s | Method = $method | Seed = $seed | n_x = $n_x, m = $m")
                println("-"^50)

                total_elapsed = @elapsed begin
                    x_opt, τ_opt, scenarios, algo_time = cutting_plane_algorithm(
                        c, A, b, s, gamma;
                        method=method,
                        max_iter=5000,
                        tol=1e-4,
                        limit_time=1000.0,
                        x_init=x_nominal,
                        seed=seed,
                        n_x=n_x,
                        m=m
                    )
                end

                push!(all_results, (
                    seed = seed,
                    n_x = n_x,
                    m = m,
                    method = method,
                    sparsity = s,
                    objective = τ_opt,
                    scenarios = length(scenarios),
                    algo_time = algo_time,
                    nominal_obj = nominal_obj
                ))

                final_τ_str = isfinite(τ_opt) ? round(τ_opt, digits=6) : "Inf"
                println("🏁 $method (Seed $seed, n_x=$n_x, m=$m): FINAL: τ = $final_τ_str | #Scenarios = $(length(scenarios)) | CP Time = $(round(algo_time, digits=4))s")

                # Save x_opt with n_x and m in filename
                df_x = DataFrame(index=1:length(x_opt), x=x_opt)
                CSV.write("x_opt_$(method)_s$(s)_seed$(seed)_nx$(n_x)_m$(m).csv", df_x)
                println("💾 Saved: x_opt_$(method)_s$(s)_seed$(seed)_nx$(n_x)_m$(m).csv")
            end
        end
    end
end

# =============================================================================
# 8. RESULTS ANALYSIS
# =============================================================================
println("\n📊 FINAL AGGREGATE REPORT (Across all 5 Seeds and Configurations)")

if isempty(all_results)
    println("❌ No results collected.")
else
    df_final = DataFrame(all_results)
    rename!(df_final, :sparsity => :s, :objective => :τ, :scenarios => Symbol("#Scenarios"), :algo_time => Symbol("Time (s)"))
    sort!(df_final, [:n_x, :m, :seed, :s, :method])

    # Summary table
    df_summary = combine(
        groupby(df_final, [:n_x, :m, :s, :method]),
        :τ => (x -> begin
            finite_vals = filter(isfinite, x)
            if length(finite_vals) == 0
                "Inf"
            else
                round(mean(finite_vals), digits=4)
            end
        end) => Symbol("LB (τ)"),
        Symbol("Time (s)") => (x -> round(mean(x), digits=2)) => Symbol("Time (s)")
    )
    sort!(df_summary, [:n_x, :m, :s, :method])

    println("\n--- Summary Statistics (Averaged Across 5 Seeds) ---")
    pretty_table(df_summary, title="Cutting-Plane Algorithm Performance",
                 header=["n_x", "m", "s", "Method", "LB (τ)", "Time (s)"])

    # Save final reports with n_x and m
    CSV.write("final_report_aggregate_linear_nx_m_2.csv", df_final)
    CSV.write("final_report_summary_linear_nx_m_2.csv", df_summary)
    println("💾 Saved detailed report: final_report_aggregate_linear_nx_m.csv")
    println("💾 Saved summary report: final_report_summary_linear_nx_m.csv")

    println("\n🎉 Experiment completed for all 5 seeds and all problem configurations.")
end

# =============================================================================
# 8. RESULTS ANALYSIS
# =============================================================================
println("\n📊 FINAL AGGREGATE REPORT (Across all 5 Seeds and Configurations)")

if isempty(all_results)
    println("❌ No results collected.")
else
    df_final = DataFrame(all_results)
    rename!(df_final, :sparsity => :s, :objective => :LB, :scenarios => Symbol("#Scenarios"), :algo_time => Symbol("Time (s)"))
    sort!(df_final, [:n_x, :m, :seed, :s, :method])

    # --- COMPUTE AND ADD SGAP COLUMN ---
    # Group by n_x, m, seed, and sparsity (s) to compute the best LB
    gdf = groupby(df_final, [:n_x, :m, :seed, :s])
    transform!(gdf, :LB => (LB -> maximum(filter(isfinite, LB))) => :LB_best, renamecols=false)
    # Compute SGap (%) = (LB_best - LB) / (|LB| + 1e-4) * 100
    transform!(gdf, [:LB_best, :LB] => ((LB_best, LB) -> (LB_best .- LB) ./ (abs.(LB) .+ 1e-4) .* 100) => Symbol("SGap (%)"), renamecols=false)
    # Drop the temporary LB_best column
    select!(df_final, Not(:LB_best))

    # --- Save Detailed Results ---
    for (n_x, m) in problem_configs
        df_final_subset = filter(row -> row.n_x == n_x && row.m == m, df_final)
        CSV.write("final_report_aggregate_nx$(n_x)_m$(m).csv", df_final_subset)
        println("💾 Saved detailed report: final_report_aggregate_nx$(n_x)_m$(m).csv")
    end

    # --- Aggregate Summary Statistics ---
    for (n_x, m) in problem_configs
        df_subset = filter(row -> row.n_x == n_x && row.m == m, df_final)
        df_summary = combine(
            groupby(df_subset, [:s, :method]),
            :LB => (x -> round(mean(filter(isfinite, x)), digits=4)) => Symbol("Avg LB"),
            :LB => (x -> round(std(filter(isfinite, x)), digits=4)) => Symbol("Std LB"),
            Symbol("Time (s)") => (x -> round(mean(x), digits=2)) => Symbol("Avg Time (s)"),
            Symbol("Time (s)") => (x -> round(std(x), digits=2)) => Symbol("Std Time (s)"),
            Symbol("SGap (%)") => (x -> round(mean(x), digits=2)) => Symbol("Avg SGap (%)"),
            Symbol("SGap (%)") => (x -> round(std(x), digits=2)) => Symbol("Std SGap (%)")
        )
        sort!(df_summary, [:s, :method])

        # Display summary table
        header = ["s", "Method", "Avg LB", "Std LB", "Avg Time (s)", "Std Time (s)", "Avg SGap (%)", "Std SGap (%)"]
        println("\n--- Summary Statistics (Averaged Across 5 Seeds) — Problem Size: n_x = $n_x, m = $m ---")
        try
            pretty_table(
                df_summary,
                header = header,
                title = "Average Performance (5 Seeds, n_x = $n_x, m = $m)",
                alignment = :l,
                tf = tf_unicode
            )
        catch e
            println("⚠️ Error in pretty_table: $e")
            println("📋 Falling back to manual table printing:")
            println(join(header, " | "))
            println("-"^80)
            for row in eachrow(df_summary)
                println(join([row[:s], row[:method], row[Symbol("Avg LB")], row[Symbol("Std LB")], row[Symbol("Avg Time (s)")], row[Symbol("Std Time (s)")], row[Symbol("Avg SGap (%)")], row[Symbol("Std SGap (%)")]], " | "))
            end
        end

        # Save summary report
        CSV.write("final_report_summary_nx$(n_x)_m$(m).csv", df_summary)
        println("💾 Saved summary report: final_report_summary_nx$(n_x)_m$(m).csv")
    end

    println("\n🎉 Experiment completed for all 5 seeds and all problem configurations.")
end