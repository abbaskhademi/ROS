# ============================================================
# --- 0. Package Setup ---
# ============================================================
import Pkg
 required_packages = ["JuMP", "MosekTools", "CSV", "DataFrames", "Distributions", "LinearAlgebra", "Printf", "Random","HTTP","PrettyTables","XLSX", "Optim", "LineSearches","Statistics"]
for pkg in required_packages
    try
        eval(Meta.parse("using $pkg"))
    catch e
        @info "$pkg not found, installing..."
        import Pkg
        Pkg.add(pkg)
        eval(Meta.parse("using $pkg"))
    end
end
#------------------------------------------
using Random, Distributions, LinearAlgebra, DataFrames, CSV, Printf
using Optim, LineSearches
using JuMP, MosekTools
using XLSX, PrettyTables, Statistics
#------------------------------------------
# ------------------------------------------------------------
# Output directory: current location from which the code is run
# ------------------------------------------------------------
const OUTPUT_DIR = abspath(@__DIR__)

println("Current working directory: ", OUTPUT_DIR)
println("All output files will be saved in this directory.")
# Set global random seed for reproducibility of package internals if needed
Random.seed!(2024)
#----------------------------------------
# ============================================================
# --- 1. Data Generation ---
# ============================================================
function generate_portfolio_data(n_x, n_u; seed=2024, min_L_est::Real=1.05)
    Random.seed!(seed)

    # Expected returns
    mu = rand(Uniform(0.05, 0.15), n_x)

    # Covariance matrix Sigma (Factor model + Idiosyncratic risk)
    n_factors = min(n_x, 50)
    B_factor = randn(n_x, n_factors) * 0.1
    F = Diagonal(rand(Uniform(0.02, 0.05), n_factors))
    D = Diagonal(rand(Uniform(0.01, 0.03), n_x))
    Sigma = B_factor * F * B_factor' + D
    Sigma = (Sigma + Sigma') / 2
    Sigma += 1e-6 * I

    # Factor-exposure matrix B for stress factors
    B = rand(Uniform(0.01, 0.09), n_u, n_x)

    # Q: Stress intensity matrix, normalized to be O(1) in spectral norm
    # regardless of n_u. L has nonzero-mean entries (Uniform(0,1)), so
    # unnormalized L'*L has a dominant rank-1 eigenvalue ≈ 0.25*n_u^2 that
    # swamps the linear term u'Bx as n_u grows -- this is what drove
    # τ → 0 and the near-instant spurious "convergence". Dividing L by
    # n_u (not sqrt(n_u) -- the mean-spike, not the MP bulk, is what
    # dominates here) keeps λ_max(Q) ≈ 0.25 for any n_u.
    L = rand(n_u, n_u) / n_u
    Q = Symmetric(L' * L)

    # --- Enforce λ_min(Q) ≥ 0.1 deterministically (guarantees Q ≻ 0) ---
    λmin = eigmin(Q)
    if λmin < 0.1
        Q = Symmetric(Q + (0.1 - λmin) * I)
    end

    # --- Enforce L_est = 2*λ_max(Q) > 1, with a small safety margin ---
    # (now actually binding: λ_max(Q) ~ 0.25-0.35 pre-floor, not ~1e4-1e5)
    λmax = eigmax(Q)
    target_λmax = min_L_est / 2
    if λmax < target_λmax
        Q = Symmetric(Q * (target_λmax / λmax))
    end

    return mu, Sigma, B, Matrix(Q)
end


function generate_portfolio_data00(n_x, n_u; seed=2024)
    Random.seed!(seed)

    # Expected returns
    mu = rand(Uniform(0.05, 0.15), n_x)

    # Covariance matrix Sigma (Factor model + Idiosyncratic risk)
    n_factors = min(n_x, 50)
    B_factor = randn(n_x, n_factors) * 0.1
    F = Diagonal(rand(Uniform(0.02, 0.05), n_factors))
    D = Diagonal(rand(Uniform(0.01, 0.03), n_x))
    Sigma = B_factor * F * B_factor' + D
    Sigma = (Sigma + Sigma') / 2
    Sigma += 1e-6 * I

    # Factor-exposure matrix B for stress factors
    B = rand(Uniform(0.01, 0.09), n_u, n_x)

    # Q: Stress intensity matrix (Guaranteed SPD by strict diagonal dominance)
    L = rand(n_u,n_u);
    Q = L'*L;
    Q += (max(1.0 - eigmin(Q), 0)) * I    # enforce λ_min ≥ 1.0 deterministically
    return mu, Sigma, B, Q
end
# --- 3. Worst-Case Subproblem (All Variants) ---
#for a give x: f(u) = u^\top Q u - 2u^\top B x
# ============================================================
# --- 2. Worst-Case Subproblem Solvers ---
# ============================================================

# --- Iterative Hard Thresholding (IHT) ---
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
        #if i % 10 == 0
        #    @printf("IHT: iter = %5d, f(x) = %5.4f\n", i, fx)
        #end
        # Existing stopping criterion
        if norm(x - x_prev) <= epsilon
         #   @printf("🛑 Stopped IHT early at iter %d: ||x_%d - x_%d|| = %.2e <= %.2e\n",
         #           i, i, i-1, norm(x - x_prev), epsilon)
         #   iters = i
         #   @printf("IHT: iter = %5d, f(x) = %5.4f\n", i, fx)
            break
        end
    end
    fx = f(x)
    elapsed_time = time() - start_time

    #println("  Elapsed time: $(round(elapsed_time, digits=4)) seconds")
    #flush(stdout)
    return x, fx, elapsed_time, iters
end

# --- Adaptive Gradient Projection (AGP) ---
function AGP(f, g, s, x0, N; gamma=1/4, delta=1e-10, beta=2.0, epsilon=1e-5)
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

    # Enforce sparsity on initial point if needed
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

    # Initialize prev_grad and x_prev to avoid UndefVarError
    x_prev = copy(x_curr)
    prev_grad = copy(current_grad)

    for k in 1:N
        # Estimate Lipschitz constant L_k
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
            if x_diff_norm < 1e-10  # Safeguard against zero division (though stopping criterion should catch)
                break
            end
            L_k = gamma * (grad_diff_norm / x_diff_norm + delta)
        end
        push!(L_ks, L_k)

        # Backtracking line search
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

        # Update for next iteration
        x_prev = copy(x_curr)
        prev_grad = copy(current_grad)
        x_curr = x_next
        fx = f(x_next)
        current_grad = g(x_curr)
        iterations = k
        push!(fx_values, fx)
        push!(backtrack_counts, backtrack_count)
        push!(recent_backtracks, backtrack_count)

        # Adaptive gamma adjustment every 10 iterations
        if k % 10 == 0
            total_backtracks = sum(recent_backtracks)
            if total_backtracks == 0
                gamma = gamma * 0.9  # Decrease gamma, with a lower bound
            elseif total_backtracks > 10
                gamma = min(1, gamma * 1.1)  # Increase gamma, with an upper bound
            end
            push!(gamma_history, gamma)
            recent_backtracks = Int[]  # Reset for the next 10 iterations
        end

        # Logging every 500 iterations
        #if k % 10 == 0
        #    @printf("adaptive_IHT: iter = %5d, f(x) = %5.4f\n", k, fx)
        #end

        # Early stopping based on iterate change
        if norm(x_curr - x_prev) <= epsilon
        #    @printf("🛑 Adaptive IHT stopped early: ||x_%d - x_%d|| = %.2e <= %.2e\n",
        #            k, k-1, norm(x_curr - x_prev), epsilon)
        #    iterations = k
        #    @printf("APG: iter = %5d, f(x) = %5.4f\n", k, fx)
            break
        end
       # @printf("adaptive_IHT: iter = %5d, f(x) = %5.4f\n", k, fx)
    end

    elapsed_time = time() - start_time

  #  println("Adaptive IHT completed:")
  #  println("  Final objective: $(round(fx, digits=8))")
  #  println("  Total iterations: $iterations")
  #  println("  Total backtracking steps: $(sum(backtrack_counts))")
  #  println("  Final gamma: $(round(gamma, digits=4))")
  #  println("  Elapsed time: $(round(elapsed_time, digits=4)) seconds")

    return x_curr, fx, elapsed_time, iterations
end


function PSS(f, grad_f, s, N, x0; epsilon=1e-5)
    start_time = time()
    n = length(x0)
    x = copy(x0)

    # Initial hard‑thresholding (ensure x in C_s)
    if count(!iszero, x) > s
        perm = sortperm(abs.(x))
        keep = perm[(n-s+1):n]
        x_new = zeros(n)
        x_new[keep] = x[keep]
        x = x_new
    end

    fx = f(x)
    iters = N
    for k in 1:N
        x_prev = copy(x)
        sparsity = count(!iszero, x)

        # ==============================
        #  Case 1: ||x||_0 < s
        # ==============================
        if sparsity < s
            t_i = zeros(n)
            f_i = fill(Inf, n)
            for i in 1:n
                # solve min_t f(x + t*e_i)
                model_i = Model(Mosek.Optimizer)
                set_silent(model_i)
                @variable(model_i, t)
                e_i = zeros(n); e_i[i] = 1.0
                @objective(model_i, Min, f(x + t * e_i))
                optimize!(model_i)
                t_i[i] = value(t)
                f_i[i] = objective_value(model_i)
            end
            # pick the coordinate that gives the smallest f_i
            i_k = argmin(f_i)   # or argmin of all i
            if f_i[i_k] < fx
                e_ik = zeros(n); e_ik[i_k] = 1.0
                x = x + t_i[i_k] * e_ik
                fx = f(x)
            else
                # no improvement → STOP (as per template)
                iters = k
                break
            end

        # ==============================
        #  Case 2: ||x||_0 == s
        # ==============================
        elseif sparsity == s
            I1 = findall(!iszero, x)
            I0 = setdiff(1:n, I1)

            # 1. Compute t_i, f_i for i in I1
            t_i = zeros(n)
            f_i = fill(Inf, n)
            for i in I1
                model_i = Model(Mosek.Optimizer)
                set_silent(model_i)
                @variable(model_i, t)
                e_i = zeros(n); e_i[i] = 1.0
                @objective(model_i, Min, f(x + t * e_i))
                optimize!(model_i)
                t_i[i] = value(t)
                f_i[i] = objective_value(model_i)
            end

            # 2. i_k^1  ∈ argmax { f_i : i ∈ I1 }   (template) % problem is min f(x) s.t. \|x\|_0<=s argmin or argmax
            # Since f_i is the minimized objective value, maximum decrease <=> argmin f_i.
            i_k1 = I1[argmin(f_i[I1])]   

            # 3. i_k^2  ∈ argmax { |∇_i f(x^k)| : i ∈ I0 }
            grad_f_x = grad_f(x)
            i_k2 = I0[argmax(abs.(grad_f_x[I0]))]

            # 4. m_k ∈ argmin { |x_i^k| : i ∈ I1 }
            m_k = I1[argmin(abs.(x[I1]))]

            # 5. D_k^1 = f_i[i_k1] , T_k^1 = t_i[i_k1]
            D_k1 = f_i[i_k1]
            T_k1 = t_i[i_k1]

            # 6. D_k^2, T_k^2 : min_t f(x - x_{m_k} e_{m_k} + t e_{i_k2})
            model_2 = Model(Mosek.Optimizer)
            set_silent(model_2)
            @variable(model_2, t)
            e_mk = zeros(n); e_mk[m_k] = 1.0
            e_i2 = zeros(n); e_i2[i_k2] = 1.0
            base_vec = x - x[m_k] * e_mk
            @objective(model_2, Min, f(base_vec + t * e_i2))
            optimize!(model_2)
            T_k2 = value(t)
            D_k2 = objective_value(model_2)

            # 7. Update
            if D_k1 < D_k2
                e_i1 = zeros(n); e_i1[i_k1] = 1.0
                x = x + T_k1 * e_i1
            else
                x = base_vec + T_k2 * e_i2
            end
            fx = f(x)

        else
            # sparsity > s should never happen because of the hard‑thresholding
            error("Sparsity exceeded s")
        end

        # Early stopping based on iterate change (optional)
        if norm(x - x_prev) <= epsilon
            iters = k
            break
        end
    end

    elapsed = time() - start_time
    return x, fx, elapsed, iters
end
# --- Partial Sparse-Simplex (PSS) ---
function PSSq(f, grad_f, s, N, x0; epsilon=1e-5)
    optimizer=Mosek.Optimizer
    start_time = time()
    n = length(x0)
    x = copy(x0)

    # --- Initialization: enforce x^0 in C_s (hard thresholding) ---
    if count(!iszero, x) > s
        perm = sortperm(abs.(x))
        keep = perm[(n-s+1):n]
        x_new = zeros(n)
        x_new[keep] = x[keep]
        x = x_new
    end

    fx = f(x)
    iters = N
    for k in 1:N
        x_prev = copy(x)
        supp = findall(!iszero, x)

                # =============== Case 2: ||x||_0 = s ===============
        # Define the index sets I_1(x^k) and I_0(x^k)
        I1 = findall(!iszero, x)
        I0 = setdiff(1:n, I1)

        # 1. Compute for every i in I_1(x^k): t_i and f_i
        t_i = zeros(n)
        f_i = fill(Inf, n)
        
        for i in I1
            model_i = Model(optimizer)
            set_silent(model_i)
            @variable(model_i, t)
            
            e_i = zeros(n); e_i[i] = 1.0
            
            # NOTE: f must be formulated as a valid JuMP expression for Mosek.
            # If f is a black-box function, you must switch to a nonlinear solver like Ipopt.
            @objective(model_i, Min, f(x + t * e_i)) 
            
            optimize!(model_i)
            t_i[i] = value(t)
            f_i[i] = objective_value(model_i) # Corrected from value(f(...))
        end

        # 2. Let i_k^1 in argmin {f_i : i in I_1(x^k)}
        # (Correction: The template states 'argmax', which is a typographical error 
        # for a minimization problem. We use 'argmin' here.)
        i_k1 = I1[argmin(f_i[I1])]
        
        # 3. Let i_k^2 in argmax {|grad_i f(x^k)| : i in I_0(x^k)}
        grad_f_x = grad_f(x)
        i_k2 = I0[argmax(abs.(grad_f_x[I0]))]
        
        # 4. Let m_k in argmin {|x_i^k| : i in I_1(x^k)}
        m_k = I1[argmin(abs.(x[I1]))]

        # 5. Compute D_k^1 and T_k^1
        # (Correction: The template writes e_{i_k}, but contextually it must be e_{i_k^1})
        T_k1 = t_i[i_k1]
        D_k1 = f_i[i_k1]

        # 6. Compute D_k^2 and T_k^2
        model_2 = Model(optimizer)
        set_silent(model_2)
        @variable(model_2, t)
        
        e_mk = zeros(n); e_mk[m_k] = 1.0
        e_i2 = zeros(n); e_i2[i_k2] = 1.0
        
        # Formulate: x^k - x_{m_k}^k e_{m_k} + t e_{i_k^2}
        base_vec = x - x[m_k] * e_mk 
        
        @objective(model_2, Min, f(base_vec + t * e_i2))
        
        optimize!(model_2)
        T_k2 = value(t)
        D_k2 = objective_value(model_2) # Corrected from value(f(...))

        # 7. Update x^{k+1} based on the comparison of D_k^1 and D_k^2
        if D_k1 < D_k2
            e_i1 = zeros(n); e_i1[i_k1] = 1.0
            x = x + T_k1 * e_i1
        else
            x = x - x[m_k] * e_mk + T_k2 * e_i2
        end
        
        fx = f(x)


        # early stopping on iterate change
        if norm(x - x_prev) <= epsilon
            # @printf("🛑 Stopped PSS early at iter %d: ||x_%d - x_%d|| = %.2e <= %.2e\n",
            #         k, k, k-1, norm(x - x_prev), epsilon)
            iters = k
            break
        end
    end

    elapsed = time() - start_time
    #@printf("Elapsed time: %.4f seconds\n", elapsed)
    return x, fx, elapsed, iters
end

# --- Wrapper for Subproblems ---
function worst_case_subproblem(f, g, s, u0, N, method, L_est; epsilon=1e-5)
    if method == "IHT"
        return IHT(f, g, s, L_est, u0, N; epsilon=epsilon)
    elseif method == "AGP"
        return AGP(f, g, s, u0, N; epsilon=epsilon)
    elseif method == "PSS"
        return PSS(f, g, s, N, u0; epsilon=epsilon)
    else
        error("Unknown method: $method")
    end
end
# ============================================================
# --- 3. Master and Nominal Problem Solvers ---
# ============================================================

function solve_master_problem(mu, Sigma, B, Q, scenarios; lambda=0.5, optimizer=Mosek.Optimizer)
    n_x = length(mu)
    model = Model(optimizer)
    set_silent(model)

    @variable(model, x[1:n_x] >= 0)
    @variable(model, t)

    @objective(model, Max, mu' * x - lambda * (x' * Sigma * x) - t)
    @constraint(model, sum(x) == 1)

    for u in scenarios
        @constraint(model, 2 * u' * B * x - u' * Q * u <= t)
    end

    optimize!(model)
    status = termination_status(model)

    if status != MOI.OPTIMAL
        @warn "Model did not solve to optimality! Status: $status"
        return zeros(n_x), -Inf, -Inf, status
    end
    #@printf("UB = %5.4f\n",  objective_value(model))
    return value.(x), objective_value(model), termination_status(model)
end

function solve_nominal_problem(mu, Sigma; lambda=0.5, optimizer=Mosek.Optimizer)
    n_x = length(mu)
    model = Model(optimizer)
    set_silent(model)

    @variable(model, x[1:n_x] >= 0)
    @objective(model, Max, mu' * x - lambda * (x' * Sigma * x))
    @constraint(model, sum(x) == 1)

    optimize!(model)
    if termination_status(model) != MOI.OPTIMAL
        return zeros(n_x), -Inf, termination_status(model)
    end

    return value.(x), objective_value(model), termination_status(model)
end
# ============================================================
# --- 4. Cutting Plane Algorithm ---
# ============================================================
function cutting_plane_algorithm(mu, Sigma, B, Q, s;
    lambda=0.5, method="AGP", max_iter=5000, max_sub_iter=5000,
    tol=1e-5, limit_time=1000.0, verbose=true, optimizer=Mosek.Optimizer,
    x_init=nothing, seed::Int=2024, n_x_dim::Int=0, n_u_dim::Int=0)

    n_u = size(B, 1)
    S = Vector{Vector{Float64}}()
    tau_history = Vector{Tuple{Int, Float64}}()

    if isnothing(x_init)
        x_init, _, _ = solve_nominal_problem(mu, Sigma; lambda=lambda, optimizer=optimizer)
    end

    x_current = x_init
    total_algorithm_time = 0.0
    τ_prev = -Inf
    Random.seed!(seed)
    u0 = randn(n_u)
    lambda_max = maximum(eigvals(Symmetric(Q)))
    L_est = 2.0 * lambda_max

    for ℓ in 1:max_iter
        if total_algorithm_time > limit_time
            if verbose println("Time limit exceeded at iteration $ℓ") end
            break
        end

        Bx = B * x_current
        f(u) = dot(u, Q * u) - 2.0 * dot(u, Bx)
        g(u) = 2.0 * (Q * u) - 2.0 * Bx


        iter_time = @elapsed u_star, fu, _, sub_iters = worst_case_subproblem(f, g, s, u0, max_sub_iter, method, L_est)

        #iter_time = @elapsed u_star, _, _, _ = worst_case_subproblem(f, g, s, u0, max_sub_iter, method, L_est)
        push!(S, u_star)
        #u0 = u_star # Warm start
        Random.seed!(seed)
        u0 = randn(n_u)
        t_master = @elapsed x_opt, τ_current, _ = solve_master_problem(mu, Sigma, B, Q, S; lambda=lambda, optimizer=optimizer)


        if verbose && (ℓ % 10 == 0)
            @printf("Outer ℓ=%d | %s  τ (bound)=%.6f\n", ℓ, method, τ_current)
            flush(stdout)
        end

        iter_time += t_master
        total_algorithm_time += iter_time

        push!(tau_history, (ℓ, τ_current))
        x_current = x_opt

        if ℓ > 1 && abs(τ_current - τ_prev) <= tol
            if verbose
            println("✅ Converged at iteration $ℓ")
            println("   Final bound: τ = $(round(τ_current, digits=4))")
            end
            break
        end
        τ_prev = τ_current
    end

    # Save tau history with dimensions in filename to prevent overwriting
    df_tau = DataFrame(iteration=[t[1] for t in tau_history], tau=[t[2] for t in tau_history])
    CSV.write("tau_history_$(method)_nx$(n_x_dim)_nu$(n_u_dim)_s$(s)_seed$(seed).csv", df_tau)

    return x_current, τ_prev, S, total_algorithm_time
end
# ============================================================
# --- 5. Main Execution Loop ---
# ============================================================
n_x_list = [1000, 2000]
n_u_list = [250, 500]
sparsities = [5, 10, 20]
SEEDS = [23, 24, 25, 26, 27]
methods = ["IHT","AGP","PSS"]
lambda = 0.5

all_results = []

println("\n🚀 STARTING EXPERIMENT: 12 Groups × 5 Seeds = 60 Instances per Method")

for n_x in n_x_list
    for n_u in n_u_list
        for seed in SEEDS
            mu, Sigma, B, Q = generate_portfolio_data(n_x, n_u; seed=seed)
            x_nominal, _, _ = solve_nominal_problem(mu, Sigma; lambda=lambda, optimizer=Mosek.Optimizer)
            nominal_return = mu' * x_nominal

            for method in methods
                for s in sparsities
                    println("▶️ Solving: nx=$n_x, nu=$n_u, s=$s, method=$method, seed=$seed")

                    x_opt, τ_opt, scenarios, algo_time = cutting_plane_algorithm(
                        mu, Sigma, B, Q, s;
                        lambda=lambda, method=method, max_iter=5000, tol=1e-5,
                        limit_time=1000.0,verbose=true, x_init=x_nominal, seed=seed,
                        n_x_dim=n_x, n_u_dim=n_u
                    )

                    push!(all_results, (
                        n_x = n_x, n_u = n_u, seed = seed, method = method,
                        sparsity = s, tau = τ_opt, scenarios = length(scenarios),
                        algo_time = algo_time, nominal_return = nominal_return
                    ))
                end
            end
        end
    end
end
println("🎉 All instances solved.")
# ============================================================
# --- 6. Post-Processing and Excel Export (SIMPLIFIED) ---
# ============================================================
df_final = DataFrame(all_results)
sort!(df_final, [:n_x, :n_u, :sparsity, :seed, :method])

# Get unique problems
unique_problems = unique(df_final[:, [:n_x, :n_u, :sparsity]])
sort!(unique_problems, [:n_x, :n_u, :sparsity])

problem_labels = ["#$(i) ($(row.n_x),$(row.n_u),$(row.sparsity))"
                  for (i, row) in enumerate(eachrow(unique_problems))]

# Pivot to wide format
methods_list = unique(df_final.method)
wide_df = DataFrame(Problem = problem_labels)

for m in methods_list
    m_data = filter(row -> row.method == m, df_final)
    m_data_unique = combine(groupby(m_data, [:n_x, :n_u, :sparsity]),
                            :tau => first => :tau,
                            :algo_time => first => :algo_time)
    sort!(m_data_unique, [:n_x, :n_u, :sparsity])

    wide_df[!, "CP-$(m)_UB"] = m_data_unique.tau
    wide_df[!, "CP-$(m)_Time"] = m_data_unique.algo_time
end

# Write to Excel (simplified - no merge)
# Write to Excel
excel_filename = joinpath(OUTPUT_DIR, "final_results_table.xlsx")
# excel_filename = "final_results_table.xlsx"
XLSX.openxlsx(excel_filename, mode="w") do xf
    sheet = xf[1]
    XLSX.rename!(sheet, "Results")

    # ✅ SIMPLIFIED HEADERS (no merging)
    sheet["A1"] = "Problem (n_x, n_u, s)"
    sheet["B1"] = "CP-AGP_UB"
    sheet["C1"] = "CP-AGP_Time"
    sheet["D1"] = "CP-IHT_UB"
    sheet["E1"] = "CP-IHT_Time"
    sheet["F1"] = "CP-PSS_UB"
    sheet["G1"] = "CP-PSS_Time"

    # Data rows (rows 2+)
    for (row_idx, row) in enumerate(eachrow(wide_df))
        sheet["A$(row_idx + 1)"] = row.Problem
        sheet["B$(row_idx + 1)"] = round(row["CP-AGP_UB"], digits=6)
        sheet["C$(row_idx + 1)"] = round(row["CP-AGP_Time"], digits=2)
        sheet["D$(row_idx + 1)"] = round(row["CP-IHT_UB"], digits=6)
        sheet["E$(row_idx + 1)"] = round(row["CP-IHT_Time"], digits=2)
        sheet["F$(row_idx + 1)"] = round(row["CP-PSS_UB"], digits=6)
        sheet["G$(row_idx + 1)"] = round(row["CP-PSS_Time"], digits=2)
    end
end

println("💾 Saved Excel: $excel_filename")

# Also save CSV (raw data)
# CSV.write("final_results_detailed.csv", df_final)
# Also save CSV raw data
CSV.write(joinpath(OUTPUT_DIR, "final_results_detailed.csv"), df_final)
println("💾 Saved CSV: final_results_detailed.csv")

# Summary statistics
println("\n📊 SUMMARY STATISTICS")
summary = combine(groupby(df_final, :method),
                  :tau => mean => :mean_UB,
                  :tau => std => :std_UB,
                  :algo_time => mean => :mean_time,
                  :algo_time => maximum => :max_time)
println(summary)
CSV.write(joinpath(OUTPUT_DIR, "summary_statistics.csv"), summary)
# CSV.write("summary_statistics.csv", summary)
println("💾 Saved summary: summary_statistics.csv")

println("\n✅ Post-processing complete!")