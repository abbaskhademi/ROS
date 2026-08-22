# Load required packages
required_packages = ["LinearAlgebra","Statistics","Random", "Printf","Arpack","SparseArrays"]
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




#                               Main Algorithems functions-------------------
using Printf
using LinearAlgebra
using Random
#-------------------------IHT--------------------------------------------------#
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
        # Existing stopping criterion
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
#############
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

        # Logging every 100 iterations
        if k % 100 == 0
            @printf("adaptive_IHT: iter = %5d, f(x) = %5.4f\n", k, fx)
        end

        # Early stopping based on iterate change
        if norm(x_curr - x_prev) <= epsilon
            @printf("🛑 Adaptive IHT stopped early: ||x_%d - x_%d|| = %.2e <= %.2e\n",
                    k, k-1, norm(x_curr - x_prev), epsilon)
            iterations = k
            @printf("adaptive_IHT: iter = %5d, f(x) = %5.4f\n", k, fx)
            break
        end
    end

    elapsed_time = time() - start_time

  #  println("Adaptive IHT completed:")
  #  println("  Final objective: $(round(fx, digits=8))")
  #  println("  Total iterations: $iterations")
  #  println("  Total backtracking steps: $(sum(backtrack_counts))")
  #  println("  Final gamma: $(round(gamma, digits=4))")
    println("  Elapsed time: $(round(elapsed_time, digits=4)) seconds")

    return x_curr, fx, elapsed_time, iterations, fx_values, L_ks, backtrack_counts, gamma_history
end
################################################
#---------------------------------greedy_sparse_simplex-------------------------
function greedy_sparse_simplex(f, g, s, N, x0; epsilon=1e-5)
    # Greedy Sparse Simplex Method with early stopping
    #
    # Solves: min f(x)  s.t.  ||x||_0 <= s
    #
    # INPUT:
    # f ......... objective function f(x)
    # g ......... 1D optimization oracle: g(x, S) returns (val, fval, ind)
    # s ......... sparsity level
    # N ......... max iterations
    # x0 ........ initial vector
    # epsilon ... stopping tolerance: stop if ||x_k - x_{k+1}|| <= epsilon
    #
    # OUTPUT:
    # X ......... list of iterates (each as vector)
    # fun_val ... final objective value
    # elapsed_time ... in seconds

    start_time = time()
    n = length(x0)
    x = copy(x0)
    X = Vector{Vector{Float64}}()  # Store iterates

    fold = Inf
    fun_val = f(x)
    iter_stuck = 0  # For objective stagnation

    for iter in 1:N

        d = count(!iszero, x)  # Current sparsity

        # Enforce sparsity: keep only s largest in magnitude
        if d > s
            perm = sortperm(abs.(x), by=abs)
            x[perm[(s+1):n]] .= 0
            d = s
        end

        # Store current iterate
        push!(X, copy(x))

        # Initialize update flag
        ok = 0

        if d < s
            # Under-sparsity: add one variable
            val, fval, ind = g(x, 1:n)
            x[ind] = val
            fun_val = fval
            ok = 1
        end

        if d == s && s > 0
            # Full support: try swap: remove one, add one
            I1 = findall(!iszero, x)
            min_funval = Inf
            min_index_out = 0
            min_index_in = 0
            min_val = 0.0

            for i in eachindex(I1)
                idx_out = I1[i]
                xtilde = copy(x)
                xtilde[idx_out] = 0
                val, fval, idx_in = g(xtilde, 1:n)

                if fval < min_funval
                    min_funval = fval
                    min_index_out = idx_out
                    min_index_in = idx_in
                    min_val = val
                end
            end

            # Perform swap
            x_old = copy(x)
            x[min_index_out] = 0
            x[min_index_in] = min_val
            fun_val = min_funval
            ok = iszero(x[min_index_in]) ? 1 : 0  # support changed?

            # ✅ Check stopping criterion: ||x_k - x_{k+1}|| <= epsilon
            if norm(x - x_old) <= epsilon
                @printf("🛑 Stopped greedy_sparse_simplex: ||x_%d - x_%d|| = %.2e <= %.2e\n",
                        iter, iter-1, norm(x - x_old), epsilon)
                @printf("greedy_sparse_simplex: iter =%3d fun_val = %5.5f \n", iter,f(x))
                break
            end
        end

        # Recompute objective to avoid drift
        fun_val = f(x)

        # Print progress
        if iter % 100 == 0
          @printf("greedy_sparse_simplex: iter =%3d fun_val = %5.4f change = %d\n", iter, fun_val, ok)
        end
    end

    fun_val = f(x)
    elapsed_time = time() - start_time

    println("  Elapsed time: $(round(elapsed_time, digits=4)) seconds")
    return X, fun_val, elapsed_time
end


#####################################################3
function g_LI(A, b, x, S)
    # Greedy update oracle for f(x) = ||Ax - b||^2
    #
    # Finds the index i ∈ S that, when optimally updated (with others fixed),
    # causes the greatest decrease in ||Ax - b||^2.
    #
    # Returns: (val, fun_val, i)
    #   val     = new optimal value of x[i]
    #   fun_val = new objective value f(x) after update
    #   i       = index in S chosen for update
    #
    # INPUT:
    # A ....... m×n matrix
    # b ....... m×1 vector
    # x ....... current n×1 vector
    # S ....... set of candidate indices (1D array or vector)

    # Gradient: ∇f(x) = 2*A'*(A*x - b), but factor of 2 cancels in update
    # We can work with g = A'*(A*x - b)
    r = A * x - b
    g = A' * r  # gradient (without factor 2)

    # Squared column norms: ||A[:,i]||^2 for each i
    norm_square_vector = vec(sum(A .^ 2, dims=1))  # size (n,)

    # Restrict to indices in S
    g_S = g[S]
    norm_sq_S = norm_square_vector[S]

    # Compute reduction in objective: g_i^2 / ||A_i||^2 (up to factor)
    # This is proportional to the decrease in f(x) when updating x[i]
    val_all = (g_S .^ 2) ./ norm_sq_S  # decrease potential for each i in S

    # Find index in S with maximum decrease
    _, imax = findmax(val_all)
    i_S = S[imax]  # actual index in 1:n

    # Optimal update: x[i] ← x[i] - g[i] / ||A[:,i]||^2
    delta = g[i_S] / norm_square_vector[i_S]
    x_new_i = x[i_S] - delta

    # New objective value: ||Ax - b - A[:,i]*delta||^2
    # = ||r - A[:,i]*delta||^2 = r'r - 2*delta*r'A[:,i] + delta^2 * ||A[:,i]||^2
    # But r'A[:,i] = g[i], so:
    # fun_val = norm(r)^2 - 2*delta*g[i] + delta^2 * norm_square_vector[i]
    r_norm_sq = norm(r)^2
    fun_val = r_norm_sq - 2 * delta * g[i_S] + delta^2 * norm_square_vector[i_S]

    # Alternatively: fun_val = norm(r - A[:, i_S] * delta)^2
    # But we avoid forming full residual again for efficiency

    return x_new_i, fun_val, i_S
end


################################################################
#--------------------partial_sparse_simplex----------------------------------
function partial_sparse_simplex(f, f_grad, g, s, N, x0; epsilon=1e-5)
    # Partial Sparse Simplex Method
    #
    # Solves: min f(x)  s.t.  ||x||_0 <= s
    #
    # Uses:
    # - f: objective
    # - f_grad: gradient oracle
    # - g(x, S): 1D optimizer: returns (val, fval, ind) for best i ∈ S
    #
    # At each step:
    #   if nnz < s: add best variable in 1:n
    #   if nnz == s: try swapping smallest |x_i| with best zero (via gradient or g)
    #   if nnz > s: randomly zero out excess
    #
    # Returns: X (list of iterates), fun_val (final f(x)), elapsed_time

    start_time = time()
    n = length(x0)
    x = copy(x0)
    X = Vector{Vector{Float64}}()  # Store iterates

    fold = Inf
    fun_val = f(x)
    iter_stuck = 0

    for iter in 1:N
        d = count(!iszero, x)  # nnz(x)
        # If over-sparsified: randomly zero out excess (but we already have d > s?)
        if d > s
            P = randperm(n)
            # Zero out n-s entries (but only among nonzeros? or all?)
            # Here: zero out n-s entries randomly in full vector
            to_zero = P[1:(n-s)]
            x[to_zero] .= 0
            d = count(!iszero, x)  # recompute
        end

        # Store current iterate
        push!(X, copy(x))

        ok = 0  # flag: 1 if support changed

        if d < s
            # Add one variable: use g(x, 1:n)
            _, fval, ind = g(x, 1:n)
            x[ind] = g(x, 1:n)[1]  # set to optimal value
            fun_val = fval
            ok = 0  # no swap, just add
        end

        if d == s
            # Current support
            I1 = findall(!iszero, x)
            I0 = setdiff(1:n, I1)

            # Default: update smallest nonzero
            _, _, min_index = g(x, I1)
            x_min_update = g(x, I1)[1]

            # Find smallest nonzero in magnitude
            _, ind_min = findmin(abs.(x[I1]))
            i_index = I1[ind_min]  # index to potentially remove

            # Find best zero entry via gradient (largest |grad|)
            grad = f_grad(x)
            if length(I0) > 0
                _, ind_max = findmax(abs.(grad[I0]))
                j_index = I0[ind_max]  # candidate to add
            else
                j_index = 0
            end

            if j_index != 0
                # Try updating x[j_index] optimally (with others fixed)
                xtilde = copy(x)
                xtilde[i_index] = 0  # temporarily remove i_index
                val, fun_val_candidate, _ = g(xtilde, [j_index])

                # Compare: is swapping better?
                if fun_val_candidate < fun_val + 1e-8
                    x[i_index] = 0
                    x[j_index] = val
                    ok = 1  # support changed
                else
                    # Just update the smallest nonzero
                    x[min_index] = x_min_update
                end
            else
                # No zero entries — just update smallest nonzero
                x[min_index] = x_min_update
            end
        end

        # Recompute objective
        fun_val = f(x)

        # Print progress
        if iter % 100 == 0
          @printf("partial_sparse_simplex iter =%3d fun_val = %5.4f change = %d\n", iter, fun_val, ok)
        end

        # ✅ Optional: stopping criterion ||x_k - x_{k+1}|| <= epsilon
        if iter > 1
            x_prev = X[end]
            if norm(x - x_prev) <= epsilon
                @printf("🛑 Stopped partial_sparse_simplex: ||x_%d - x_%d|| = %.2e <= %.2e\n",
                        iter, iter-1, norm(x - x_prev), epsilon)
                @printf("partial_sparse_simplex iter =%3d fun_val = %5.4f \n", iter, fun_val)
                break
            end
        end
    end

    fun_val = f(x)
    elapsed_time = time() - start_time
     total_iterations = length(X) - 1 # Calculates the number of updates

     println("  Elapsed time: $(round(elapsed_time, digits=4)) seconds")

    return X, fun_val, elapsed_time,total_iterations
end
###########################################




using Printf                              # this
using LinearAlgebra
using Random
using DataFrames
using CSV
using Statistics

# ------------------- ALGORITHM FUNCTIONS -------------------
# Assume IHT, adaptive_IHT, partial_sparse_simplex, g_LI are defined elsewhere

# ------------------- PROBLEM GENERATION -------------------
function generate_sparse_ls_instance(m, n, s; seed=nothing, cond_num=50, noise_level=0.0)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    p = min(m, n)
    U = Matrix(qr(randn(m, p)).Q)
    V = Matrix(qr(randn(n, p)).Q)

    if p == 1
        σ = [1.0]
    else
        σ = [cond_num^(-(i-1)/(p-1)) for i in 1:p]
    end

    A = U * Diagonal(σ) * V'
    A ./= sqrt.(sum(A .^ 2, dims=1))  # column-normalized

    x_true = zeros(n)
    support = randperm(n)[1:s]
    x_true[support] = randn(s)

    b = A * x_true + noise_level * randn(m)
    x0 = rand(n)
    L = 2 * (opnorm(A)^2)
    return A, b, x_true, x0, L
end

# ------------------- OBJECTIVE & GRADIENT -------------------
f(x, A, b) = norm(A * x - b)^2
g(x, A, b) = 2 * A' * (A * x - b)

# ------------------- EXPERIMENT SETUP -------------------
groups = [
    (m=100, n=30000, s=5),
    (m=100, n=30000, s=10),
    (m=100, n=30000, s=20),
    (m=100, n=50000, s=5),
    (m=100, n=50000, s=10),
    (m=100, n=50000, s=20),
    (m=200, n=30000, s=5),
    (m=200, n=30000, s=10),
    (m=200, n=30000, s=20),
    (m=200, n=50000, s=5),
    (m=200, n=50000, s=10),
    (m=200, n=50000, s=20),

]

seeds = [23, 24, 25, 26, 27]
N = 3000

# Initialize results as a vector of NamedTuples (easier for SGap later)
all_results = NamedTuple[]

println("\n" * "^"^70)
println("🚀 STARTING Sparse Signal Recovery BENCHMARK (12 GROUPS × 5 instances)")
println("^"^70)

for (gid, (m, n, s)) in enumerate(groups)
    println("\n" * "#" * "="^68 * "#")
    println("### 🌐 GROUP $gid: (m=$m, n=$n, s=$s) ###")
    println("#" * "="^68 * "#")

    for seed in seeds
        println("\n🌱 Seed = $seed")
        A, b, x_true, x0, L = generate_sparse_ls_instance(m, n, s; seed=seed)

        # --- IHT ---
        println("-------------------- Method = IHT----------------------------")
        try
            x, fx, time, iters = IHT(x -> f(x, A, b), x -> g(x, A, b), s, L, x0, N, epsilon=1e-5)
            push!(all_results, (group_id=gid, m=m, n=n, s=s, seed=seed, method="IHT", UB=fx, Time=time))
            println("✅ IHT:                     UB = $(round(fx, digits=4)), Time = $(round(time, digits=2))s")
        catch e
            @warn "❌ IHT failed: $e"
            push!(all_results, (group_id=gid, m=m, n=n, s=s, seed=seed, method="IHT", UB=Inf, Time=0.0))
        end

        # --- Adaptive IHT ---
        println("-----------------------Method = Adaptive_IHT-----------------")
        try
            x, fx, time, iters, _, _, backtrack_counts, _ =
                adaptive_IHT(x -> f(x, A, b), x -> g(x, A, b), s, x0, N, epsilon=1e-5)
            push!(all_results, (group_id=gid, m=m, n=n, s=s, seed=seed, method="Adaptive_IHT", UB=fx, Time=time))
            println("✅ Adaptive_IHT:            UB = $(round(fx, digits=4)), Time = $(round(time, digits=2))s")
        catch e
            @warn "❌ Adaptive_IHT failed: $e"
            push!(all_results, (group_id=gid, m=m, n=n, s=s, seed=seed, method="Adaptive_IHT", UB=Inf, Time=0.0))
        end

        # --- Partial Sparse Simplex ---
        println("-------------------  Method = Partial_Sparse_Simplex-------------")
        try
            X, fx, time = partial_sparse_simplex(
                x -> f(x, A, b), x -> g(x, A, b), (x, S) -> g_LI(A, b, x, S), s, N, x0, epsilon=1e-5)
            push!(all_results, (group_id=gid, m=m, n=n, s=s, seed=seed, method="Partial_Sparse_Simplex", UB=fx, Time=time))
            println("✅ Partial_Sparse_Simplex:  UB = $(round(fx, digits=4)), Time = $(round(time, digits=2))s")
        catch e
            @warn "❌ Partial_Sparse_Simplex failed: $e"
            push!(all_results, (group_id=gid, m=m, n=n, s=s, seed=seed, method="Partial_Sparse_Simplex", UB=Inf, Time=0.0))
        end
    end
end
################################################################################



# ------------------- SAVE RESULTS -------------------
CSV.write("sparse_ls_results_all.csv", all_results)
println("\n💾 Saved all results to: sparse_ls_results_all.csv")

# Load data
df = CSV.read("sparse_ls_results_all.csv", DataFrame)

# Rename for clarity
rename!(df, :UB => :UB, :Time => Symbol("Time (s)"))

# Sort
sort!(df, [:m, :n, :s, :seed, :method])

# --- STEP 1: Compute SGap within each (m, n, s, seed) group ---
gdf = groupby(df, [:m, :n, :s, :seed])

# Best UB per instance (i.e., per seed in a fixed problem)
transform!(gdf, :UB => (x -> minimum(x)) => :UB_best)

# Compute SGap (%)
df[!, Symbol("SGap (%)")] = (df.UB .- df.UB_best) ./ (abs.(df.UB) .+ 1e-4) .* 100

# Remove helper
select!(df, Not(:UB_best))

# Save detailed
CSV.write("sparse_detailed_with_correct_sgap.csv", df)

# --- STEP 2: Summary over seeds, grouped by (m, n, s, method) ---
summary_df = combine(
    groupby(df, [:m, :n, :s, :method]),
    Symbol("Time (s)") => (x -> round(mean(x), digits=2)) => Symbol("Avg Time"),
    Symbol("Time (s)") => (x -> round(std(x), digits=2)) => Symbol("Std Time"),
    Symbol("SGap (%)") => (x -> round(mean(x), digits=2)) => Symbol("Avg SGap"),
    Symbol("SGap (%)") => (x -> round(std(x), digits=2)) => Symbol("Std SGap")
)

sort!(summary_df, [:m, :n, :s, :method])

# Print nicely
println("m\tn\ts\tmethod\t\t\tAvg Time\tStd Time\tAvg SGap\tStd SGap")
for row in eachrow(summary_df)
    @printf("%d\t%d\t%d\t%-25s\t%.2f\t\t%.2f\t\t%.2f\t\t%.2f\n",
        row.m, row.n, row.s, row.method,
        row[Symbol("Avg Time")], row[Symbol("Std Time")],
        row[Symbol("Avg SGap")], row[Symbol("Std SGap")]
    )
end

CSV.write("sparse_summary_correct_sgap.csv", summary_df)
println("\n✅ Summary saved to: sparse_summary_correct_sgap.csv")
