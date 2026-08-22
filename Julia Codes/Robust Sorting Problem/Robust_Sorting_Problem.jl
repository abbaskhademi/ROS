# Load required packages
required_packages = ["JuMP", "MosekTools", "CSV", "DataFrames", "Distributions", "LinearAlgebra", "Printf", "Random","HTTP","PrettyTables","DataStructures","StatsBase"]
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
#----------------------------------------
# main code
using JuMP
using MosekTools
using Printf
using Random
using LinearAlgebra
using Statistics

# =============================================================================
# ALGORITHM A (Bertsimas & Sim 2003) — Exact Reference
#
# Uncertainty: ũ_i ∈ [c_i, d_i], deviation ĥ_i = d_i - c_i ≥ 0
#
# Z*(Γ) = min_{l=1,...,n+1} G_l
# G_l   = Γ·ĥ_l + min_{x∈X} { c'x + Σ_{j=1}^{l} (ĥ_j - ĥ_l) x_j }
# ĥ_1 ≥ ĥ_2 ≥ ... ≥ ĥ_n ≥ ĥ_{n+1} = 0
# =============================================================================
function robust_sorting(c::Vector{Float64}, d::Vector{Float64}, k::Int, Γ::Number)
    n          = length(c)
    dev        = d .- c
    perm       = sortperm(dev, rev=true)
    dev_sorted = dev[perm]
    c_perm     = c[perm]
    push!(dev_sorted, 0.0)

    best_obj = Inf
    best_x   = zeros(n)

    for l in 1:(n + 1)
        dl = dev_sorted[l]
        mc = copy(c_perm)
        for j in 1:min(l, n)
            mc[j] += (dev_sorted[j] - dl)
        end

        model = Model(Mosek.Optimizer); set_silent(model)
        @variable(model, x[1:n], Bin)
        @constraint(model, sum(x) == k)
        @objective(model, Min, dot(mc, x))
        optimize!(model)

        if termination_status(model) == MOI.OPTIMAL
            G_l = Γ * dl + objective_value(model)
            if G_l < best_obj
                best_obj = G_l
                xv       = value.(x)
                best_x   = zeros(n)
                for i in 1:n; best_x[perm[i]] = xv[i]; end
            end
        end
    end
    return best_obj, best_x
end


# =============================================================================
# PROBLEM (30) — Direct MIP Reformulation (Bertsimas & Sim, Theorem 1)
#
# For pure cost uncertainty ũ_j ∈ [c_j, c_j + ĥ_j] with ĥ_j = d_j - c_j,
# the robust counterpart of the sorting problem is (eq. 3 / eq. 30 of paper):
#
#   min   c'x + Γ·z + Σ_j p_j
#   s.t.  z + p_j ≥ ĥ_j · x_j,   ∀j          (dual of inner max)
#         Σ_i x_i = k
#         p_j ≥ 0,   z ≥ 0
#         x ∈ {0,1}^n
#
# This is an exact MIP of size O(n) — no decomposition, no cutting plane.
# It solves the robust problem in a SINGLE MIP solve.
# =============================================================================
function robust_sorting_MIP30(c::Vector{Float64}, d::Vector{Float64},
                               k::Int, Γ::Number)
    n   = length(c)
    h   = d .- c                    # ĥ_j = d_j - c_j ≥ 0

    model = Model(Mosek.Optimizer); set_silent(model)

    @variable(model, x[1:n], Bin)
    @variable(model, z >= 0)
    @variable(model, p[1:n] >= 0)

    @constraint(model, sum(x) == k)

    # Dual constraints: z + p_j ≥ ĥ_j · x_j  ∀j
    for j in 1:n
        @constraint(model, z + p[j] >= h[j] * x[j])
    end

    # Objective: c'x + Γ·z + Σ_j p_j  (= nominal cost + robust penalty)
    @objective(model, Min, dot(c, x) + Γ * z + sum(p))

    optimize!(model)

    status = termination_status(model)
    status ∉ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) && error("MIP30 failed: $status")

    return objective_value(model), value.(x)
end


# =============================================================================
# HELPERS
# =============================================================================
function compute_worst_case_cost(x_sol::Vector{Float64},
                                 c::Vector{Float64},
                                 h::Vector{Float64},
                                 Γ::Number)
    active = h .* x_sol
    sidx   = sortperm(active, rev=true)
    wc_dev = sum(active[sidx[t]] for t in 1:min(Int(Γ), length(sidx)))
    return dot(c, x_sol) + wc_dev
end

function project_sparsity(ũ::Vector{Float64}, Γ::Number)
    sidx = sortperm(abs.(ũ), rev=true)
    u    = zeros(length(ũ))
    for t in 1:min(Int(Γ), length(ũ)); u[sidx[t]] = ũ[sidx[t]]; end
    return u
end

function solve_master(c::Vector{Float64}, k::Int, S::Vector{Vector{Float64}})
    n_x   = length(c)
    model = Model(Mosek.Optimizer); set_silent(model)
    @variable(model, 0 <= x[1:n_x] <= 1)
    @variable(model, τ)
    @constraint(model, sum(x) == k)
    for s in S
        @constraint(model, dot(c .+ s, x) <= τ)
    end
    @objective(model, Min, τ)
    optimize!(model)
    status = termination_status(model)
    status ∉ (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL) && error("Master failed: $status")
    return value.(x), value(τ)
end


# =============================================================================
# CP-SP: Cutting Plane with Scenario Projection (LP relaxation)
# =============================================================================
function CP_SP(c::Vector{Float64},
               d::Vector{Float64},
               k::Int,
               Γ::Number;
               ε::Float64    = 1e-6,
               max_iter::Int = 500,
               verbose::Bool = false)

    n_x = length(c)
    h   = d .- c

    m0 = Model(Mosek.Optimizer); set_silent(m0)
    @variable(m0, 0 <= x0[1:n_x] <= 1)
    @constraint(m0, sum(x0) == k)
    @objective(m0, Min, dot(c, x0))
    optimize!(m0)
    x_cur = value.(x0)
    τ_cur = dot(c, x_cur)

    S = Vector{Vector{Float64}}()

    for ℓ in 1:max_iter
        ũ  = h .* x_cur
        u  = project_sparsity(ũ, Γ)
        wc = dot(c .+ u, x_cur)

        verbose && @printf("  [SP] iter %3d | τ=%10.4f | WC=%10.4f | gap=%10.6f | |S|=%d\n",
                           ℓ, τ_cur, wc, wc - τ_cur, length(S))

        if wc - τ_cur <= ε
            verbose && println("  [SP] ✓ Converged at iter $ℓ")
            break
        end

        any(norm(s - u, Inf) < 1e-10 for s in S) && begin
            verbose && println("  [SP] ⚠ Duplicate → stop")
            break
        end
        push!(S, copy(u))

        x_cur, τ_cur = solve_master(c, k, S)
        ℓ == max_iter && @warn "[SP] Max iterations reached"
    end

    final_wc = compute_worst_case_cost(x_cur, c, h, Γ)
    return τ_cur, x_cur, length(S)
end
#--------------------------------
################################################## run
function compare_methods()
    println("=" ^ 100)
    println("COMPARISON: Algorithm A  vs  CP-SP (LP relaxation)")
    println("  n ∈ {500,1000,2000}, k=n/2,  Γ ∈ {10,20,30,40},  seeds ∈ {23,24,25,26,27}")
    println("=" ^ 100)

    n_x_values = [500, 1000, 2000]
    Γ_values   = [10, 20, 30, 40]
    seeds      = [23, 24, 25, 26, 27]

    gap_pct(objA, lb) = abs(objA - lb) / (abs(objA) + 1e-4) * 100

    all_results = Dict{Int, Dict{Int, Vector{NamedTuple}}}()

    for n_x in n_x_values
        k = div(n_x, 2)
        all_results[n_x] = Dict{Int, Vector{NamedTuple}}()

        println("\n", "#" ^ 100)
        @printf("  n_x = %d  |  k = %d\n", n_x, k)
        println("#" ^ 100)

        for seed in seeds
            all_results[n_x][seed] = NamedTuple[]

            Random.seed!(seed)
            c = rand(50.0:200.0, n_x)
            d = c .+ rand(10.0:100.0, n_x)

            println("\n", "=" ^ 100)
            @printf("  n_x = %d  |  SEED = %d  |  c ∈ [%.1f, %.1f]  |  ĥ ∈ [%.1f, %.1f]\n",
                    n_x, seed, minimum(c), maximum(c),
                    minimum(d .- c), maximum(d .- c))
            println("=" ^ 100)

            for Γ in Γ_values
                println("\n  ▶ Γ = $Γ")

                tA  = @elapsed valA,  xA       = robust_sorting(c, d, k, Γ)
                tSP = @elapsed valSP, xSP, nSP = CP_SP(c, d, k, Γ)

                g = gap_pct(valA, valSP)

                @printf("    Alg.A    : obj=%12.4f  time=%8.4fs\n", valA, tA)
                @printf("    CP-SP    : obj=%12.4f  time=%8.4fs  cuts=%d  Gap(%%)=%8.4f\n",
                        valSP, tSP, nSP, g)

                push!(all_results[n_x][seed],
                      (seed=seed, n_x=n_x, k=k, Γ=Γ,
                       valA=valA, tA=tA,
                       valSP=valSP, tSP=tSP,
                       nSP=nSP,    gap=g))
            end
        end
    end

    # =========================================================================
    # FINAL SUMMARY TABLE  (one block per n_x)
    # =========================================================================
    for n_x in n_x_values
        k = div(n_x, 2)

        println("\n\n", "=" ^ 100)
        @printf("FINAL SUMMARY TABLE  |  n_x = %d  |  k = %d\n", n_x, k)
        println("=" ^ 100)

        @printf("%-4s | ", "Γ")
        for seed in seeds
            @printf("%-44s | ", "Seed $seed")
        end
        println()

        @printf("%-4s | ", "")
        for _ in seeds
            @printf("%-12s %-7s | %-12s %-7s %-9s | ",
                    "A-obj",  "A-t",
                    "SP-obj", "SP-t", "Gap(%)")
        end
        println()
        println("-" ^ (6 + length(seeds) * 47))

        for Γ in Γ_values
            @printf("%-4d | ", Γ)
            for seed in seeds
                r = all_results[n_x][seed][findfirst(x -> x.Γ == Γ, all_results[n_x][seed])]
                @printf("%12.4f %7.3f | %12.4f %7.3f %9.4f | ",
                        r.valA,  r.tA,
                        r.valSP, r.tSP,
                        r.gap)
            end
            println()
        end
    end

    # =========================================================================
    # AGGREGATED SUMMARY: mean ± std across 5 seeds  (one block per n_x)
    # =========================================================================
    for n_x in n_x_values
        k = div(n_x, 2)

        println("\n\n", "=" ^ 100)
        @printf("AGGREGATED SUMMARY  (mean ± std over 5 seeds)  |  n_x = %d  |  k = %d\n", n_x, k)
        println("=" ^ 100)
        @printf("%-4s | %-26s | %-26s | %-20s\n",
                "Γ", "Alg. A", "CP-SP (LP)", "Gap(%)")
        @printf("%-4s | %-26s | %-26s | %-20s\n",
                "",
                "mean obj    mean t(s)",
                "mean obj    mean t(s)",
                "mean        std")
        println("-" ^ 82)

        for Γ in Γ_values
            rows = [all_results[n_x][s][findfirst(x -> x.Γ == Γ, all_results[n_x][s])]
                    for s in seeds]

            A_obj  = [r.valA  for r in rows]
            A_t    = [r.tA    for r in rows]
            SP_obj = [r.valSP for r in rows]
            SP_t   = [r.tSP   for r in rows]
            gaps   = [r.gap   for r in rows]

            @printf("%-4d | %12.4f  %8.4f | %12.4f  %8.4f | %8.4f  %8.4f\n",
                    Γ,
                    mean(A_obj), mean(A_t),
                    mean(SP_obj), mean(SP_t),
                    mean(gaps),  std(gaps))
        end
    end

    # =========================================================================
    # CONSOLIDATED INSTANCE TABLE
    # One row per (n_x, seed, Γ) combination — 3 n_x × 4 Γ × 5 seeds = 60 rows
    # Ordering: outer = n_x, middle = Γ, inner = seed
    #
    #   #1  – #5  : (500,  250, 10), seeds 23–27
    #   #6  – #10 : (500,  250, 20), seeds 23–27
    #   ...
    #   #56 – #60 : (2000,1000, 40), seeds 23–27
    # =========================================================================
    println("\n\n", "=" ^ 100)
    println("CONSOLIDATED INSTANCE TABLE  (60 instances: 3×n  ×  4×Γ  ×  5×seeds)")
    println("=" ^ 100)

    # Header line 1 — method names
    @printf("%-5s  %-22s | %-26s | %-26s | %-10s\n",
            "",
            "Problem",
            "Alg. A",
            "CP-SP (LP)",
            "Gap(%)")

    # Header line 2 — column labels
    @printf("%-5s  %-6s %-5s %-5s %-5s | %-12s %-8s | %-12s %-8s %-5s | %-10s\n",
            "#",
            "n", "k", "Γ", "seed",
            "obj", "time(s)",
            "obj", "time(s)", "cuts",
            "Gap(%)")
    println("-" ^ 95)

    inst = 0
    for n_x in n_x_values
        k = div(n_x, 2)
        for Γ in Γ_values
            for seed in seeds
                inst += 1
                r = all_results[n_x][seed][findfirst(x -> x.Γ == Γ, all_results[n_x][seed])]

                @printf("%-5d  %-6d %-5d %-5d %-5d | %12.4f %8.4f | %12.4f %8.4f %-5d | %10.4f\n",
                        inst,
                        n_x, k, Γ, seed,
                        r.valA,  r.tA,
                        r.valSP, r.tSP, r.nSP,
                        r.gap)
            end
            println("-" ^ 95)   # separator between Γ blocks
        end
        println("=" ^ 95)       # separator between n_x blocks
    end

    return all_results
end

println("Running comparison...")
results = compare_methods()
#----------------------------------
# Run insyances

# =============================================================================
# COMPARISON
# =============================================================================
function compare_methods()
    println("=" ^ 100)
    println("COMPARISON: Algorithm A  vs  CP-SP (LP relaxation)")
    println("  n ∈ {500,1000,2000}, k=n/2,  Γ ∈ {10,20,30,40},  seeds ∈ {23,24,25,26,27}")
    println("=" ^ 100)

    n_x_values = [500, 1000, 2000]
    Γ_values   = [10, 20, 30, 40]
    seeds      = [23, 24, 25, 26, 27]

    # Gap(%): integrality gap between MIP (Alg.A) and LP relaxation (CP-SP)
    # Since Z*_LP ≤ Z*_MIP, gap ≥ 0 always
    gap_pct(objA, lb) = (objA - lb) / (abs(objA) + 1e-4) * 100

    all_results = Dict{Int, Dict{Int, Vector{NamedTuple}}}()

    for n_x in n_x_values
        k = div(n_x, 2)
        all_results[n_x] = Dict{Int, Vector{NamedTuple}}()

        println("\n", "#" ^ 100)
        @printf("  n = %d  |  k = %d\n", n_x, k)
        println("#" ^ 100)

        for seed in seeds
            all_results[n_x][seed] = NamedTuple[]

            Random.seed!(seed)
            c = rand(50.0:200.0, n_x)
            d = c .+ rand(10.0:100.0, n_x)

            println("\n", "=" ^ 100)
            @printf("  n=%d  |  SEED=%d  |  c ∈ [%.1f,%.1f]  |  ĥ ∈ [%.1f,%.1f]\n",
                    n_x, seed, minimum(c), maximum(c),
                    minimum(d .- c), maximum(d .- c))
            println("=" ^ 100)

            for Γ in Γ_values
                println("\n  ▶ Γ = $Γ")

                tA  = @elapsed valA,  xA       = robust_sorting(c, d, k, Γ)
                tSP = @elapsed valSP, xSP, nSP = CP_SP(c, d, k, Γ)

                g = gap_pct(valA, valSP)

                @printf("    Alg.A : obj=%12.4f  time=%8.4fs\n", valA, tA)
                @printf("    CP-SP : obj=%12.4f  time=%8.4fs  cuts=%d  Gap(%%)=%8.4f\n",
                        valSP, tSP, nSP, g)

                push!(all_results[n_x][seed],
                      (seed=seed, n_x=n_x, k=k, Γ=Γ,
                       valA=valA,  tA=tA,
                       valSP=valSP, tSP=tSP,
                       nSP=nSP,    gap=g))
            end
        end
    end

    # =========================================================================
    # AGGREGATED SUMMARY per (n, Γ) group — mean ± std over 5 seeds
    # =========================================================================
    println("\n\n", "=" ^ 100)
    println("AGGREGATED SUMMARY  (mean ± std over 5 seeds)")
    println("=" ^ 100)
    @printf("%-4s  %-22s | %-26s | %-26s | %-20s\n",
            "#", "Group (n,k,Γ)", "Alg. A", "CP-SP (LP)", "Gap(%)")
    @printf("%-4s  %-22s | %-26s | %-26s | %-20s\n",
            "", "",
            "mean obj     mean t(s)",
            "mean obj     mean t(s)",
            "mean         std")
    println("-" ^ 105)

    group_idx = 0
    group_stats = NamedTuple[]     # store for LaTeX table

    for n_x in n_x_values
        k = div(n_x, 2)
        for Γ in Γ_values
            group_idx += 1
            rows = [all_results[n_x][s][findfirst(x -> x.Γ == Γ, all_results[n_x][s])]
                    for s in seeds]

            A_t    = [r.tA    for r in rows]
            SP_t   = [r.tSP   for r in rows]
            A_obj  = [r.valA  for r in rows]
            SP_obj = [r.valSP for r in rows]
            gaps   = [r.gap   for r in rows]

            @printf("#%-3d (%4d,%4d,%2d) | %12.4f  %8.4f | %12.4f  %8.4f | %8.4f  %8.4f\n",
                    group_idx, n_x, k, Γ,
                    mean(A_obj),  mean(A_t),
                    mean(SP_obj), mean(SP_t),
                    mean(gaps),   std(gaps))

            push!(group_stats,
                  (idx=group_idx, n_x=n_x, k=k, Γ=Γ,
                   mA_t=mean(A_t),   sA_t=std(A_t),
                   mSP_t=mean(SP_t), sSP_t=std(SP_t),
                   mGap=mean(gaps),  sGap=std(gaps)))
        end
        println("-" ^ 105)
    end

    # =========================================================================
    # LaTeX TABLE
    # Format: mean [std] for time_A, time_SP, Gap(%)
    # =========================================================================
    println("\n\n", "=" ^ 100)
    println("LaTeX TABLE")
    println("=" ^ 100)

    println(raw"""
\begin{table}[htbp]
\centering
\caption{Summary statistics of solution gaps (\%) and computation times (seconds)
         for Robust Sorting Problem instances.
         Each entry reports mean [std] over 5 independent random instances.}
\label{tab:summary_robust_sorting}
\begin{tabular}{lcccc}
\toprule
Group $(n,k,\Gamma)$ & Time Alg.\ A & Time SP & Gap(\%) \\
\midrule""")

    prev_n = 0
    for r in group_stats
        # Insert \midrule between n blocks
        if r.n_x != prev_n && prev_n != 0
            println(raw"\midrule")
        end
        prev_n = r.n_x

        @printf("\\#%-2d (%4d,%4d,%2d) & %.2f [%.2f] & \\textbf{%.2f} [%.2f] & %.2f [%.2f] \\\\\n",
                r.idx, r.n_x, r.k, r.Γ,
                r.mA_t,  r.sA_t,
                r.mSP_t, r.sSP_t,
                r.mGap,  r.sGap)
    end

    println(raw"""\bottomrule
\end{tabular}
\end{table}""")

    return all_results, group_stats
end

println("Running comparison...")
results, stats = compare_methods()