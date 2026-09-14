#!/usr/bin/env julia

module WirelessExperimentDriverV11

ENV["GKSwstype"] = "100"

using CSV
using DataFrames
using Statistics
using Printf
using Plots

include(joinpath(@__DIR__, "WirelessSparseRO_v11.jl"))
include(joinpath(@__DIR__, "WirelessSparseARO_v11.jl"))

default(fmt = :pdf)

function main(args::Vector{String} = copy(ARGS))
quick = "--quick" in args
ro_only = "--ro-only" in args
aro_only = "--aro-only" in args
run_aadr_rl_el = "--with-aadr-rl-el" in args
run_aadr_rl = !("--skip-aadr-rl" in args)
run_ro_reference = !("--skip-ro-reference" in args)
resume = "--resume" in args
ro_only && aro_only && error("Choose at most one of --ro-only and --aro-only")

if "--config-check" in args
    WirelessSparseRO.ROConfig()
    WirelessSparseARO.AROConfig()
    println("Driver module and RO/ARO configuration bindings: PASS")
    return nothing
end

function argvalue(flag::String, default_value::String)
    idx = findfirst(==(flag), args)
    return idx === nothing ? default_value : args[idx + 1]
end

outdir = argvalue("--outdir", joinpath(@__DIR__, "..", "..", "reproduced", "wireless"))
mkpath(outdir)
seeds = quick ? [1] : collect(1:3)
ro_methods = String.(split(argvalue("--ro-methods", "SP,RL,EL"), ','))
aro_separator_methods = String.(split(argvalue("--aro-methods", "SP,RL,EL"), ','))

# -----------------------------------------------------------------------------
# Experiment ladders
# -----------------------------------------------------------------------------
# RO tuple: (measured antennas, potential jammers, users, s, pools)
ro_specs = quick ? [(47, 200, 20, 8, 67)] : [
    (47, 200, 20, 8, 67),
    (47, 250, 30, 12, 84),
    (47, 300, 40, 14, 100),
    (47, 300, 50, 14, 100),
]

# ARO tuple: (measured antennas, users, potential jammers, s, pools).
# The scalable suite uses the same measured channels and resource pools as the
# static experiment. The groups increase the user and profile counts. The small
# exact check uses subsets of the same measurements.
aro_specs = quick ? [(47, 15, 150, 7, 50)] : [
    (47, 25, 250, 12, 84),
    (47, 30, 300, 14, 100),
    (47, 35, 300, 14, 100),
]

# The loss scale is fixed across the measured static and adjustable cases.
WIRELESS_JAMMER_FRACTION = 0.75
WIRELESS_RESOURCE_CAPACITY = 3.20

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
function finitevals(values)
    return [Float64(v) for v in values if v isa Real && isfinite(v)]
end

function mean_sd(values)
    x = finitevals(values)
    if isempty(x)
        return NaN, NaN
    end
    return mean(x), length(x) > 1 ? std(x) : 0.0
end

function mean_or_nan(values)
    x = finitevals(values)
    return isempty(x) ? NaN : mean(x)
end

fmt2(x) = isfinite(x) ? @sprintf("%.2f", x) : "NA"
fmt4(x) = isfinite(x) ? @sprintf("%.4f", x) : "NA"
fmtsci(x) = isfinite(x) ? (abs(x) < 1e-5 ? "0" : @sprintf("%.2e", x)) : "NA"
latex_row_end = " \\\\"

function usable_aro_objective(method::String, status::String, objective)
    isfinite(objective) || return false
    if method == "AADR-RL/EL"
        return status == "CERTIFIED_AFFINE_POLICY"
    elseif method == "AADR-RL"
        return status == "CERTIFIED_AADR_RL_POLICY"
    elseif method == "RO-EL"
        return status == "CERTIFIED_ONE_STAGE"
    end
    # DB values remain valid lower bounds even when the local separator stops by
    # iteration/time limit, provided a finite master objective is returned.
    return startswith(method, "DB-")
end

function aro_role(method::String)
    startswith(method, "DB-") && return "PROPOSED_METHOD"
    method == "AADR-EXACT" && return "EXACT_TINY_COMPETITOR"
    method == "AADR-RL" && return "RESTRICTED_AFFINE_POLICY"
    method == "AADR-RL/EL" && return "COMPETITOR_POLICY_USING_DB_SEPARATION"
    method == "RO-EL" && return "RESTRICTED_POLICY_REFERENCE"
    method == "EXACT-FULL" && return "EXACT_TINY_REFERENCE"
    return "REFERENCE"
end

function runtime_use(method::String)
    startswith(method, "DB-") && return "PRIMARY_METHOD_COMPARISON"
    method == "AADR-EXACT" && return "TINY_EXACT_COMPETITOR_ONLY"
    method == "EXACT-FULL" && return "TINY_EXACT_REFERENCE_ONLY"
    method == "AADR-RL" && return "SUPPLEMENTARY_POLICY"
    return "NOT_AN_INDEPENDENT_RUNTIME_COMPETITOR"
end

function incomplete_status(status::String)
    return status in (
        "TIME_LIMIT", "ITERATION_LIMIT", "MAX_ITERATIONS", "STALLED",
        "FAILED_OR_TIMEOUT", "INFEASIBLE",
    )
end

function first_method_row(df::DataFrame, method::String)
    y = df[df.method .== method, :]
    return nrow(y) == 0 ? nothing : y[1, :]
end

function resume_source(checkpoint_name::String, final_name::String)
    checkpoint = joinpath(outdir, checkpoint_name)
    final = joinpath(outdir, final_name)
    return isfile(checkpoint) ? checkpoint : final
end

function load_resume_rows(
    checkpoint_name::String,
    final_name::String,
    key_columns::Vector{Symbol},
)
    resume || return NamedTuple[]
    source = resume_source(checkpoint_name, final_name)
    isfile(source) || return NamedTuple[]
    df = CSV.read(source, DataFrame)
    isempty(df) && return NamedTuple[]
    missing_columns = setdiff(key_columns, Symbol.(names(df)))
    isempty(missing_columns) || error(
        "Cannot resume from $source; missing key columns $(join(missing_columns, ", ")).",
    )
    before = nrow(df)
    df = unique(df, key_columns)
    before == nrow(df) || @printf(
        "Resume: removed %d duplicate checkpoint row(s) from %s.\n",
        before - nrow(df), basename(source),
    )
    @printf("Resume: loaded %d row(s) from %s.\n", nrow(df), basename(source))
    return [NamedTuple(row) for row in eachrow(df)]
end

row_key(row, fields::Tuple) = Tuple(getproperty(row, field) for field in fields)

function completed_case_keys(rows, key_fields::Tuple, expected_methods::Vector{String})
    methods_by_key = Dict{Tuple,Set{String}}()
    for row in rows
        key = row_key(row, key_fields)
        push!(get!(methods_by_key, key, Set{String}()), String(row.method))
    end
    expected = Set(expected_methods)
    return Set(key for (key, methods) in methods_by_key if expected ⊆ methods)
end

function drop_case!(rows, key_fields::Tuple, key::Tuple)
    filter!(row -> row_key(row, key_fields) != key, rows)
    return rows
end

# -----------------------------------------------------------------------------
# Static RO
# -----------------------------------------------------------------------------
ro_key_fields = (:group, :seed, :n_tx, :n_j, :m_ue, :s, :q)
ro_rows = aro_only ? NamedTuple[] : load_resume_rows(
    "ro_raw_checkpoint.csv", "ro_raw.csv",
    [:group, :seed, :n_tx, :n_j, :m_ue, :s, :q, :method],
)
completed_ro = completed_case_keys(ro_rows, ro_key_fields, ro_methods)
if !aro_only
    for (group_id, spec) in enumerate(ro_specs), seed in seeds
        ntx, nj, mue, sparsity, q = spec
        case_key = (group_id, seed, ntx, nj, mue, sparsity, q)
        if case_key in completed_ro
            @printf("Resume: skipping RO group %d seed %d (checkpoint complete).\n",
                    group_id, seed)
            continue
        end
        drop_case!(ro_rows, ro_key_fields, case_key)
        cfg = WirelessSparseRO.ROConfig(
            seed = seed,
            instance_design = :dichasus,
            n_tx = ntx,
            n_j = nj,
            m_ue = mue,
            q_groups = q,
            sparsity = sparsity,
            jammer_stress_fraction = WIRELESS_JAMMER_FRACTION,
            resource_capacity = WIRELESS_RESOURCE_CAPACITY,
            mip_time_limit_s = quick ? 90.0 : 180.0,
            method_time_limit_s = quick ? 300.0 : 1000.0,
            max_iterations = 80,
        )
        @printf("\nRO group %d seed %d: (%d,%d,%d,%d,q=%d)\n",
                group_id, seed, ntx, nj, mue, sparsity, q)
        _, _, results = WirelessSparseRO.run_one_ro(
            cfg;
            methods = ro_methods,
            validate = true,
            verbose = false,
        )
        for result in results
            push!(ro_rows, merge((
                group = group_id,
                seed = seed,
                n_tx = ntx,
                n_j = nj,
                m_ue = mue,
                s = sparsity,
                q = q,
            ), result))
        end
        CSV.write(joinpath(outdir, "ro_raw_checkpoint.csv"), DataFrame(ro_rows))
    end
end

rodf = DataFrame(ro_rows)
if nrow(rodf) > 0
    CSV.write(joinpath(outdir, "ro_raw.csv"), rodf)
end

# -----------------------------------------------------------------------------
# ARO exact validation and scalable suite
# -----------------------------------------------------------------------------
validation_key_fields = (:seed,)
validation_methods = vcat(
    ["EXACT-FULL", "AADR-EXACT", "RO-EL"],
    ["DB-$(uppercase(method))" for method in aro_separator_methods],
)
validation_rows = ro_only ? NamedTuple[] : load_resume_rows(
    "aro_validation_raw_checkpoint.csv", "aro_validation_raw.csv", [:seed, :method],
)
completed_validation = completed_case_keys(validation_rows, validation_key_fields, validation_methods)

aro_key_fields = (:group, :seed, :n_tx, :m_ue, :n_j, :s, :q)
aro_methods = ["DB-$(uppercase(method))" for method in aro_separator_methods]
run_ro_reference && push!(aro_methods, "RO-EL")
run_aadr_rl && push!(aro_methods, "AADR-RL")
run_aadr_rl_el && push!(aro_methods, "AADR-RL/EL")
aro_rows = ro_only ? NamedTuple[] : load_resume_rows(
    "aro_raw_checkpoint.csv", "aro_raw.csv",
    [:group, :seed, :n_tx, :m_ue, :n_j, :s, :q, :method],
)
if resume && !isempty(aro_rows)
    checkpoint_methods = Set(String(row.method) for row in aro_rows)
    if "AADR-RL/EL" in checkpoint_methods && !run_aadr_rl_el
        error(
            "The checkpoint contains AADR-RL/EL rows. Resume with " *
            "--with-aadr-rl-el to keep the method set consistent.",
        )
    end
    if "AADR-RL" in checkpoint_methods && !run_aadr_rl
        error(
            "The checkpoint contains AADR-RL rows. Do not use --skip-aadr-rl " *
            "when resuming this run.",
        )
    end
end
completed_aro = completed_case_keys(aro_rows, aro_key_fields, aro_methods)

diag_key_fields = (:group, :seed)
aro_diag_rows = ro_only ? NamedTuple[] : load_resume_rows(
    "aro_generation_diagnostics_checkpoint.csv",
    "aro_generation_diagnostics.csv",
    [:group, :seed],
)
completed_diag = Set(row_key(row, diag_key_fields) for row in aro_diag_rows)

if !ro_only
    println("\nARO classification: DB-SP/RL/EL = lower-bound methods; AADR-EXACT = small exact comparison; AADR-RL = restricted affine policy.")
    println("Scalable AADR-RL/EL policy-quality procedure: ", run_aadr_rl_el ? "enabled" : "disabled (use --with-aadr-rl-el to enable)")

    # Tiny exact validation: the enumerated AADR is the independent
    # affine-policy comparison. It is confined to this finite test.
    for seed in seeds
        validation_key = (seed,)
        if validation_key in completed_validation
            @printf("Resume: skipping ARO exact validation seed %d (checkpoint complete).\n", seed)
            continue
        end
        drop_case!(validation_rows, validation_key_fields, validation_key)
        cfg = WirelessSparseARO.AROConfig(
            seed = seed,
            instance_design = :dichasus,
            n_tx = 5,
            n_j = 5,
            m_ue = 5,
            sparsity = 3,
            q_groups = 5,
            jammer_stress_fraction = WIRELESS_JAMMER_FRACTION,
            resource_capacity = WIRELESS_RESOURCE_CAPACITY,
            min_group_size = 3,
            recovery_cluster_size = 5,
            mip_time_limit_s = quick ? 90.0 : 180.0,
            method_time_limit_s = quick ? 300.0 : 1000.0,
            aadr_batch_size = 10,
        )
        _, _, results = WirelessSparseARO.run_one_aro(
            cfg;
            methods = aro_separator_methods,
            run_aadr_rl_el = false,
            run_aadr_rl = false,
            run_ro = true,
            exact_validation = true,
        )
        for result in results
            push!(validation_rows, merge((
                seed = seed,
                role = aro_role(result.method),
                runtime_use = runtime_use(result.method),
            ), result))
        end
        CSV.write(joinpath(outdir, "aro_validation_raw_checkpoint.csv"), DataFrame(validation_rows))
    end

    # Scalable clustered ARO suite.
    for (group_id, spec) in enumerate(aro_specs), seed in seeds
        ntx, mue, nj, sparsity, q = spec
        case_key = (group_id, seed, ntx, mue, nj, sparsity, q)
        if case_key in completed_aro && (group_id, seed) in completed_diag
            @printf("Resume: skipping ARO group %d seed %d (checkpoint complete).\n",
                    group_id, seed)
            continue
        end
        drop_case!(aro_rows, aro_key_fields, case_key)
        drop_case!(aro_diag_rows, diag_key_fields, (group_id, seed))
        cfg = WirelessSparseARO.AROConfig(
            seed = seed,
            instance_design = :dichasus,
            n_tx = ntx,
            n_j = nj,
            m_ue = mue,
            sparsity = sparsity,
            q_groups = q,
            p_max = 80.0,
            jammer_stress_fraction = WIRELESS_JAMMER_FRACTION,
            resource_capacity = WIRELESS_RESOURCE_CAPACITY,
            memberships_per_jammer = 4,
            min_group_size = 6,
            budget_multiplier_low = 1.5,
            budget_multiplier_high = 2.4,
            normalize_sinr_rows = false,
            recovery_cluster_size = 5,
            aadr_batch_size = 10,
            aadr_screen_with_rl = true,
            mip_time_limit_s = quick ? 90.0 : 180.0,
            method_time_limit_s = quick ? 300.0 : 1000.0,
            max_iterations = 80,
            alternating_starts = 10,
            alternating_max_iterations = 30,
        )
        @printf("\nARO group %d seed %d: (%d,%d,%d,%d,q=%d, stress=%.2f)\n",
                group_id, seed, ntx, mue, nj, sparsity, q, cfg.jammer_stress_fraction)
        ins, _, results = WirelessSparseARO.run_one_aro(
            cfg;
            methods = aro_separator_methods,
            run_aadr_rl_el = run_aadr_rl_el,
            run_aadr_rl = run_aadr_rl,
            run_ro = run_ro_reference,
            exact_validation = false,
        )
        for result in results
            push!(aro_rows, merge((
                group = group_id,
                seed = seed,
                n_tx = ntx,
                m_ue = mue,
                n_j = nj,
                s = sparsity,
                q = q,
                jammer_fraction = cfg.jammer_stress_fraction,
                recovery_cluster_size = cfg.recovery_cluster_size,
                role = aro_role(result.method),
                runtime_use = runtime_use(result.method),
            ), result))
        end
        push!(aro_diag_rows, (
            group = group_id,
            seed = seed,
            nominal_objective = get(ins.diagnostics, "nominal_objective", NaN),
            n_recovery_clusters = get(ins.diagnostics, "n_recovery_clusters", NaN),
            recovery_cluster_size = get(ins.diagnostics, "recovery_cluster_size", NaN),
            jammer_scale = get(ins.diagnostics, "jammer_scale", NaN),
            mean_jammer_memberships = get(ins.diagnostics, "mean_jammer_memberships", NaN),
            mean_group_size = get(ins.diagnostics, "mean_group_size", NaN),
            min_group_size = get(ins.diagnostics, "min_group_size", NaN),
            max_group_size = get(ins.diagnostics, "max_group_size", NaN),
            dichasus_measurement = get(ins.diagnostics, "dichasus_measurement", 0.0),
            measurement_duration_s = get(ins.diagnostics, "measurement_duration_s", NaN),
            measured_antenna_count = get(ins.diagnostics, "measured_antenna_count", NaN),
            subband_count = get(ins.diagnostics, "subband_count", NaN),
        ))
        CSV.write(joinpath(outdir, "aro_raw_checkpoint.csv"), DataFrame(aro_rows))
        CSV.write(joinpath(outdir, "aro_generation_diagnostics_checkpoint.csv"), DataFrame(aro_diag_rows))
    end
end

audf = DataFrame(validation_rows)
arodf = DataFrame(aro_rows)
if nrow(audf) > 0
    CSV.write(joinpath(outdir, "aro_validation_raw.csv"), audf)
end
if nrow(arodf) > 0
    CSV.write(joinpath(outdir, "aro_raw.csv"), arodf)
    CSV.write(joinpath(outdir, "aro_generation_diagnostics.csv"), DataFrame(aro_diag_rows))
end

# -----------------------------------------------------------------------------
# RO summaries
# -----------------------------------------------------------------------------
rosdf = DataFrame()
if nrow(rodf) > 0
    ro_summary_rows = NamedTuple[]
    for group_id in sort(unique(rodf.group))
        group_data = rodf[rodf.group .== group_id, :]
        spec = group_data[1, :]
        el = group_data[group_data.method .== "EL", :]
        for method in ["SP", "RL", "EL"]
            x = group_data[group_data.method .== method, :]
            objective_mean, objective_sd = mean_sd(x.objective)
            completed = occursin.(Ref("CONVERGED"), String.(x.status))
            time_mean, time_sd = mean_sd(x.wall_time_s[completed])
            gaps = Float64[]
            if method != "EL"
                for seed in unique(x.seed)
                    a = x[x.seed .== seed, :]
                    b = el[el.seed .== seed, :]
                    if nrow(a) > 0 && nrow(b) > 0 &&
                       isfinite(a.objective[1]) && isfinite(b.objective[1]) &&
                       occursin("CONVERGED", String(a.status[1])) &&
                       occursin("CONVERGED", String(b.status[1]))
                        push!(gaps, 100.0 * (b.objective[1] - a.objective[1]) /
                                    max(abs(b.objective[1]), 1.0e-12))
                    end
                end
            else
                gaps = [0.0]
            end
            gap_mean, gap_sd = mean_sd(gaps)
            exact_violations = finitevals(x.exact_max_violation)
            exact_violation_mean, exact_violation_sd = mean_sd(exact_violations)
            exact_violation_max = isempty(exact_violations) ? NaN : maximum(exact_violations)
            unfinished = sum(.!completed)
            push!(ro_summary_rows, (
                group = group_id,
                n_tx = spec.n_tx,
                n_j = spec.n_j,
                m_ue = spec.m_ue,
                s = spec.s,
                q = spec.q,
                method = method,
                objective_mean = objective_mean,
                objective_sd = objective_sd,
                time_mean = time_mean,
                time_sd = time_sd,
                gap_to_el_mean = gap_mean,
                gap_to_el_sd = gap_sd,
                exact_violation_mean = exact_violation_mean,
                exact_violation_sd = exact_violation_sd,
                exact_violation_max = exact_violation_max,
                incomplete_count = unfinished,
            ))
        end
    end
    rosdf = DataFrame(ro_summary_rows)
    CSV.write(joinpath(outdir, "ro_summary.csv"), rosdf)
end

# -----------------------------------------------------------------------------
# ARO validation metrics and scalable summaries
# -----------------------------------------------------------------------------
metdf = DataFrame()
if nrow(audf) > 0
    function validation_objective(seed, method)
        x = audf[(audf.seed .== seed) .& (audf.method .== method), :]
        return nrow(x) == 0 ? NaN : Float64(x.objective[1])
    end

    metric_rows = NamedTuple[]
    for seed in unique(audf.seed)
        exact_full = validation_objective(seed, "EXACT-FULL")
        db_rl = validation_objective(seed, "DB-RL")
        db_el = validation_objective(seed, "DB-EL")
        aadr = validation_objective(seed, "AADR-EXACT")
        ro_el = validation_objective(seed, "RO-EL")
        push!(metric_rows, (
            seed = seed,
            exact_full = exact_full,
            db_rl = db_rl,
            db_el = db_el,
            aadr = aadr,
            ro_el = ro_el,
            db_rl_lb_gap_pct = 100.0 * (exact_full - db_rl) / exact_full,
            db_el_lb_gap_pct = 100.0 * (exact_full - db_el) / exact_full,
            affine_premium_pct = 100.0 * (aadr - exact_full) / exact_full,
            one_stage_premium_pct = 100.0 * (ro_el - exact_full) / exact_full,
        ))
    end
    metdf = DataFrame(metric_rows)
    CSV.write(joinpath(outdir, "aro_validation_metrics.csv"), metdf)
end

arosdf = DataFrame()
policydf = DataFrame()
if nrow(arodf) > 0
    aro_summary_rows = NamedTuple[]
    scalable_methods = ["DB-SP", "DB-RL", "DB-EL", "AADR-RL", "AADR-RL/EL", "RO-EL"]
    for group_id in sort(unique(arodf.group)), method in scalable_methods
        x = arodf[(arodf.group .== group_id) .& (arodf.method .== method), :]
        nrow(x) == 0 && continue
        el = arodf[(arodf.group .== group_id) .& (arodf.method .== "DB-EL"), :]
        spec = x[1, :]
        usable_mask = [usable_aro_objective(method, String(x.status[i]), x.objective[i]) for i in 1:nrow(x)]
        usable_objectives = [Float64(x.objective[i]) for i in 1:nrow(x) if usable_mask[i]]
        objective_mean, objective_sd = mean_sd(usable_objectives)
        time_mean, time_sd = mean_sd(x.wall_time_s)
        db_el_differences = Float64[]
        if startswith(method, "DB-")
            if method == "DB-EL"
                db_el_differences = [0.0]
            else
                for seed in sort(unique(x.seed))
                    a = x[x.seed .== seed, :]
                    b = el[el.seed .== seed, :]
                    if nrow(a) > 0 && nrow(b) > 0 &&
                       isfinite(a.objective[1]) && isfinite(b.objective[1])
                        push!(
                            db_el_differences,
                            100.0 * (b.objective[1] - a.objective[1]) /
                            max(abs(b.objective[1]), 1.0e-12),
                        )
                    end
                end
            end
        end
        db_el_difference_mean, db_el_difference_sd = mean_sd(db_el_differences)
        incomplete_count = count(i -> incomplete_status(String(x.status[i])), 1:nrow(x))
        push!(aro_summary_rows, (
            group = group_id,
            n_tx = spec.n_tx,
            m_ue = spec.m_ue,
            n_j = spec.n_j,
            s = spec.s,
            q = spec.q,
            method = method,
            role = aro_role(method),
            runtime_use = runtime_use(method),
            objective_mean = objective_mean,
            objective_sd = objective_sd,
            time_mean = time_mean,
            time_sd = time_sd,
            difference_to_db_el_mean = db_el_difference_mean,
            difference_to_db_el_sd = db_el_difference_sd,
            usable_count = length(usable_objectives),
            attempted_count = nrow(x),
            incomplete_count = incomplete_count,
        ))
    end
    arosdf = DataFrame(aro_summary_rows)
    CSV.write(joinpath(outdir, "aro_summary.csv"), arosdf)

    # Seed-level policy-to-lower-bound separations.  These are deliberately not
    # called optimality gaps.  Every policy value is used only when certified.
    comparison_rows = NamedTuple[]
    for group_id in sort(unique(arodf.group)), seed in sort(unique(arodf.seed))
        d = arodf[(arodf.group .== group_id) .& (arodf.seed .== seed), :]
        nrow(d) == 0 && continue
        rl = first_method_row(d, "DB-RL")
        aadr_rl = first_method_row(d, "AADR-RL")
        aadr_rl_el = first_method_row(d, "AADR-RL/EL")
        ro = first_method_row(d, "RO-EL")
        lower = rl === nothing ? NaN : Float64(rl.objective)
        aadr_rl_value = (aadr_rl !== nothing && String(aadr_rl.status) == "CERTIFIED_AADR_RL_POLICY") ? Float64(aadr_rl.objective) : NaN
        aadr_rl_el_value = (aadr_rl_el !== nothing && String(aadr_rl_el.status) == "CERTIFIED_AFFINE_POLICY") ? Float64(aadr_rl_el.objective) : NaN
        ro_value = (ro !== nothing && String(ro.status) == "CERTIFIED_ONE_STAGE") ? Float64(ro.objective) : NaN
        push!(comparison_rows, (
            group = group_id,
            seed = seed,
            db_rl_lower_bound = lower,
            aadr_rl_policy_value = aadr_rl_value,
            aadr_rl_el_certified_value = aadr_rl_el_value,
            ro_el_certified_value = ro_value,
            aadr_rl_minus_db_rl_pct = isfinite(lower) && isfinite(aadr_rl_value) ?
                100.0 * (aadr_rl_value - lower) / max(abs(lower), 1.0e-12) : NaN,
            aadr_rl_el_minus_db_rl_pct = isfinite(lower) && isfinite(aadr_rl_el_value) ?
                100.0 * (aadr_rl_el_value - lower) / max(abs(lower), 1.0e-12) : NaN,
            ro_el_minus_db_rl_pct = isfinite(lower) && isfinite(ro_value) ?
                100.0 * (ro_value - lower) / max(abs(lower), 1.0e-12) : NaN,
            aadr_rl_improvement_over_ro_pct = isfinite(aadr_rl_value) && isfinite(ro_value) ?
                100.0 * (ro_value - aadr_rl_value) / max(abs(ro_value), 1.0e-12) : NaN,
        ))
    end
    policydf = DataFrame(comparison_rows)
    CSV.write(joinpath(outdir, "aro_policy_to_lower_bound.csv"), policydf)
end

# -----------------------------------------------------------------------------
# LaTeX tables
# -----------------------------------------------------------------------------
if nrow(rosdf) > 0
    open(joinpath(outdir, "ro_summary_table.tex"), "w") do io
        println(io, "\\begin{table}[htpb!]\\centering\\scriptsize")
        println(io, "\\caption{Static wireless summary from the DICHASUS measurements. Each entry of the form a[b] gives the mean a and sample standard deviation b over three instances. Exact viol. is obtained by fixing the returned power vector and solving the binary separator for every user.}\\label{tab:summary-results-wireless-poly}")
        println(io, "\\begin{tabular}{llrrrrr}\\toprule Group & Method & Obj. & Time & Diff. to EL (\\%) & Exact viol. & NC/TL \\\\ \\midrule")
        for r in eachrow(rosdf)
            println(io, "\\#$(r.group) ($(r.n_tx),$(r.n_j),$(r.m_ue),$(r.s)) & $(r.method) & $(fmt4(r.objective_mean)) [$(fmt4(r.objective_sd))] & $(fmt2(r.time_mean)) [$(fmt2(r.time_sd))] & $(fmt2(r.gap_to_el_mean)) [$(fmt2(r.gap_to_el_sd))] & $(fmtsci(r.exact_violation_max)) & $(r.incomplete_count) \\\\")
        end
        println(io, "\\bottomrule\\end{tabular}\\end{table}")
    end
end

if nrow(metdf) > 0
    affine_mean, affine_sd = mean_sd(metdf.affine_premium_pct)
    static_mean, static_sd = mean_sd(metdf.one_stage_premium_pct)
    lower_mean, lower_sd = mean_sd(metdf.db_rl_lb_gap_pct)
    open(joinpath(outdir, "aro_validation_table.tex"), "w") do io
        println(io, "\\begin{table}[htpb!]\\centering\\scriptsize")
        println(io, "\\caption{Small instance exact validation of the adjustable wireless model using DICHASUS measurements over $(length(seeds)) instances.}\\label{tab:aro-wireless-small-validation}")
        println(io, "\\begin{tabular}{lrr}\\toprule Metric & Mean & SD \\\\ \\midrule")
        println(io, "DB-RL gap to exact ARO (\\%) & $(fmt2(lower_mean)) & $(fmt2(lower_sd)) \\\\")
        println(io, "AADR-EXACT premium over exact ARO (\\%) & $(fmt2(affine_mean)) & $(fmt2(affine_sd)) \\\\")
        println(io, "One-stage policy premium over exact ARO (\\%) & $(fmt2(static_mean)) & $(fmt2(static_sd)) \\\\")
        println(io, "\\bottomrule\\end{tabular}\\end{table}")
    end
end

if nrow(arosdf) > 0
    primary = arosdf[startswith.(String.(arosdf.method), Ref("DB-")), :]
    open(joinpath(outdir, "aro_summary_table.tex"), "w") do io
        println(io, "\\begin{table}[htpb!]\\centering\\scriptsize")
        println(io, "\\caption{Scalable adjustable wireless results for the proposed DB-SP, DB-RL, and DB-EL methods. Each objective is a valid lower bound. The difference to DB-EL compares locally separated lower bounds and is not an optimality gap.}\\label{tab:aro-wireless-summary}")
        println(io, "\\begin{tabular}{llrrrrr}\\toprule Group & Method & Lower bound & Time & Diff. to DB-EL (\\%) & Usable & NC/TL \\\\ \\midrule")
        for r in eachrow(primary)
            println(io, "\\#$(r.group) ($(r.n_tx),$(r.m_ue),$(r.n_j),$(r.s)) & $(r.method) & $(fmt4(r.objective_mean)) [$(fmt4(r.objective_sd))] & $(fmt2(r.time_mean)) [$(fmt2(r.time_sd))] & $(fmt2(r.difference_to_db_el_mean)) [$(fmt2(r.difference_to_db_el_sd))] & $(r.usable_count)/$(r.attempted_count) & $(r.incomplete_count) \\\\")
        end
        println(io, "\\bottomrule\\end{tabular}\\end{table}")
    end

    affine_rl = arosdf[arosdf.method .== "AADR-RL", :]
    if nrow(affine_rl) > 0
        open(joinpath(outdir, "aro_affine_rl_table.tex"), "w") do io
            println(io, "\\begin{table}[htpb!]\\centering\\scriptsize")
            println(io, "\\caption{Direct AADR-RL affine policy over the RL polyhedral relaxation. This is a policy comparison, not an independent timing comparison.}\\label{tab:aro-wireless-aadr-rl-policy}")
            println(io, "\\begin{tabular}{lrrrr}\\toprule Group & Policy value & Time & Usable & NC/TL \\\\ \\midrule")
            for r in eachrow(affine_rl)
                println(io, "\\#$(r.group) ($(r.n_tx),$(r.m_ue),$(r.n_j),$(r.s)) & $(fmt4(r.objective_mean)) [$(fmt4(r.objective_sd))] & $(fmt2(r.time_mean)) [$(fmt2(r.time_sd))] & $(r.usable_count)/$(r.attempted_count) & $(r.incomplete_count)", latex_row_end)
            end
            println(io, "\\bottomrule\\end{tabular}\\end{table}")
        end
    end

    policy_quality = arosdf[in.(arosdf.method, Ref(["AADR-RL/EL", "RO-EL"])), :]
    if nrow(policy_quality) > 0
        open(joinpath(outdir, "aro_policy_quality_table.tex"), "w") do io
            println(io, "\\begin{table}[htpb!]\\centering\\scriptsize")
            println(io, "\\caption{Policy comparisons on the scalable cases. AADR-RL/EL uses RL and EL separation, while RO-EL uses EL separation. Values are shown only for certified policies.}\\label{tab:aro-wireless-policy-quality}")
            println(io, "\\begin{tabular}{llrr}\\toprule Group & Policy & Certified value & Certified/attempted \\\\ \\midrule")
            for r in eachrow(policy_quality)
                println(io, "\\#$(r.group) ($(r.n_tx),$(r.m_ue),$(r.n_j),$(r.s)) & $(r.method) & $(fmt4(r.objective_mean)) [$(fmt4(r.objective_sd))] & $(r.usable_count)/$(r.attempted_count)", latex_row_end)
            end
            println(io, "\\bottomrule\\end{tabular}\\end{table}")
        end
    end
end

# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------
function paired_bar_figure(
    summary::DataFrame,
    methods::Vector{String},
    difference_column::Symbol,
    difference_reference::String,
    stem::String,
)
    groups = sort(unique(summary.group))
    colors = Dict(
        methods[1] => RGB(0.18, 0.42, 0.66),
        methods[2] => RGB(0.90, 0.48, 0.13),
        methods[3] => RGB(0.24, 0.58, 0.35),
    )
    offsets = Dict(methods[1] => -0.25, methods[2] => 0.0, methods[3] => 0.25)
    common = (
        fontfamily = "Computer Modern",
        framestyle = :box,
        guidefontsize = 9,
        tickfontsize = 8,
        legendfontsize = 8,
        xgrid = false,
        ygrid = true,
        gridalpha = 0.17,
        gridlinewidth = 1.0,
        foreground_color_axis = :black,
        foreground_color_border = :black,
        background_color = :white,
        background_color_legend = :white,
        foreground_color_legend = :black,
    )

    largest_difference = maximum(Float64.(summary[!, difference_column]))
    difference_plot = plot(
        ; common...,
        xlabel = "Instance group",
        ylabel = "Difference from $difference_reference (%)",
        xticks = groups,
        yticks = 0.0:10.0:40.0,
        xlims = (minimum(groups) - 0.52, maximum(groups) + 0.52),
        ylims = (0.0, max(5.0, 1.28 * largest_difference)),
        legend = :outertop,
        legend_column = 2,
        size = (380, 300),
        left_margin = 8Plots.mm,
        right_margin = 3Plots.mm,
        bottom_margin = 6Plots.mm,
        top_margin = 1Plots.mm,
    )
    for method in methods[1:2]
        rows = sort(summary[summary.method .== method, :], :group)
        values = Float64.(rows[!, difference_column])
        bar!(
            difference_plot,
            rows.group .+ offsets[method],
            values,
            bar_width = 0.22,
            color = colors[method],
            fillalpha = 0.92,
            linecolor = :black,
            linewidth = 1.4,
            label = method,
        )
    end

    positive_times = Float64.(summary.time_mean[summary.time_mean .> 0.0])
    time_floor = 10.0 ^ floor(log10(minimum(positive_times)))
    time_ceiling = 10.0 ^ ceil(log10(2.2 * maximum(positive_times)))
    time_plot = plot(
        ; common...,
        xlabel = "Instance group",
        ylabel = "Computational time (s)",
        yscale = :log10,
        xticks = groups,
        xlims = (minimum(groups) - 0.52, maximum(groups) + 0.52),
        ylims = (time_floor, time_ceiling),
        legend = :outertop,
        legend_column = 3,
        size = (380, 300),
        left_margin = 8Plots.mm,
        right_margin = 3Plots.mm,
        bottom_margin = 6Plots.mm,
        top_margin = 1Plots.mm,
    )
    for method in methods
        rows = sort(summary[summary.method .== method, :], :group)
        values = Float64.(rows.time_mean)
        bar!(
            time_plot,
            rows.group .+ offsets[method],
            values,
            bar_width = 0.22,
            fillrange = time_floor,
            color = colors[method],
            fillalpha = 0.92,
            linecolor = :black,
            linewidth = 1.4,
            label = method,
        )
    end

    savefig(difference_plot, joinpath(outdir, "$(stem)_difference.pdf"))
    savefig(time_plot, joinpath(outdir, "$(stem)_time.pdf"))
    combined = plot(
        difference_plot,
        time_plot,
        layout = (1, 2),
        size = (900, 360),
        margin = 2Plots.mm,
    )
    savefig(combined, joinpath(outdir, "$(stem)_summary.pdf"))
end

try
    nrow(rosdf) > 0 && paired_bar_figure(
        rosdf,
        ["SP", "RL", "EL"],
        :gap_to_el_mean,
        "EL",
        "wireless_static",
    )
    nrow(arosdf) > 0 && paired_bar_figure(
        arosdf[startswith.(String.(arosdf.method), Ref("DB-")), :],
        ["DB-SP", "DB-RL", "DB-EL"],
        :difference_to_db_el_mean,
        "DB-EL",
        "wireless_adjustable",
    )
catch err
    @warn "Plot generation failed; CSV and TeX tables were still written" exception = err
end

# -----------------------------------------------------------------------------
# Generated manuscript paragraphs
# -----------------------------------------------------------------------------
if nrow(rosdf) > 0
    open(joinpath(outdir, "ro_results_paragraph.tex"), "w") do io
        valid = rosdf[(rosdf.method .== "RL") .& isfinite.(rosdf.gap_to_el_mean), :]
        if nrow(valid) > 0
            gap_mean = mean(valid.gap_to_el_mean)
            if length(unique(rosdf.group)) > 1
                println(io, "Across groups for which EL completed, RL stayed close to the EL value, with a mean relative difference of $(fmt2(gap_mean))\\%. The runtime comparison shows how RL and EL separation differ as the number of potential jammers and the sparsity increase.")
            else
                println(io, "In the short check, RL is within $(fmt2(gap_mean))\\% of the certified EL value. This run verifies the experiment pipeline; scaling statements use the complete set of groups.")
            end
        end
    end
end

if nrow(metdf) > 0
    open(joinpath(outdir, "aro_results_paragraph.tex"), "w") do io
        affine_mean, _ = mean_sd(metdf.affine_premium_pct)
        static_mean, _ = mean_sd(metdf.one_stage_premium_pct)
        lower_mean, _ = mean_sd(metdf.db_rl_lb_gap_pct)
        scalable_text = ""
        if nrow(arosdf) > 0
            sp = arosdf[arosdf.method .== "DB-SP", :]
            rl = arosdf[arosdf.method .== "DB-RL", :]
            el = arosdf[arosdf.method .== "DB-EL", :]
            sp_diff = mean_or_nan(sp.difference_to_db_el_mean)
            rl_diff = mean_or_nan(rl.difference_to_db_el_mean)
            runtime_ratio = mean_or_nan([
                el.time_mean[i] / rl.time_mean[i]
                for i in 1:min(nrow(el), nrow(rl))
                if isfinite(el.time_mean[i]) && isfinite(rl.time_mean[i]) &&
                   rl.time_mean[i] > 0.0
            ])
            affine_improvement = nrow(policydf) > 0 ?
                mean_or_nan(policydf.aadr_rl_improvement_over_ro_pct) : NaN
            policy_text = affine_improvement >= 0.0 ?
                "The affine RL policy is $(fmt2(affine_improvement))\\% less costly than static EL on average." :
                "The affine RL policy is $(fmt2(abs(affine_improvement)))\\% more costly than static EL on average because the RL set is conservative."
            scalable_text = " On scalable instances, DB-SP is $(fmt2(sp_diff))\\% below DB-EL on average, while DB-RL differs by $(fmt2(rl_diff))\\%; DB-EL takes $(fmt2(runtime_ratio)) times the DB-RL runtime. $policy_text"
        end
        println(io, "On the small exact instances, the DB-RL lower bound is within $(fmt2(lower_mean))\\% of the fully adjustable optimum on average, whereas the exact affine policy costs $(fmt2(affine_mean))\\% and suppressing recourse entirely costs $(fmt2(static_mean))\\% on average.$(scalable_text) DB-SP, DB-RL, and DB-EL compute lower bounds; direct AADR-RL is an affine policy. Differences among scalable lower bounds and between policies and lower bounds are not labelled optimality gaps.")
    end
end

println("\nFinished. Results written to $outdir")
return nothing
end # function main

end # module WirelessExperimentDriverV11

# Resolve the newly defined driver module and entry point in the latest Julia
# world. This makes repeated notebook `include(...)` calls independent of any
# model modules or exported names already present in `Main`.
Base.@invokelatest WirelessExperimentDriverV11.main(copy(ARGS))
