module SparseAGP

using LinearAlgebra
using Random

export AGPResult, agp_maximize, hard_threshold

struct AGPResult
    point::Vector{Float64}
    value::Float64
    iterations::Int
    backtracks::Int
    final_gamma::Float64
end

"""Project onto the `s`-sparse set, with smaller indices winning ties."""
function hard_threshold(v::AbstractVector{<:Real}, s::Int)
    0 <= s <= length(v) || error("s must lie between zero and the vector dimension")
    order = sortperm(collect(eachindex(v)); by = i -> (-abs(v[i]), i))
    projected = zeros(Float64, length(v))
    for i in order[1:s]
        projected[i] = v[i]
    end
    return projected
end

"""
    agp_maximize(value_function, gradient_function, s, initial_point, max_iterations; ...)

Adaptive Gradient Projection for maximizing a smooth function over the set of
vectors with at most `s` nonzero components. The update is the gradient-ratio
curvature estimate, gamma scaling, hard-thresholding projection, and sufficient-
increase backtracking stated in Algorithm 1 of the manuscript.
"""
function agp_maximize(
    value_function,
    gradient_function,
    s::Int,
    initial_point::AbstractVector{<:Real},
    max_iterations::Int;
    gamma_initial::Float64 = 0.25,
    gamma_minimum::Float64 = 1.0e-4,
    delta::Float64 = 1.0e-10,
    beta::Float64 = 2.0,
    epsilon::Float64 = 1.0e-5,
    initial_seed::Int = 23,
    adapt_gamma::Bool = true,
)
    0.0 < gamma_minimum <= gamma_initial <= 1.0 ||
        error("gamma must satisfy 0 < gamma_minimum <= gamma_initial <= 1")
    delta > 0.0 || error("delta must be positive")
    beta > 1.0 || error("beta must exceed one")
    epsilon >= 0.0 || error("epsilon must be nonnegative")

    current = hard_threshold(initial_point, s)
    current_value = Float64(value_function(current))
    current_gradient = Vector{Float64}(gradient_function(current))

    # The supplied experimental routines use the same deterministic perturbation
    # to obtain the first gradient-ratio estimate.
    rng = Xoshiro(initial_seed)
    initial_displacement = 1.0e-3 .* randn(rng, length(current))
    previous = current .+ initial_displacement
    previous_gradient = Vector{Float64}(gradient_function(previous))

    gamma = gamma_initial
    recent_backtracks = 0
    total_backtracks = 0

    for iteration in 1:max_iterations
        displacement_norm = norm(current - previous)
        displacement_norm > 0.0 ||
            return AGPResult(current, current_value, iteration - 1, total_backtracks, gamma)

        curvature = norm(current_gradient - previous_gradient) / displacement_norm + delta
        local_L = gamma * curvature
        backtracks = 0
        candidate = copy(current)
        candidate_value = current_value

        while true
            candidate = hard_threshold(current + current_gradient / local_L, s)
            step = candidate - current
            candidate_value = Float64(value_function(candidate))
            sufficient_value = current_value + dot(current_gradient, step) -
                               0.5 * local_L * norm(step)^2
            if candidate_value >= sufficient_value
                break
            end
            local_L *= beta
            backtracks += 1
        end

        previous = current
        previous_gradient = current_gradient
        current = candidate
        current_value = candidate_value
        current_gradient = Vector{Float64}(gradient_function(current))
        total_backtracks += backtracks
        recent_backtracks += backtracks

        if norm(current - previous) <= epsilon
            return AGPResult(current, current_value, iteration, total_backtracks, gamma)
        end

        if adapt_gamma && iteration % 10 == 0
            if recent_backtracks == 0
                gamma = max(gamma_minimum, 0.9 * gamma)
            elseif recent_backtracks > 10
                gamma = min(1.0, 1.1 * gamma)
            end
            recent_backtracks = 0
        end
    end

    return AGPResult(current, current_value, max_iterations, total_backtracks, gamma)
end

end
