module SparseRobustRevision

include(joinpath(@__DIR__, "..", "SparseAGP.jl"))
using .SparseAGP: AGPResult, agp_maximize, hard_threshold

export AGPResult, agp_maximize, hard_threshold

end
