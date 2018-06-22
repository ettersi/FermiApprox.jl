module FermiApprox

include("functions.jl")
export fermi, nfermipoles, fermipoles, fermidiff

include("conductivity.jl")
export rationalfactor, approx_conductivity, eval_sparse, eval_diag

end # module
