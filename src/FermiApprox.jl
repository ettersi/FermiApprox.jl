module FermiApprox

include("functions.jl")
export fermi, nfermipoles, fermipoles, fermidiff

include("conductivity.jl")
export rationalfactor, approx_conductivity

end # module
