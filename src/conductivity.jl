using ApproxTools

function rationalfactor(β,η)
    T = promote_type(typeof.((β,η))...)
    return rationalfactor(convert(real(T),β),convert(T,η))
end
function rationalfactor(β::T,η::Union{T,Complex{T}}) where {T <: Real}
    b = semiminor(1+η)
    k = nfermipoles(b,β)
    return x->(x^2+b^2)^k/mapreduce(b->(x^2+b^2),*,π/β*(1:2:2k-1))
end

function approx_conductivity(β::Number,η::Number, npoly::Integer,nrat::Integer)
    T = promote_type(typeof.((β,η))...)
    return approx_conductivity(convert(real(T),β),convert(T,η), npoly,nrat)
end
function approx_conductivity(
    β::T,
    η::Union{T,Complex{T}},
    npoly::Integer,
    nrat::Integer
) where {T <: Real}
    q = interpolate(rationalfactor(β,η), Chebyshev(nrat))
    p = interpolate(
        Semiseparated((x1,x2)->fermidiff(x1,x2,β)/(x1-x2+η), (x->1/q(x),x->1/q(x))),
        Chebyshev.((npoly,npoly))
    )
    return Semiseparated(p,(q,q))
end
