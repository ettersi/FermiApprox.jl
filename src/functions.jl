"""
    fermi(z,β)

Evaluate `1/(1+exp(β*z))` in a numerically stable way.
"""
function fermi(z,β)
    T = promote_type(typeof.((z,β))...)
    return fermi(convert(T,z),convert(real(T),β))
end
function fermi(z::Union{T,Complex{T}},β::T) where {T <: Real}
    if abs(β*z) < log(realmax(T))
        return 1/(1 + exp(β*z))
    else
        return (1-sign(real(z)))/2
    end
end


"""
    nfermipoles(b,β)

Number of poles of the Fermi-Dirac function between `0` and `b*im`
"""
nfermipoles(b,β) = nfermipoles(promote(b,β)...)
nfermipoles(b::T,β::T) where {T} = floor(Int,(β/π*b + 1)/2)

"""
    fermipoles(k,β) -> z

The `2k` poles of the Fermi-Dirac function closest to the real line,
i.e. `z = π/β*(-2k+1:2:2k-1)`.
"""
fermipoles(k,β) = π/β*im*(-2k+1:2:2k-1)


"""
    fermidiff(z1,z2,β)

Evaluate `( fermi(z1,β) - fermi(z2,β) ) / ( z1 - z2 )` in
a numerically stable way.
"""
function fermidiff(z1,z2,β)
    T = promote_type(typeof.((z1,z2,β))...)
    return fermidiff(convert(T,z1),convert(T,z2),convert(real(T),β))
end
function fermidiff(z1::Union{T,Complex{T}},z2::Union{T,Complex{T}},β::T) where {T <: Real}
    if z1 ≈ z2
        z = (z1+z2)/2
        return -β/4/cosh(β*z/2)^2
    else
        return (fermi(z1,β)-fermi(z2,β))/(z1-z2)
    end
end
