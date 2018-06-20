using ApproxTools
using PyPlot
using PyCall;
@pyimport matplotlib.colors as mplcolors;

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

function plot_convergence()
    β = 100
    η = 1/β*im

    f = (x1,x2)->fermidiff(x1,x2,β)/(x1-x2+η)

    n = 1:50:2000
    @time err = [begin
        p = approx_conductivity(β,η, n,n)
        fnorm(f,p)/fnorm(f)
    end
    for n = n]

    fig = figure(figsize=(6,4.5))
    try
        semilogy(n, err, label="relative error");
        semilogy(n, abs(ijouk(η/2)).^(-2n), "k--", label="theoretical convergence rate")
        xlabel("polynomial degree")
        legend(loc="best")

        savefig("convergence.png")
        println("Plot saved at $(pwd())/convergence.png")
    finally
        close(fig)
    end
end

function plot_coeffs()
    β = 100
    η = 1/β*im
    n = 2001

    @time p = approx_conductivity(β,η,n,n)
    C = abs.(coeffs(p.core))
    C ./= maximum(C)

    fig = figure(figsize=(6,4.5))
    try
        imshow(C, norm=mplcolors.LogNorm(), vmin=1e-14)
        colorbar()
        savefig("coefficients.png")
        println("Plot saved at $(pwd())/coefficients.png")
    finally
        close(fig)
    end
end
