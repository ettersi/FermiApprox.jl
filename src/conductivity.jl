using ApproxTools
using PyPlot
using PyCall;
@pyimport matplotlib.colors as mplcolors;
using Compat

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


struct PeriodicVector{T} <: AbstractVector{T}
    data::Vector{T}
end
PeriodicVector{T}(l::Integer) where {T} = PeriodicVector{T}(Vector{T}(l))
Base.size(v::PeriodicVector) = size(v.data)
Base.getindex(v::PeriodicVector,i::Int) = v.data[mod1(i,length(v))]
Base.setindex!(v::PeriodicVector,val,i::Int) = v.data[mod1(i,length(v))] = val

type CachedRangeIter{Iter,State,Eltype} <: AbstractVector{Eltype}
    iter::Iter
    state::State
    cache::PeriodicVector{Eltype}
    offset::Int
    first::Int
    last::Int
end
function CachedRangeIter(iter,maxlength)
    state = start(iter)
    return CachedRangeIter{
        typeof(iter),
        typeof(state),
        eltype(iter)
    }(
        iter,
        state,
        PeriodicVector{eltype(iter)}(maxlength),
        1,
        1,0
    )
end
Base.indices(c::CachedRangeIter) = (c.first:c.last,)
function Base.getindex(c::CachedRangeIter,i::Int)
    @assert i in indices(c,1)
    return c.cache[c.offset+i-c.first]
end

function movefirst!(c::CachedRangeIter, i::Integer)
    @assert i >= c.first
    @assert i <= length(c.iter)
    df = i - c.first
    dl = max(0,i-c.last-1)

    # Advance iterator
    for j = 1:dl
        @assert !done(c.iter,c.state)
        _,c.state = next(c.iter,c.state)
    end

    # Free old memory
    if method_exists(eltype(c), Tuple{})
        for j = 1:df
            c.cache[c.offset+j-1] = eltype(c)()
        end
    end

    # Update indices
    c.offset += df
    c.first = i
    c.last += dl
    return c
end
function movelast!(c::CachedRangeIter, i::Integer)
    @assert i >= c.last
    @assert i - c.first < length(c.cache)
    @assert i <= length(c.iter)
    dl = i - c.last
    len = c.last-c.first+1

    # Fill cache
    for j = 1:dl
        @assert !done(c.iter,c.state)
        c.cache[c.offset+len+j-1],c.state = next(c.iter,c.state)
    end

    # Update indices
    c.last = i
    return c
end

function update!(p, c::CachedRangeIter)
    df = findfirst(p, c.first:length(c.iter))-1
    nf = c.first + df
    dl = findlast(p, max(nf,c.last):min(nf+length(c.cache)-1, length(c.iter)))-1
    nl = nf > c.last ?  nf + dl : c.last + max(dl,0)

    if df >= 0
        # If a valid range has been found
        movefirst!(c, nf)
        movelast!(c, nl)
    else
        # If no valid range has been found
        movefirst!(c,c.last+1)
    end
    return c
end


function eval_sparse(f::Semiseparated,H,Da,Db,ε)
    n = size(H,1)
    p = f.core::LinearCombination
    q = f.factors[1]
    b = p.basis[1]::Chebyshev
    c = p.coefficients

    @assert size(H) == size(Da) == size(Db) == (n,n)
    @assert f.factors == (q,q)
    @assert p.basis == (b,b)

    v1 = zeros(n); v1[1] = 1
    v1 = q(H,v1)
    v2 = full(Db[:,1])
    v2 = q(H,v2)

    σ = zero(complex(eltype(H)))
    Tv1_cache = CachedRangeIter(b(H,v1), 3*findlast(c->abs(c) > ε, @view(c[:,1])))
    # Tv1_cache = CachedRangeIter(b(H,v1), size(c,1))
    # movelast!(Tv1_cache, size(c,1))
    for (i2,Tv2) in enumerate(b(H,v2))
        update!(i1->abs(c[i1,i2]) > ε, Tv1_cache)

        for (i1,Tv1) in enumerate(IndexLinear(),Tv1_cache)
            σ += c[i1,i2]*(Tv1'*Da*Tv2)
        end
    end
    return σ/n
end

# Evaluate the conductivity formula by diagonalising the Hamiltonian
function eval_diag(f,H,Da,Db)
    n = size(H,1)
    E,Ψ = eig(full(H))
    D̃a = Ψ'*Da*Ψ
    D̃b = Ψ'*Db[:,1]
    F = grideval(f,(E,E))
    return 1/n*sum(
        F[k1,k2] *
        Ψ[1,k1]*D̃a[k1,k2]*D̃b[k2]
        for k1 = 1:n, k2 = 1:n
    )
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
