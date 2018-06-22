@testset "CachedRangeIter" begin
    F = FermiApprox
    a = F.CachedRangeIter(1:10,3)
    @test collect( F.movelast!(a,3)) == [1,2,3]
    @test collect(F.movefirst!(a,3)) == [3]
    @test collect(F.movefirst!(a,4)) == []
    @test collect( F.movelast!(a,4)) == [4]
    @test collect( F.movelast!(a,6)) == [4,5,6]
    @test collect(F.movefirst!(a,5)) == [5,6]
    @test collect(F.movefirst!(a,6)) == [6]
    @test collect(F.movefirst!(a,7)) == []
    @test collect( F.movelast!(a,8)) == [7,8]
    @test collect( F.movelast!(a,9)) == [7,8,9]

    a = F.CachedRangeIter(1:10,3)
    @test collect(F.update!(i->i in 1:2, a)) == [1,2]
    @test collect(F.update!(i->i in 2:3, a)) == [2,3]
    @test collect(F.update!(i->i in 2:4, a)) == [2,3,4]
    @test collect(F.update!(i->i in 5:4, a)) == []
    @test collect(F.update!(i->i in 5:6, a)) == [5,6]
end

@testset "eval_sparse vs eval_diag" begin
    T = ApproxTools
    F = FermiApprox
    β = 50
    η = 1/β*im
    n = 200
    p = F.approx_conductivity(β,η,n,n)

    N = 100
    H = Tridiagonal(-ones(N-1),zeros(N),-ones(N-1))/2

    ∂H = Diagonal(1:N)*H - H*Diagonal(1:N)


    @time σ1 = F.eval_sparse(p,H,∂H,∂H,1e-3*maximum(abs.(p.core.coefficients)))
    @time σ2 = F.eval_diag(p,H,∂H,∂H)
    @show abs(σ1-σ2)/abs(σ2)
end
