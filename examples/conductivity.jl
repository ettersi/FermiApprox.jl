using ApproxTools
using FermiApprox
using PyPlot

nterms(ε,r) = ceil(Int, -log(ε*(1-1/r)) / log(r))

β = 20.0
η = 1/β*im

f = (x1,x2) -> fermidiff(x1,x2,β)/(x1-x2+η)
maxf = fnorm(f)

N = 80
H = Tridiagonal(-ones(N-1),zeros(N),-ones(N-1))/2
∂H = Diagonal(1:N)*H - H*Diagonal(1:N)

σref = eval_diag(f,H,∂H,∂H)

ε = 0.5.^(1:30)

err_approx = zeros(length(ε))
err_sparse = zeros(length(ε))
err_dense = zeros(length(ε))
for i = 1:length(ε)
    n = nterms(ε[i]/maxf, abs(ijouk(η/2))^2)
    bwidth = nterms(ε[i]/maxf, abs(ijouk(1+η)))
    p = approx_conductivity(β,η,n,n)
    err_approx[i] = fnorm(f,p)/maxf
    err_sparse[i] = abs(σref - eval_sparse(p,H,∂H,∂H,bwidth))/abs(σref)
    err_dense[i] = abs(σref - eval_sparse(p,H,∂H,∂H,n))/abs(σref)
end


fig = figure(figsize=(6,4.5))
try
    loglog(ε, err_approx, label="approximation")
    loglog(ε, err_sparse, label="diagonal coeffs");
    loglog(ε, err_dense, label="dense coeffs")
    loglog(ε, ε, "k--", label="tolerance");
    xlabel("tolerance")
    ylabel("relative error")
    legend(loc="best")
    tight_layout()
    savefig("errors.png")
    println("Plot saved at $(pwd())/errors.png")
finally
    close(fig)
end
