module Pith

using StatsFuns

export pith
include("kernels.jl")

function pith{T}(x, X::Vector{T}; m::Kernel = Gaussian())
    n = length(X)
    ĝ = zeros(T, n)
    h = (1.364 * std(X))/(pi*n^(1/5))
    @inbounds for i in 1:length(x)
        xi = x[i]
        for j = 1:length(X)
            ĝ[i] += m((X[j]-xi)/h)
        end
    end
    ĝ/(n*h)
end

end # module
