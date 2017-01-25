abstract Kernel
type Uniform<:Kernel; end
(::Uniform){T}(x::T) = abs(x) < one(T) ? 1/2 : 0
type Triangular<:Kernel; end
(::Triangular){T}(x::T) = abs(x) < one(T) ? one(T)-abs(x) : zero(T)
type Gaussian<:Kernel; end
(::Gaussian)(x) = normpdf(x)
type Epanechnikov<:Kernel; end
(::Epanechnikov){T}(x::T) = abs(x) < one(T) ? T(3/4)*(one(T)-x^2) : zero(T)
type Biweight<:Kernel; end
(::Biweight){T}(x::T) = abs(x) < one(T) ? T(15/16)*(one(T)-x^2)^2 : zero(T)
type Triweight<:Kernel; end
(::Triweight){T}(x::T) = abs(x) < one(T) ? T(35/32)*(one(T)-x^2)^3 : zero(T)
type Tricube<:Kernel; end
(::Tricube){T}(x::T) = abs(x) < one(T) ? T(70/81)*(one(T)-abs(x)^3)^3 : zero(T)
type Cosine<:Kernel; end
(::Cosine){T}(x::T) = abs(x) < one(T) ? T(π/4)*cos(x*π/2) : zero(T)
type Logistic<:Kernel; end
(::Logistic){T}(x::T) = 1/(2+exp(x)+exp(-x))
type Sigmoid<:Kernel; end
(::Sigmoid){T}(x::T) = 2/(π(exp(x)+exp(-x)))
type Silverman<:Kernel; end
(::Silverman){T}(x::T) = 1/2*exp(-abs(x)/sqrt(2))*sin(abs(x)/sqrt(2)+π/4)
