# ProblemSetup.jl
using LinearAlgebra
using NLPModels
using CUTEst

Base.@kwdef struct ProblemSetup
    K::Int = 1000
    Seed::Int = 41
    ProblemName::String
    L::Float64
    Gamma::Float64
    epsilon_g::Float64 = 1e-8
    epsilon_c::Float64 = 1e-8
    epsilon_J::Float64 = 1e-8

    Nlp::CUTEstModel{Float64} = CUTEstModel{Float64}(ProblemName)
    x_1::Vector{Float64} = Nlp.meta.x0
    n::Int = length(x_1)
    m::Int = length(cons(Nlp, x_1))
    H::Matrix{Float64} = Matrix(I, n, n)

    # Algorithmic parameters
    tau_0::Float64 = 1.0
    ksi_0::Float64 = 1.0
    beta::Float64 = 1.0
    eta::Float64 = 0.5
    sigma::Float64 = 0.1
    epsilon_tau::Float64 = 1e-2
    epsilon_ksi::Float64 = 1e-2
    theta::Float64 = 10.0
end



