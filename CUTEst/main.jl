using LinearAlgebra
using NLPModels
using CUTEst
using Distributions
using SparseArrays
using Random

include("estimator.jl")
include("updating.jl")
include("ProblemSetup.jl")

function SQPEq(ProblemName, L, Gamma)
    setup = ProblemSetup(
        ProblemName = ProblemName,
        L = L,
        Gamma = Gamma,
    )

    Random.seed!(setup.Seed)  
    K = setup.K
    epsilon_g = setup.epsilon_g
    epsilon_c = setup.epsilon_c
    epsilon_J = setup.epsilon_J

    Nlp = setup.Nlp
    x_1 = setup.x_1
    n = setup.n
    m = setup.m
    H = setup.H

    tau_0 = setup.tau_0
    ksi_0 = setup.ksi_0
    beta = setup.beta
    eta = setup.eta
    sigma = setup.sigma
    epsilon_tau = setup.epsilon_tau
    epsilon_ksi = setup.epsilon_ksi
    theta = setup.theta

    tau = []
    ksi = []
    alpha = []
    x = [x_1]
    delta_l_ = []

    for k in 1:K
        Jac_deterministic = J(Nlp, x[end], m, n)
        Grad_deterministic = g(Nlp, x[end])
        Cons_deterministic = C(Nlp, x[end], m)

        Jac = J_rv(Jac_deterministic, m, n, epsilon_J)
        Cons = c_rv(Cons_deterministic, m, epsilon_c)
        Grad = g_rv(Grad_deterministic, n, epsilon_g)

        solution = cal_d_and_y(H, m, Jac, Grad, Cons)
        d_k = solution[1:n]
        delta_l = cal_delta_l(k == 1 ? tau_0 : tau[end], Grad, d_k, Cons)
        push!(delta_l_, delta_l)

        if minimum(svdvals(Jac * Jac')) <= 1e-12
            return "$(ProblemName) does not satisfy the LICQ condition at the $k step."
        end

        if norm(d_k, Inf) > 1e-8
            tau_k = cal_tau(H, d_k, sigma, k == 1 ? tau_0 : tau[end], epsilon_tau, Grad, Cons)
            push!(tau, tau_k)

            ksi_k = cal_ksi(d_k, tau[end], k == 1 ? ksi_0 : ksi[end], epsilon_ksi, Grad, Cons)
            push!(ksi, ksi_k)

            alpha_nn = cal_alpha(d_k, eta, beta, ksi_k, tau_k, L, Gamma, theta, Grad, Cons)
            push!(alpha, alpha_nn)
        else
            push!(tau, k == 1 ? tau_0 : tau[k-1])
            push!(ksi, k == 1 ? ksi_0 : ksi[k-1])
            alpha_nn = 2 * (1 - eta) * beta * ksi[end] * tau[end] / (tau[end] * L + Gamma)
            push!(alpha, alpha_nn)
        end

        push!(x, x[end] + alpha_nn * d_k)
    end

    f, g_, c, KKT, TC, phi, d = [], [], [], [], [], [], []

    for i in 2:K+1
        Obj = F(Nlp, x[i])
        Grad = g(Nlp, x[i])
        Cons = C(Nlp, x[i], m)
        Jac = J(Nlp, x[i], m, n)
        d_k = cal_d_and_y(H, m, Jac, Grad, Cons)[1:n]

        y_kkt = -pinv(Jac') * Grad
        KKT_error = norm(Grad + Jac' * y_kkt, Inf)
        tc = norm(vcat(Grad + Jac' * y_kkt, Cons), Inf)

        push!(f, Obj)
        push!(g_, norm(Grad, Inf))
        push!(c, norm(Cons, Inf))
        push!(KKT, KKT_error)
        push!(TC, tc)
        push!(phi, tau[i-1] * Obj + c[i-1])
        push!(d, norm(d_k, Inf))
    end

    KKT_min, Con_min = cal_result(K, c, KKT)
    optimal_index = findfirst(x -> x == KKT_min, KKT)

    open("$(ProblemName)_$(setup.Seed).txt", "w") do file
        write(file, "Problem Name:\t$ProblemName\n")
        write(file, "Number of variables:\t$n\n")
        write(file, "Number of Constraints:\t$m\n")
        write(file, "Among $K steps, the optimal point is found at the $(optimal_index)th step.\n")
        write(file, "The feasibility error is $Con_min.\n")
        write(file, "The stationarity error is $KKT_min.\n")
        write(file, "Index\tf\t||g||\tFeasibility Error\tStationarity Error\tKKT Error\tphi\t||d||\tdelta_l\ttau\txi\talpha\n")

        for i in 1:K
            write(file, "$i\t$(f[i])\t$(g_[i])\t$(c[i])\t$(KKT[i])\t$(TC[i])\t$(phi[i])\t$(d[i])\t$(delta_l_[i])\t$(tau[i])\t$(ksi[i])\t$(alpha[i])\n")
        end
    end

    return (KKT_min, Con_min), length(x)-1
end
