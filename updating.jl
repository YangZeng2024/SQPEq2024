function cal_d_and_y(H, m, Jac, Grad, Cons)
   
    A = Matrix([
        H Jac';
        Jac zeros(m,m)
        ])
    b = -vcat(Grad, Cons)

    solution = A \ b

    return solution
    
end

function cal_tau(H, d_k, sigma, tau_pre, epsilon_tau, Grad, Cons)

    tau_trial = 0

    if Grad' * d_k + 0.5 * d_k' * H * d_k <= 1e-12
        tau_trial = Inf
    else
        tau_trial = (1 - sigma) * norm(Cons, 1)/(Grad' * d_k + 0.5 * d_k' * H * d_k)    
    end

    if tau_pre <= tau_trial
        return tau_pre
    else
        return min(tau_trial, (1 - epsilon_tau) * tau_pre)   
    end

end

function cal_ksi(d_k, tau, ksi, epsilon_ksi, Grad, Cons)

    ksi_trial = (-tau * Grad' * d_k + norm(Cons, 1)) / (tau * norm(d_k)^2)
    
    if ksi <= ksi_trial
        return ksi
    else
        return min(ksi_trial, (1 - epsilon_ksi) * ksi)                  
    end 
    
end

function psi(alpha, eta, beta, tau, Grad, d_k, Cons, L, Gamma)

    return (eta - 1) * alpha * beta * (-tau .* (Grad' * d_k) .+ norm(Cons, 1)) + (abs(1 .- alpha) .- 1 .+ alpha) .* norm(Cons, 1) + 0.5 * (tau * L + Gamma) * alpha ^ 2 * (d_k' * d_k)

end

function cal_alpha(d_k, eta, beta, ksi, tau, L, Gamma, theta, Grad, Cons)

    alpha_min = 2 * (1 - eta) * beta * ksi * tau/(tau * L + Gamma) 

    alpha_k = alpha_min

    while psi(1.1 * alpha_k, eta, beta, tau, Grad, d_k, Cons, L, Gamma) < 0 && 1.1 * alpha_k < alpha_min + theta * beta
        alpha_k = 1.1 * alpha_k
    end
    return alpha_k  

end

function cal_result(K, c, KKT)

    c_min = minimum(c)
    kkt_initial = 100
    KKT_min = 0
    Con_min = 0
    optimal_index = 1

    if c_min <= 1e-4
        
        for i in 1: K

            if c[i] <= 1e-4

                if KKT[i] <= kkt_initial
                    kkt_initial = KKT[i]
                    KKT_min = KKT[i]
                    Con_min = c[i]
                    optimal_index = i

                end
            end
        end

    else

        for i in 1: K

            if c[i] == c_min

                KKT_min = KKT[i]
                Con_min = c[i]
                optimal_index = i
      
            end
            
        end

    end

    return KKT_min, Con_min
    
end

function cal_delta_l(tau, Grad, d, Cons)
    return -tau * Grad' * d + norm(Cons, 1)
end

