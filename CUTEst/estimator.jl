function F(Nlp, x)

    objective_scaled_factor = 100/max(100, norm(grad(Nlp, Nlp.meta.x0), Inf))
    return obj(Nlp, x) * objective_scaled_factor

end

function g(Nlp, x) # calculate g_scale

    objective_scaled_factor = 100/max(100, norm(grad(Nlp, Nlp.meta.x0), Inf))
    return grad(Nlp, x) * objective_scaled_factor
    
end

function C(Nlp, x, m) #calculate c_scale

    constraint_ci_scaled_factor = zeros(m)
    constraint = zeros(m)
    for i in 1:m
        constraint_ci_scaled_factor[i] = 100/max(100, norm(jac(Nlp, Nlp.meta.x0)[i, :], Inf))
    end
    for i in 1:m
        constraint[i] = constraint_ci_scaled_factor[i] * cons(Nlp, x)[i]
    end
    return constraint
       
end

function J(Nlp, x, m, n) #calculate J_scale

    constraint_ci_scaled_factor = zeros(m)
    jacobian = zeros(m, n)
    for i in 1:m
        constraint_ci_scaled_factor[i] = 100/max(100, norm(jac(Nlp, Nlp.meta.x0)[i, :], Inf))
    end
    for i in 1:m
        jacobian[i, :] = constraint_ci_scaled_factor[i] * jac(Nlp, x)[i, :]
    end
    return Matrix(jacobian)

end

function g_rv(Grad_deterministic, n, epsilon_g, beta)

    mu = Grad_deterministic
    Sigma = Matrix(I,n,n) * (epsilon_g * beta^2 / n)
    d = MvNormal(mu, Sigma)
    return rand(d)

end

function c_rv(Cons_deterministic, m, epsilon_c, beta)
    
    mu = Cons_deterministic
    Sigma = Matrix(I, m, m) * (epsilon_c * beta^2 / m)
    d = MvNormal(mu, Sigma)
    return rand(d)

end

function J_rv(Jac_deterministic, m, n, epsilon_J, beta)
    
    Jr = zeros(m, n)
    for i in 1: m
        mu = Jac_deterministic[i, :]
        Sigma = Matrix(I, n, n) * (epsilon_J * beta^2 / (n * m))
        d = MvNormal(mu, Sigma)
        Jr[i, :] = rand(d)
    end
    return Jr

end
