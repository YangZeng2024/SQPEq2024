using LinearAlgebra
using NLPModels
using CUTEst
using Distributions
using SparseArrays
using Random

function load_libsvm_data_as_matrix(file_path)
    X = []
    y = []
    max_feature_index = 0  

    open(file_path, "r") do f
        for line in eachline(f)
            tokens = split(line)
            push!(y, parse(Float64, tokens[1]))
            features = Dict{Int, Float64}()
            for token in tokens[2:end]
                idx_value = split(token, ":")
                feature_index = parse(Int, idx_value[1])
                features[feature_index] = parse(Float64, idx_value[2])
                if feature_index > max_feature_index
                    max_feature_index = feature_index
                end

            end
            push!(X, features)
        end
    end

    num_samples = length(X)
    num_features = max_feature_index
    X_matrix = zeros(num_samples, num_features)

    for i in 1:num_samples
        for (feature_index, value) in X[i]
            X_matrix[i, feature_index] = value
        end
    end

    return X_matrix, y
end

function generate_x0(n)
    # Generate a random n-dimensional vector with entries from a normal distribution
    vec = randn(n)
    
    # Normalize the vector to have 2-norm 1, then scale to the desired norm
    vec *= (0.1 / norm(vec))
    
    return vec
end

function cal_f(batch_1, X, Y, x, random_vector_1)
    f_value = 0
    for i in random_vector_1
        f_value += log(1 + exp(-Y[i] * (X[i, :]' * x)))
    end
    return f_value/batch_1
end

function cal_f_true(X, Y, x, N)
    f_value = 0
    for i in 1: N
        f_value += log(1 + exp(-Y[i] * (X[i, :]' * x)))
    end
    return f_value/N
end

function cal_g(batch_1, X, Y, x, random_vector_1, n)
    g_value = zeros(n)
    for j in 1:n
        s = 0.0
        for i in random_vector_1
            s += (-Y[i] * X[i, j]) / (1 + exp(Y[i] * (X[i, :]' * x)))
        end
        g_value[j] = s / batch_1   
    end
    return g_value
end


function cal_g_true(X, Y, x, N, n)
    g_value = zeros(n)
    for j in 1:n
        s = 0.0
        for i in 1:N
            s += (-Y[i] * X[i, j]) / (1 + exp(Y[i] * (X[i, :]' * x)))
        end
        g_value[j] = s / N         
    end
    return g_value
end


function gen_Con(n, epsilon_J, batch_2)
    matrix_original = zeros(10, n)
    mu_original = 1
    Sigma_original = 100
    b1_original = zeros(10)
    for i in 1: 10
        for j in 1: n
            matrix_original[i, j] = rand(Normal(mu_original, Sigma_original))
        end
    end
    for i in 1: 10
        b1_original[i] = rand(Normal(mu_original, Sigma_original))
    end
    matrices = zeros(10, n, 1000)
    mu = zeros(n)
    Sigma = Matrix(I, n, n) * (epsilon_J / sqrt(batch_2 * 10 * n))
    for i in 1: 10
        for j in 1: 1000
            dist = MvNormal(mu, Sigma)
            random_samples = rand(dist, 1000)
            matrices[i, :, j] = random_samples[:, j] + matrix_original[i, :]
        end    
    end
    mu_b = 0
    Sigma_b = epsilon_J^2 / sqrt(batch_2 * 10)
    b1s = zeros(10, 1, 1000)
    for i in 1: 10
        for j in 1: 1000 
            dist_b = Normal(mu_b, Sigma_b)
            random_samples_b = rand(dist_b, 1000)
            b1s[i, 1, j] = random_samples_b[j] + b1_original[i]
        end
    end
    return matrices, b1s
end

function cal_c(matrices, b1s, x, batch_2, random_vector_2)
    c_value_1 = zeros(10)
    for j in 1:10
        s = 0.0
        for i in random_vector_2
            s += matrices[j, :, i]' * x - b1s[j, 1, i]
        end
        c_value_1[j] = s / batch_2           
    end
    c_value_2 = norm(x)^2 - 1
    c_value = vcat(c_value_1, c_value_2)
    return c_value
end

function cal_c_true(matrices, b1s, x)
    c_value_1 = zeros(10)
    N = size(matrices, 3)                   
    for j in 1:10
        s = 0.0
        for i in 1:N
            s += matrices[j, :, i]' * x - b1s[j, 1, i]
        end
        c_value_1[j] = s / N                
    end
    c_value_2 = norm(x)^2 - 1
    c_value = vcat(c_value_1, c_value_2)
    return c_value
end


function cal_J(matrices, x, batch_2, random_vector_2, n)
    J_value = zeros(11, n)

    for j in 1:10
        for z in 1:n
            s = 0.0
            for i in random_vector_2
                s += matrices[j, z, i]
            end
            J_value[j, z] = s / batch_2     
        end
    end

    J_value[11, :] .= 2 .* x
    return J_value
end

function cal_J_true(matrices, x, n)
    J_value = zeros(11, n)
    N = size(matrices, 3)                  
    for j in 1:10
        for z in 1:n
            s = 0.0
            for i in 1:N
                s += matrices[j, z, i]
            end
            J_value[j, z] = s / N         
        end
    end
    J_value[11, :] .= 2 .* x
    return J_value
end


function cal_d_and_y(H, Jac, Grad, Cons)   
    A = Matrix([
        H Jac';
        Jac zeros(11, 11)
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
    if tau_pre <= (1 - epsilon_tau) * tau_trial
        return tau_pre
    else
        return (1 - epsilon_tau) * min(tau_trial, tau_pre)   
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
    alpha_min = 2 * (1 - eta) * beta * ksi * tau/(tau * L + Gamma) #计算alpha_min
    alpha_k = alpha_min
    while psi(1.1 * alpha_k, eta, beta, tau, Grad, d_k, Cons, L, Gamma) < 0 && 1.1 * alpha_k < alpha_min + theta * beta
        alpha_k = 1.1 * alpha_k
    end
    return alpha_k  
end

function SQPEq(file_path, epsilon_J, Seed, L_)

    Random.seed!(Seed)
    X_matrix, y = load_libsvm_data_as_matrix(file_path)
    N = length(y)
    n = length(X_matrix[1, :])
    batch_1 = 16
    batch_2 = 16
    matrices, b1s = gen_Con(n, epsilon_J, batch_2)
    println("Generation Constraints Finished.")
    K = 10000
    Gamma = 2
    x1 = generate_x0(n)
    L = L_
    x = [x1]
    tau_0 = 0.1
    ksi_0 = 1
    H = Matrix(I, n, n)
    beta = 1
    eta = 0.5
    sigma = 0.1
    epsilon_tau = 1e-2
    epsilon_ksi = 1e-2
    theta = 10

    tau = []
    ksi = []
    alpha = []

    start_time = time()

    for k in 1: K

        #println("The $(k) step start.")

        random_vector_1 = rand(1: N, batch_1)
        random_vector_2 = rand(1: 1000, batch_2)

        Grad = cal_g(batch_1, X_matrix, y, x[end], random_vector_1, n)
        Cons = cal_c(matrices, b1s, x[end], batch_2, random_vector_2)
        Jac = cal_J(matrices, x[end], batch_2, random_vector_2, n)

       
        if minimum(svdvals(Jac * Jac')) <= 1e-12
            println("The minimum singular value is $(svdvals(Jac * Jac')).")
            return "This problem does not satisfy the LICQ condition at the $k step."

        end
   
        

        solution = cal_d_and_y(H, Jac, Grad, Cons)
        d_k = solution[1: n]
        y_k = solution[n+1: n+11]

        if norm(d_k, Inf) > 1e-8

            if k == 1

                tau_k = cal_tau(H, d_k, sigma, tau_0, epsilon_tau, Grad, Cons)
                push!(tau, tau_k)
                
            else

                tau_k = cal_tau(H, d_k, sigma, tau[end], epsilon_tau, Grad, Cons)
                push!(tau, tau_k)

            end

            if k == 1

                ksi_k = cal_ksi(d_k, tau[end], ksi_0, epsilon_ksi, Grad, Cons)
                push!(ksi, ksi_k)

            else
                
                ksi_k = cal_ksi(d_k, tau[end], ksi[end], epsilon_ksi, Grad, Cons)
                push!(ksi, ksi_k)

            end

            alpha_nn = cal_alpha(d_k, eta, beta, ksi_k, tau_k, L, Gamma, theta, Grad, Cons)
            push!(alpha, alpha_nn)
        
        else
            if k == 1

                push!(tau, tau_0)
                push!(ksi, ksi_0)

            else
                
                push!(tau, tau[k-1])
                push!(ksi, ksi[k-1])

            end

            alpha_nn = 2 * (1 - eta) * beta * ksi[end] * tau[end]/(tau[end] * L + Gamma)
            push!(alpha, alpha_nn)

        end

        push!(x, x[end] + alpha_nn * d_k)

        
    end
    
    elapsed_time = time() - start_time

    f = []
    g_ = []
    c = []
    KKT = []
    TC = []
    phi = []
    d = []

    for i in 1: 10000

        #println("The $(i) step calculation start.")

        Obj = cal_f_true(X_matrix, y, x[i], N)
        Grad = cal_g_true(X_matrix, y, x[i], N, n)
        Cons = cal_c_true(matrices, b1s, x[i])
        Jac = cal_J_true(matrices, x[i], n)
        d_k = cal_d_and_y(H, Jac, Grad, Cons)[1: n]

        y_kkt = -inv(Jac * Jac') * Jac * Grad
        KKT_error = norm(Grad + Jac' * y_kkt, Inf)
        tc = norm(vcat(Grad + Jac' * y_kkt, Cons), Inf)

        push!(f, Obj)
        push!(g_, norm(Grad, Inf))
        push!(c, norm(Cons,Inf))
        push!(KKT, KKT_error)
        push!(TC, tc)
        push!(phi, tau[i] * Obj + c[i])
        push!(d, norm(d_k, Inf))

       
    end

    c_min = minimum(c)
    kkt_initial = 100
    KKT_min = 0
    Con_min = 0
    optimal_index = 1

    if c_min <= 1e-4
        
        for i in 1: 10000

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

        for i in 1: 10000

            if c[i] == c_min

                KKT_min = KKT[i]
                Con_min = c[i]
                optimal_index = i
      
            end
            
        end

    end
    return KKT_min, Con_min, elapsed_time
 
end



