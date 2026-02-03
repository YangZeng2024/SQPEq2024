using LinearAlgebra
using NLPModels

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

Base.@kwdef struct ProblemSetup
    K::Int = 10000
    Seed::Int = 41
    ProblemName::String
    L::Float64
    Gamma::Float64 = 2
    batch_1::Int = 16
    batch_2::Int = 16


    X_matrix::Matrix{Float64} = load_libsvm_data_as_matrix(file_path)[1]
    y:: Vector{Float64} = load_libsvm_data_as_matrix(file_path)[2]
    N:: Int = length(y)
    n::Int = length(X_matrix[1, :])
    x_1::Vector{Float64} = generate_x0(n)
    H::Matrix{Float64} = Matrix(I, n, n)
    random_vector_1::Vector{Float64} = rand(1: N, batch_1)
    random_vector_2::Vector{Float64} = rand(1: 1000, batch_2)

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
