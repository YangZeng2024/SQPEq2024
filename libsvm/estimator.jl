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