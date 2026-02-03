# SQPEq2024
This code is for the algorithm in the paper Sequential Quadratic Optimization for Solving Expectation Equality Constrained Stochastic Optimization Problems.
1. This code is to test the performance of the algorithm on CUTEst or LIBSVM problem. For CUTest problem, run "main.jl" to use this code. For example, SQPEq("BT10", 1, 14).
3. If you would like to test your own problem, modify "estimator.jl" and relevant codes in "main.jl" and parameters related to the estimator in "ProblemSetup.jl". 
4. If you would like to modify other parameters such as beta, check "ProblemSetup.jl".
5. For LIBSVM problem, directly run "LIBSVM.jl". For example,  or SQPEq("a9a.txt", 1e-4, 41, 500).
6. Some names of variables may be "strange". The output file will print the exact name of them. I just list them here in case you would like to modify the code.
Names in the code     Actual name
KKT(KKT_error)        stationarity error
tc(TC)                KKT Error
ksi                   xi    
