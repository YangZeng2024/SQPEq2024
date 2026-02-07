# SQPEq2024
This code is for the algorithm in the paper [arXiv:2503.09490](https://arxiv.org/abs/2503.09490).
1. This code is to test the performance of the algorithm on CUTEst and LIBSVM problems. For CUTest, run "main.jl" to use this code. For example, run SQPEq("BT10", 1, 14) directly in the terminal.
2. If you would like to test your own problem, modify "estimator.jl" and relevant codes in "main.jl" and parameters related to the estimator in "ProblemSetup.jl". 
3. If you would like to modify other input parameters, check "ProblemSetup.jl".
4. For LIBSVM problem, directly run "LIBSVM.jl". For example, run SQPEq("a9a.txt", 1e-4, 41, 500) directly in the terminal.
  
