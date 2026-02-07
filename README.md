SQPEq2024

This repository contains the implementation of the algorithm proposed in the paper:
[arXiv:2503.09490](https://arxiv.org/abs/2503.09490)

Overview
This code is used to evaluate the performance of the proposed algorithm on CUTEst and LIBSVM problems.

Usage

Testing on CUTEst problems
To run experiments on CUTEst problems, execute “main.jl”.
For example, run the following command directly in the terminal:
 SQPEq("BT10", 1, 14).

Testing on your own problems
If you would like to test your own problem instances, please modify:
 ”estimator.jl“,
 the relevant code in ”main.jl“,
 the estimator-related parameters in ”ProblemSetup.jl“.

Modifying algorithm parameters
Other input parameters and algorithm settings can be configured in:
 "ProblemSetup.jl".

Testing on LIBSVM problems
To run experiments on LIBSVM datasets, execute "LIBSVM.jl".
For example:
 SQPEq("a9a.txt", 1e-4, 41, 500).

Notes
Please make sure that the required dependencies for CUTEst and LIBSVM are properly installed and configured before running the experiments.
The numerical results reported in the paper are generated based on this implementation.
  
