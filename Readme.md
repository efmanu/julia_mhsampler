#CPU and GPU implementation of Metropolis-Hastings algorithm using Julia

This project aimed to sample randomly in normal distribution domain. The implementation just focuses GPU and CPU implementation. There is no further optimization is done in GPU.

`dist = gaussian_mcmc(N,μ,σ)` is used for sampling with CPU. Where `N` is the number of iterations, `μ` is the expected mean of sampels and `σ` is the satandard deviation.

`distcu = cu_gaussian_mcmc(N,μ,σ)` is used for sampling with GPU.