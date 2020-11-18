using Distributions
using Plots, StatsPlots
using CUDA
using Random

Random.seed!(5236)

"""
    mh_normal(x,μ,σ)
To evaluate the probability density function of observation with 
mean mean `μ` and standard deviation `σ`
"""
function mh_normal(x,μ,σ)
    numerator = exp((-(x-μ)^2)/(2σ^2))
    denominator = σ*sqrt(2π)
    return numerator/denominator
end

"""
    random_coin(p)
function for decision making based on value `p`
"""
function random_coin(p)
    unif = rand(Uniform(0,1))
    if unif >= p
        return false
    else
        return true
    end
end

"""
    gaussian_mcmc(itr,μ,σ)
Samples and executes Metropolis-Hastings algorithm for `itr`
itr number of samples
"""
function gaussian_mcmc(itr,μ,σ)
    states =  Vector{Float64}()
    burn_in = Int(itr*0.2)
    current = rand(Uniform(-5.0,5.0))
    for i in 1:itr
    	push!(states, current)
        movement = rand(Uniform(-20*σ+μ,20*σ+μ))
        
        curr_prob = mh_normal(current,μ,σ)
        move_prob = mh_normal(movement,μ,σ)
        
        acceptance = min(move_prob/curr_prob,1)
        if random_coin(acceptance)
            current = movement
        end
    end
    return states[burn_in:end]
end

"""
    sampling_kernel(cu_samples, cu_status, cu_accept)
CUDA kernel for Metropolis-Hastings algorithm to decide 
regarding acceptance or rejection of samples. 
"""
function sampling_kernel(cu_samples, cu_status, cu_accept)
	idx = threadIdx().x
	current = 0.0
	movement = cu_samples[idx]
	curr_prob = mh_normal(current,0.0,1.0)
    move_prob = mh_normal(movement,0.0,1.0)
    ratio = move_prob/curr_prob
    acceptance = min(move_prob/curr_prob,1.0)
    if cu_accept[idx] < acceptance
        cu_status[idx] = true
    end
    return nothing
end
"""
    cu_gaussian_mcmc(itr,μ,σ)
Samples and executes Metropolis-Hastings algorithm for `itr`
itr number of samples with CUDA implemetation
"""
function cu_gaussian_mcmc(itr,μ,σ) 
    samples = rand(Uniform(-5*σ+μ,5*σ+μ), itr) 
    status = fill(false,itr)
    accept = rand(Uniform(0.0,1.0), itr)
    cu_accept = CuArray(accept)
    cu_samples = CuArray(samples)
    cu_status = CuArray(status)
    @cuda threads=itr sampling_kernel(cu_samples, cu_status, cu_accept)
    out_status = Array(cu_status)
    loc = findall(out_status)
    return samples[loc]
end

N = 1000
# lines = range(-3, 3, length=N)
# normal_curve = [mh_normal(l,0.0,1.0) for l in lines]
μ = 0.0
σ = 1.0
"""
CPU sampling using Metropolis-Hastings algorithm
"""
dist = gaussian_mcmc(N,μ,σ)
"""
GPU sampling using Metropolis-Hastings algorithm
"""
distcu = cu_gaussian_mcmc(N,μ,σ)

"""
To plot samples generated with and without CUDA
"""
l = @layout[a; b]
p1 = histogram(dist,bins=30, normed=true)
p2 = histogram(distcu,bins=30, normed=true)
plot(p1, p2, layout = l)


# histogram(dist,bins=20, normed=true) 
# plot!(lines, normal_curve)

# histogram(distcu,bins=20, normed=true) 
# plot!(lines, normal_curve)


