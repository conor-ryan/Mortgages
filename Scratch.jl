using Random
using LinearAlgebra
using Statistics
using Optim
using Distributions
using Plots
include("halton.jl")
include("types.jl")
include("functions.jl")
include("estimate.jl")
include("illustration.jl")

draw = Draws(100,100)

balance_sheet_penalty = 0.8
m_r = 0.02
m_s = 0.02
ffr_funding_slope = 0.0
# λ = 0.01
# ffr = 0.01

ffr_moment_vector =       [0.0025, 0.02, 0.04,    0.06]
ffr_rate_spread_moments = [0.0020, 0.00, -0.0040, -0.0070]
ffr_sec_moments          = [0.72,  0.68, 0.60,    0.5]
search_moments          = [0.502309996, 0.360460984, 0.116616744, 0.013606133, 0.007006143]
search_discount_moments = [-0.00042,     -0.00095,  -0.0013, -0.0016]

μ_ϵ = -0.25
σ_ϵ = 0.1
μ_ν =0.0
σ_ν = 0.1
μ_η = -4.5
σ_η = 1.0
# par_vec = [balance_sheet_penalty,m_r,m_s,μ_ϵ,σ_ϵ,μ_ν,σ_ν,μ_η,σ_η]
# par_vec_orig = [μ_ϵ,σ_ϵ,μ_ν,σ_ν,μ_η,σ_η,ffr_funding_slope,balance_sheet_penalty]

# temp_estimate = [  0.06759753058909282,
# 0.32655036857499115,
# -0.2629563808463488,
# 0.15301660847640602,
# -4.377577720914874,
# 1.1755984866713165,
# 4.6738604791134275,
# 0.73]
draw = Draws(100,100)

temp_estimate_adj = [0.25,
0.05,
-0.1,
0.2,
-3.4,
0.7,
5,
1.0,
0.018,
1,
0.02,
1]

f = range(0.005,0.06,step=0.005)
spread = rate_spread.(Ref(temp_estimate_adj),f,Ref(draw))
spread_mat = [tup[k] for tup in spread,k in 1:length(spread[1])]


r = range(0.01,0.15,step=0.005)
mon_prof = mechanism_mon_profit.(Ref(temp_estimate_adj),Ref(0.01),r,Ref(draw))
mon_prof = [tup[k] for tup in mon_prof,k in 1:length(mon_prof[1])]
plot(r,mon_prof)



bids = mechanism_bid.(Ref(temp_estimate_adj),f,Ref(draw))
bid_mat = [tup[k] for tup in bids,k in 1:length(bids[1])]

plot(f,bid_mat)


r = range(0.00025,0.15,length=50)
f1 = 0.05
prof1 = mechanism_profit.(Ref(temp_estimate_adj),Ref(f1),r,Ref(draw))
prof1 = [tup[k] for tup in prof1,k in 1:length(prof1[1])]
f2 = 0.01


prof2 = mechanism_profit.(Ref(temp_estimate_adj),Ref(f2),r,Ref(draw))
prof2 = [tup[k] for tup in prof2,k in 1:length(prof2[1])]

plot(r,prof1)
plot(r,prof2)



no_otd = counterfactual.(Ref(temp_estimate_adj),f,Ref(draw))
no_otd = [tup[k] for tup in no_otd,k in 1:length(no_otd[1])]

plot(f,[no_otd[:,1],no_otd[:,2]])

bids = mechanism_bid.(Ref(temp_estimate_adj),f,Ref(draw))
bid_mat = [tup[k] for tup in bids,k in 1:length(bids[1])]


f1 = 0.05
r = range(0.00025,0.08,length=50)

profit = mechanism_profit.(Ref(temp_estimate_adj),Ref(f1),r,Ref(draw))

f2 = 0.01
r = range(0.00025,0.15,length=50)

profit = mechanism_profit.(Ref(temp_estimate_adj),Ref(f2),r,Ref(draw))


# res = toy_estimate(par_vec_orig,draw)






temp_estimate = [  0.18890551380463114,
0.18545542897906309,
-0.25882186516000094,
3.585833966638673e-8,
-7.493866503529498,
0.7485978238959378,
8.362689857946513]

f = range(0.00025,0.06,length=50)

spread = rate_spread.(Ref(temp_estimate),f,Ref(draw))
spread_mat = [tup[k] for tup in spread,k in 1:length(spread[1])]

plot(f,spread_mat[:,1])
plot(f,spread_mat[:,2])
plot(f,spread_mat[:,3])
plot(f,spread_mat[:,4]) # Sold
plot(f,spread_mat[:,5]) # Hold
# for i in 1:10
#     draw = Draws(100,100)
#     obj = toy_est_obj(par_vec_orig,draw)
#     println(obj)
# end


p = Primitives(0.01,draw)
θ = Parameters(temp_estimate_adj,p)

res = model_outcomes(θ,p)
p_avg = sum(res.price.*res.search)/sum(res.search)
s_avg = mean(res.search,dims=1)


res_no_otd = model_outcomes(θ,p,allow_otd=false)
p_avg_no_otd = sum(res_no_otd.price.*res_no_otd.search)/sum(res_no_otd.search)
s_avg_no_otd = mean(res_no_otd.search,dims=1)




reduced_search_cost = [  0.06759753058909282,
0.32655036857499115,
-0.2629563808463488,
0.15301660847640602,
-4.377577720914874*(1.1),
1.1755984866713165,
4.6738604791134275]

p = Primitives(0.0025,draw)
θ = Parameters(reduced_search_cost,p)

res2 = model_outcomes(θ,p)
p2_avg = sum(res2.price.*res2.search)/sum(res2.search)
s2_avg = mean(res2.search,dims=1)


res2_no_otd = model_outcomes(θ,p,allow_otd=false)
p2_avg_no_otd = sum(res2_no_otd.price.*res2_no_otd.search)/sum(res2_no_otd.search)
s2_avg_no_otd = mean(res2_no_otd.search,dims=1)




res = model_outcomes(θ,p)


avg_frac_sold = sum(target.search.*target.sold,dims=2)
avg_price_sold = sum(target.price.*target.sold.*target.search,dims=2)./avg_frac_sold

p_s = sum(avg_price_sold[avg_frac_sold.>0].*avg_frac_sold[avg_frac_sold.>0])/sum(avg_frac_sold)

p_s = sum(target.price.*target.sold.*target.search)/sum(target.search.*target.sold)
p_h = sum(target.price.*(1 .- target.sold).*target.search)/sum(target.search.*(1 .- target.sold))


r = range(0.00025,0.15,length=50)

prob = 1 .-cdf.(Ref(θ.η_dist),r)

dProb = -pdf.(Ref(θ.η_dist),r)

plot(r,prob)
plot(r,dProb)

hazard = prob./dProb
plot(r,hazard)

target = model_outcomes(θ,p)



p = Primitives(0.06,draw)
r = range(0.00025,0.15,length=50)
y1 = dπdr_sell.(r,Ref(p))

y2 = dπdr_hold.(r,Ref(p))

# balance_sheet_penalty = 0.5
# m_r = 0.02
# m_s = 0.02
# μ_ϵ = 0.0
# σ_ϵ = 0.1
# μ_ν = -0.25
# σ_ν = 0.1
# μ_η = -4.5
# σ_η = 1.0

N = 20
mc_res = Matrix{Float64}(undef,length(par_vec_orig),N)

for n in 1:N
    println("Monte Carlo Iteration $n")
    par_vec = par_vec_orig .+ rand(length(par_vec_orig)) .- 0.5
    println("Starting Parameters: $par_vec")
    res = estimate(par_vec,target,p)
    println("Estimated Parameters: $(res.minimizer)")
    mc_res[:,n] = res.minimizer
end



θhat = Parameters(par_vec,p)



x_vec = range(-0.5,1.0,length=50)
obj_val = Vector{Float64}(undef,length(x_vec))

par_vec = par_vec_orig[:]
for i in  eachindex(x_vec)
    par_vec[1] = x_vec[i]
    obj_val[i] = objective(par_vec,target,p)
end

plot(x_vec,obj_val)

objective(par_vec,res,p)



# r = range(0.0,0.12,length=20)
# p = Primitives(θ)
# price = P.(r,Ref(p))



# plot(r,price)


# price_2 = P_inv.(r,ffr)

# i = 1
# monopoly_expected_profit_hold

σ = 4
n = 2
γ = log(2*exp(0))
W  = σ*(log(n*exp(0.5/σ)) - γ)

p = Primitives(0.01,draw)
θ = Parameters(temp_estimate_adj,p)

res = model_outcomes(θ,p)