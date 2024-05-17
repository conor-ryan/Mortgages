using Random
using LinearAlgebra
using Statistics
using Optim
using Distributions
using Plots
include("halton.jl")
include("types_logit.jl")
include("functions_logit.jl")
include("equilibrium_logit.jl")
include("illustration_logit.jl")
# include("illustration.jl")

draw = Draws(25,100)
p = Primitives(0.01,draw)

### Low Rate Environment
pars = [1,1,1,1, # deltas
    -130, # alpha
    100, # alpha variance
    0, # alpha correlation with lambda
    .02, # discount factor parameters
    0,
    .0201,
    0,
    0.0,0.0,0.0,0.0, # Lending Cost
    0.05,0.05,0.05,0.05, # Balance Sheet Cost
    0.005, # Balance Sheet Shock Variance
    ]


θ = Parameters(pars,p)

r_eq = solve_eq(θ,p)

spread, sold = orig_spread(r_eq,θ,p)

r,sell,q, λ, a = outcomes_plot(r_eq,θ,p)
plot(sell,r,seriestype=:scatter,markersize=q*50)

plot(λ,sell,seriestype=:scatter,markersize=q*50)
plot(λ,r,seriestype=:scatter,markersize=q*50)

plot(a,sell,seriestype=:scatter,markersize=q*50)
plot(a,r,seriestype=:scatter,markersize=q*50)


### High Rate Environment
pars = [1,1,1,1, # deltas
    -130, # alpha
    100, # alpha variance
    0, # alpha correlation with lambda
    .07, # discount factor parameters
    0,
    .06,
    0,
    0.0,0.0,0.0,0.0, # Lending Cost
    -0.2,-0.2,-0.2,-0.2, # Balance Sheet Cost
    0.005, # Balance Sheet Shock Variance
    ]


θ = Parameters(pars,p)

r_eq = solve_eq(θ,p)

spread, sold = orig_spread(r_eq,θ,p)

r,sell,q, λ, a = outcomes_plot(r_eq,θ,p)
plot(sell,r,seriestype=:scatter,markersize=q*50)

plot(λ,sell,seriestype=:scatter,markersize=q*50)
plot(λ,r,seriestype=:scatter,markersize=q*50)

plot(a,sell,seriestype=:scatter,markersize=q*50)
plot(a,r,seriestype=:scatter,markersize=q*50)


#### Data Size ####
M = 1 #Markets
T = 10 # Time Periods
J = 5 # Lenders
N = 200 # Consumers per market


#### Model Parameters
δ = repeat([1.0],J) # Lender Demand Parameters

m0_r = 0.02 # Initial balance-sheet discount Rate
m1_r = (0.07-0.02)/0.05 # balance-sheet discount slope

ω = repeat([0.0],J) # Lender Specific Origination Cost
ϵ0 = 0.05 # Balance-sheet cost intercept
ϵ1 = (-0.2 - 0.05)/0.05 # Balance-sheet FFR slope
λ = 0.2 # Cost of Consumer Credit Risk
σ = 0.005 # Balance-sheet variance

## Parameters Uncovered Not Estimated 
α_mean = -80 # Mean price sensitivity
α_var = 100 # Price sensitivity dispersion

z_mean = 0.0 # Standard Normal Consumer Characteristics
z_std = 1.0 

m0_mbs = 0.02 # MBS discount Rate
m1_mbs = (0.07-0.02)/0.05 # MBS discount slope

x = vcat(δ,ω,ϵ0,ϵ1,λ,σ,m0_r,m1_r)




# r = [0.03,0.03,0.03,0.03]
# elas= elasticities(r,θ)



# r_mat = [tup[k] for tup in r_eq,k in 1:length(r_eq[1])]

# hold, sell = balance_sheet_alloc(r_eq,θ,p)

# expected_foc(r,θ,p)


# orig_spread(θ,p)


# r = Vector{Float64}(range(0.0,0.2,length=100))

# p_test = plot_profit(r,r_eq,1,θ,p)

# plot(r,p_test)

# x = randn(10)
# y = randn(10)
# z = randn(10)

# plot(x,y,seriestype=:scatter,markersize=z)
