using Random
using LinearAlgebra
using Statistics
using Optim
using Distributions
using Plots
using CSV
using DataFrames
include("halton.jl")
include("types_logit.jl")
include("functions_logit.jl")
include("equilibrium_logit.jl")
include("illustration_logit.jl")


Random.seed!(222024238)

#### Data Size ####
M = 1 #Markets
J = 5# Lenders
N = 1000 # Consumers per market


#### Model Parameters

#### Discount Data
ffr = [0.035,0.02]
T = length(ffr)

D_data = Matrix{Float64}(undef,T,T)
D_data[:].= 0.0
for t in 1:T
    D_data[t,t] = 1.0
end

# Discount Parameters
β_d = [0.5,0.7]

#### Lender Demand Data 
X = Matrix{Float64}(undef,J,J)
X[:].=0.0
for j in 1:J
    X[j,j] = 1.0
end

# Lender Demand Parameters (J)
β_x = [12.9,12.6, 12.4, 12.2,12.0]

#### Cost Data
cost_bounds = [[0,5],[1,100],[0,10]]
W = Matrix{Float64}(undef,J,length(cost_bounds))
for i in 1:size(W,1)
    for j in 1:size(W,2)
        W[i,j] = cost_bounds[j][1] + rand()*(cost_bounds[j][2] - cost_bounds[j][1])
    end
end

# Cost Parameters
ϵ1 = (-0.1 - 0.05)/0.05 # -3.0
# γ_WH = [1.0, 1.0, 1.1, 1.1, 0.9, -3.0]
# γ_WS = [-0.05, -0.0,-0.1,-0.1,-0.2, 3.0]
γ_WH = [0.1, 0.1, 0.2, 0.2, 0.9, -3.0]
γ_WS = [0.15, 0.2,0.1,0.1,-0.3, 3.0]

#### Consumer Data
cons_bounds = [[1,1],[620,850],[0,500],[30,95],[0,20],[0,1],[40,600]]
Z = Matrix{Float64}(undef,M*T*N,length(bounds))
for i in 1:size(Z,1)
    for j in 1:size(Z,2)
        Z[i,j] = cons_bounds[j][1] + rand()*(cons_bounds[j][2] - cons_bounds[j][1])
    end
end

γ_ZH = [0.0,0.0,0.0,0.04,0,-0.05,-0.01]
γ_ZS = [-0.29,0.0,0.0,-0.04,0,0.05,0.01]


### Extra Parameters
σ = 0.05 # Balance-sheet shock variance

##### Parameters Uncovered Not Estimated 
α_min = -60 # Mean price sensitivity
α_max = -600 # Price sensitivity dispersion

m0_mbs = 0.0201 # MBS discount Rate
m1_mbs = (0.06-0.0201)/0.05 # MBS discount slope


θ = Parameters(β_d,β_x,γ_WH,γ_WS,γ_ZH,γ_ZS,σ)


## Market Data 
# Market Index
# Lender Index
# Market Characteristics
# Lender Characteristics

market_data = Matrix{Float64}(undef,J*T*M,2 + size(X,2)+size(W,2))
market_data[:].=0.0

index = 0
m_ind = 0
for m in 1:M
    index_t = 0
    for t in 1:T
        m_ind +=1
        index_t +=1
        index_j = 0 
        for j in 1:J
            index +=1
            index_j +=1
            market_data[index,1] = m_ind
            market_data[index,2] = j
            market_data[index,3:(2+size(X,2))] = X[index_j,:]
            market_data[index,(2+size(X,2)+1):(2+size(X,2)+size(W,2))] = W[index_j,:]
            market_data[index,(2+size(X,2)+size(W,2))] = ffr[index_t]
        end
    end
end


#### MBS Data ####
# Time Index
# Coupon Prices 0.5 - 15, 0.5 increments

coupon_rates = collect(0.5:0.5:15.0)./100
mbs_data = Matrix{Float64}(undef,T,1 + length(coupon_rates))

for t in 1:T
    MBS_par = D_model[t,:]'*[m0_mbs, m1_mbs]
    MBS_par = MBS(MBS_par)
    mbs_data[t,1] = t-1
    mbs_data[t,2:(1+length(coupon_rates))] = P.(coupon_rates,Ref(MBS_par))
end

#### Outside Share Data ####
# Time Index
share_data = Matrix{Float64}(undef,T,5+J)
share_data[:].=0.0
for t in 1:T
    share_data[t,1] = t-1
end



##### Data Structure
## Consumer Choice Data
# Market Index
# Consumer credit risk
# Price Sensitivity
# Chosen Lender Index
# Paid interest Rate
# Loan Sold

consumer_data = Matrix{Float64}(undef,M*T*N,3+size(Z,2)+2*size(D_data,2)+size(D_model,2)+size(W,2)+1+1+1+1+1+1)
consumer_data[:].=0.0



index = 0 
m_ind = 0
final_ind = 0 
for m in 1:M 
    index_t = 0
    for t in 1:T
        m_ind +=1
        index_t+=1
        for n in 1:N
            index +=1 
            consumer_data[index,1] = m_ind
            consumer_data[index,2] = index_t-1
            consumer_data[index,3] = α_min + rand()*(α_max-α_min)
            
            # consumer_data[index,4] = Chosen Lender
            # consumer_data[index,5] = Interest Rate
            # consumer_data[index,6] = Sold or Not
            consumer_data[index,7:(6+size(Z,2))] = Z[index,:]
            last = 6+size(Z,2)
            consumer_data[index,(last+1):(last+size(D_model,2))] = D_model[index_t,:]
            last = last+size(D_model,2)
            consumer_data[index,(last+1):(last+size(D_data,2))] = D_data[index_t,:]
            last = last+size(D_data,2)
            consumer_data[index,(last+1):(last+size(D_data,2))] = D_data[index_t,:]
            final_ind = last+size(D_data,2)
        end
    end
end


demand_ind = Int.(3:(2+size(X,2)))
cost_ind = Int.((2+size(X,2)+1):(2+size(X,2)+size(W,2)))

consumer_ind = Int.(7:(6+size(Z,2)))
last = maximum(consumer_ind)
discount_ind = Int.((last+1):(last+size(D_model,2)))
last = maximum(discount_ind)
discount_data_ind = Int.((last+1):(last+size(D_data,2)))
last = maximum(discount_data_ind)
discount_interaction_ind = Int.((last+1):(last+size(D_data,2)))
final_ind = maximum(discount_interaction_ind)

for i in 1:size(consumer_data,1)
    market = consumer_data[i,1]
    time = Int(consumer_data[i,2])
    market_index = market_data[:,1].==market
    X_i = market_data[market_index,demand_ind]
    W_i = market_data[market_index,cost_ind]
    D_i = consumer_data[i,discount_ind]
    Z_i = consumer_data[i,consumer_ind]
    dat = Data(X_i,W_i,D_i,Z_i)

    MBS_par = D_i'*[m0_mbs, m1_mbs]
    MBS_par = MBS(MBS_par)

    α = consumer_data[i,3]
    r_eq, itr1 = solve_eq_robust(α,dat,θ,MBS_par)

    # if α>(-200)
    #     r_eq, itr1 = solve_eq(α,dat,θ,MBS_par)
    # else
    #     r_eq, itr1 = solve_eq_robust(α,dat,θ,MBS_par)
    # end
    # if (α>(-500)) & (itr1>=249)
    #     r_eq, itr1 = solve_eq_robust(α,dat,θ,MBS_par)
    # end
    shares = market_shares(r_eq,α,dat,θ)
    sell = balance_sheet_alloc(r_eq,α,dat,θ,MBS_par)

    share_draw = rand()#*sum(shares)
    sell_draw = rand()
    j_select = findfirst(share_draw.<cumsum(shares))
    if isnothing(j_select)
        share_data[time+1,3] +=1
        consumer_data[i,4] = -1
        # println("Consumer $i, solve iterations $itr1, inside share: $(sum(shares)) - skip")
        continue
    else
        share_data[time+1,2] +=1
        share_data[time+1,5+j_select] +=1
    end 
    # share_data[time+1,3] +=1*(1 - sum(shares))
    # share_data[time+1,2] +=1*(sum(shares))

    consumer_data[i,4] = j_select-1
    consumer_data[i,5] = r_eq[j_select]
    consumer_data[i,6] = 1.0*(sell_draw.<sell[j_select])
    consumer_data[i,discount_interaction_ind] = consumer_data[i,discount_interaction_ind].*r_eq[j_select]
    consumer_data[i,final_ind + 1] = P(r_eq[j_select] - 0.0025,MBS_par)
    consumer_data[i,(final_ind + 2):(final_ind + 1 + size(W,2))] = W_i[j_select,:]
    consumer_data[i,final_ind + 1 + size(W,2)+1] = 1 - sum(shares)
    consumer_data[i,final_ind + 1 + size(W,2)+2] = i
    # α_est, r_eq, itr2 = solve_eq_r(r_eq[j_select],j_select,dat,θ,MBS_par)
    # consumer_data[i,7] = α_est
    # println("Consumer $i, solve iterations $itr1, inside share: $(sum(shares))")
end

consumer_data = consumer_data[consumer_data[:,4].>=0,:]
println(size(consumer_data)[1])
println(mean(consumer_data[:,6]))

share_data[:,5] = (share_data[:,2] .+ share_data[:,3])
share_data[:,4] = share_data[:,3]./share_data[:,5]


file = "consumer_data.csv"
CSV.write(file,DataFrame(consumer_data,:auto))

file = "market_data.csv"
CSV.write(file,DataFrame(market_data,:auto))

file = "mbs_data.csv"
CSV.write(file,DataFrame(mbs_data,:auto))

file = "share_data.csv"
CSV.write(file,DataFrame(share_data,:auto))

