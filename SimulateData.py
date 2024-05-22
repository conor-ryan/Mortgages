import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from FirstStageFunctions import *
from ModelTypes import *
from EquilibriumFunctions import *
from ModelFunctions import *
from EstimationFunctions import *
from ParallelFunctions import *
from Derivatives import * 
from NumericalDerivatives import *

random.seed(2220242381)

#### Data Size ####
M = 1 #Markets
J = 5# Lenders
N = 2000 # Consumers per market


#### Model Parameters

#### Discount Data
ffr = [0.024,0.018]
T = len(ffr)

D_data = np.zeros((T,T))
for t in range(T):
    D_data[t,t] = 1.0

# Discount Parameters
β_d = np.array([0.5,0.25])

#### Lender Demand Data 
X = np.zeros((J,J))

for j in range(J):
    X[j,j] = 1.0


# Lender Demand Parameters (J)
β_x = np.array([12.3,12.1, 11.9, 11.7,11.5,5.0])

#### Cost Data
#cost_bounds = [[0,5],[1,100],[0,10]]
cost_moments= [[1.9,0.7],[14,3],[-0.4,0.95]]
# cost_moments= [[1.48,0],[13,0],[0.3,0]]
W = np.zeros((J,len(cost_moments)))
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        # W[i,j] = cost_bounds[j][0] + random.random()*(cost_bounds[j][1] - cost_bounds[j][0])
        W[i,j] =random.gauss(cost_moments[j][0],cost_moments[j][1])


# Cost Parameters
# γ_WH = np.array([-0.01,-0.005,0.002])
γ_WH = np.array([0,0,0.000])
γ_WS = np.array([0.044,0.00094,-0.00514])

#### Consumer Data
# cons_bounds = [[1,1],[620,850],[0,500],[30,95],[0,20],[0,1],[40,600]]
cons_moments = [[1,0],[760,10],[119,30],[81,15],[34,5],[0,1],[294,80]]
# cons_moments = [[1,0],[760,0],[119,0],[75,0],[34,0],[.36,0],[294,0]]
Z = np.zeros((M*T*N,len(cons_moments)))
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        # Z[i,j] = cons_bounds[j][1] + random.random()*(cons_bounds[j][1] - cons_bounds[j][0])
        Z[i,j] =random.gauss(cons_moments[j][0],cons_moments[j][1])
Z[:,5] = 1.0*(Z[:,5]<-0.5)

# γ_ZH = np.array([0.32,-1e-5,-1e-5,1e-5,0.00,0.005,0.00,0.1])
γ_ZH = np.array([0.4,0.0,0,0,0,0.00,0,0.00])
γ_ZS = np.array([-0.299,1.1e-4,3e-5,-7.2e-4,0,6.55e-3,7e-5,-0.161])


### Extra Parameters
σ = 0.04 # Balance-sheet shock variance

##### Parameters Uncovered Not Estimated 
α_min = -50 # Mean price sensitivity
α_max = -200 # Price sensitivity dispersion

# Log normal distribution
α_mean = 0 # Mean price sensitivity
α_var = 0.5 # Price sensitivity dispersion
α_scale = -100


theta = Par_sim(β_x,γ_WH,γ_ZH,β_d,σ,γ_WS,γ_ZS)

## Market Data 
# Market Index
# Lender Index
# Market Characteristics
# Lender Characteristics

market_data = np.zeros((J*T*M,3 + X.shape[1]+W.shape[1]))

index = -1
m_ind =-1
for m in range(M):
    index_t = -1
    for t in range(T):
        m_ind +=1
        index_t +=1
        index_j = -1 
        for j in range(J):
            index +=1
            index_j +=1
            market_data[index,0] = m_ind
            market_data[index,1] = j
            market_data[index,2:(2+X.shape[1])] = X[index_j,:]
            market_data[index,(2+X.shape[1]):(2+X.shape[1]+W.shape[1])] = W[index_j,:]
            market_data[index,(2+X.shape[1]+W.shape[1])] = ffr[index_t]


#### MBS Data ####
# Time Index
# Coupon Prices 0.5 - 15, 0.5 increments

coupon_rates = np.arange(1,9.1,0.25)/100
prices = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9]
prices = [str(p) for p in prices]
mbs_data = pd.read_csv("mbs_data_real_subset.csv") # MBS coupon price data
#### Outside Share Data ####
# Time Index
share_data = np.zeros((T,5+J))
for t in range(T):
    share_data[t,0] = t




##### Data Structure
## Consumer Choice Data
# Market Index
# Consumer credit risk
# Price Sensitivity
# Chosen Lender Index
# Paid interest Rate
# Loan Sold

consumer_data = np.zeros((M*T*N,3+Z.shape[1]+2*D_data.shape[1]+W.shape[1]+1+1+1+1+1+1+1+1))

index = -1
m_ind = -1
final_ind = -1 
for m in range(M): 
    index_t = -1
    for t in range(T):
        m_ind +=1
        index_t+=1
        for n in range(N):
            index +=1 
            consumer_data[index,0] = m_ind
            consumer_data[index,1] = index_t
            # consumer_data[index,2] = α_min + random.random()*(α_max-α_min)
            consumer_data[index,2] = np.exp(random.gauss(α_mean,α_var))*α_scale
            # consumer_data[index,3] = Chosen Lender
            # consumer_data[index,4] = Interest Rate
            # consumer_data[index,5] = Sold or Not
            consumer_data[index,6] = ffr[index_t]
            consumer_data[index,7:(7+Z.shape[1])] = Z[index,:]
            last = 7+Z.shape[1]
            consumer_data[index,(last):(last+D_data.shape[1])] = D_data[index_t,:]
            last = last+D_data.shape[1]
            consumer_data[index,(last):(last+D_data.shape[1])] = D_data[index_t,:]
            final_ind = last+D_data.shape[1]


demand_ind = list(range(2,(2+X.shape[1])))
demand_ind.append(market_data.shape[1]-1)
cost_ind = list(range(2+X.shape[1],(2+X.shape[1]+W.shape[1])))


consumer_ind = list(range(7,(7+Z.shape[1])))
consumer_ind.append(6)
last = np.max(consumer_ind) +1
discount_data_ind = range((last),(last+D_data.shape[1]))
last = np.max(discount_data_ind) +1
discount_interaction_ind = range((last),(last+D_data.shape[1]))
final_ind = np.max(discount_interaction_ind) + 1

alpha_buy = np.zeros(consumer_data.shape[0])
alpha_out = np.zeros(consumer_data.shape[0])

check_profit = np.repeat(5.0,consumer_data.shape[0])

for i in range(consumer_data.shape[0]):
    market = consumer_data[i,0]
    time = int(consumer_data[i,1])
    market_index = market_data[:,0]==market
    X_i = market_data[market_index,:][:,demand_ind]
    W_i = market_data[market_index,:][:,cost_ind]
    D_i = consumer_data[i,discount_data_ind]
    Z_i = consumer_data[i,consumer_ind]
    dat = Data(i,X_i,W_i,D_i,Z_i,0,0,0)

    mbs_price_i = mbs_data.loc[time,prices].to_numpy()
    MBS_par = ModelTypes.MBS_Func(coupon_rates,mbs_price_i)
    α = consumer_data[i,2]
    r_eq, itr1,flag = solve_eq_optim(α,dat,theta,MBS_par)
    if not flag:
        print("WARNING: Failed Equilibirum at",i)
    # if α>(-200)
    #     r_eq, itr1 = solve_eq(α,dat,θ,MBS_par)
    # else
    #     r_eq, itr1 = solve_eq_robust(α,dat,θ,MBS_par)
    # end
    # if (α>(-500)) & (itr1>=249)
    #     r_eq, itr1 = solve_eq_robust(α,dat,θ,MBS_par)
    # end
    shares = market_shares(r_eq,α,dat,theta)
    sell = balance_sheet_alloc(r_eq,dat,theta,MBS_par)

    share_draw = random.random() #*sum(shares)
    sell_draw = random.random()
    j_select = np.argmax(share_draw<np.cumsum(shares))

    if share_draw > np.sum(shares):
        share_data[time,2] +=1
        consumer_data[i,3] = -1
        alpha_out[i] = α
        # println("Consumer $i, solve iterations $itr1, inside share: $(sum(shares)) - skip")
        continue
    else:
        alpha_buy[i] = α
        share_data[time,1] +=1
        share_data[time,5+j_select] +=1

    # share_data[time+1,3] +=1*(1 - sum(shares))
    # share_data[time+1,2] +=1*(sum(shares))

    check_profit[i] = expected_sale_profit(r_eq,dat,theta,MBS_par)[j_select]
    
    consumer_data[i,3] = j_select
    consumer_data[i,4] = r_eq[j_select]
    consumer_data[i,5] = 1.0*(sell_draw<sell[j_select])
    consumer_data[i,discount_interaction_ind] = consumer_data[i,discount_interaction_ind]*r_eq[j_select]
    consumer_data[i,final_ind] = MBS_par.P(r_eq[j_select] - 0.0025)
    consumer_data[i,(final_ind + 1):(final_ind + 1 + W_i.shape[1])] = W_i[j_select,:]
    consumer_data[i,final_ind + 1 +  W_i.shape[1]] = 1 - sum(shares)
    consumer_data[i,final_ind + 1 +  W_i.shape[1]+1] = i
    # α_est, r_eq, itr2 = solve_eq_r(r_eq[j_select],j_select,dat,θ,MBS_par)
    # consumer_data[i,7] = α_est

consumer_data = consumer_data[consumer_data[:,3]>=0,:]


np.median(consumer_data[:,4])
np.quantile(consumer_data[:,4],0.1)
np.quantile(consumer_data[:,4],0.9)
np.mean(consumer_data[:,5])


print(len(consumer_data))
print(np.mean(consumer_data[:,5]))

share_data[:,4] = (share_data[:,1] + share_data[:,2])
share_data[:,3] = share_data[:,2]/share_data[:,4]
print(share_data[:,3])

file = "consumer_data_sim.csv"
pd.DataFrame(consumer_data).to_csv(file,index=False)

file = "market_data_sim.csv"
pd.DataFrame(market_data).to_csv(file,index=False)

# file = "mbs_data.csv"
# CSV.write(file,DataFrame(mbs_data,:auto))

file = "share_data_sim.csv"
pd.DataFrame(share_data).to_csv(file,index=False)


# r = 0.045
# v = r*208 + MBS_par.P(r)*(-1/theta.sigma) + np.dot(dat.Z,theta.gamma_ZS)/theta.sigma + np.dot(dat.W,theta.gamma_WS)/theta.sigma
# np.exp(v)/(1+np.exp(v))

# α = -200
# r_eq, itr1 = solve_eq_robust(α,dat,theta,MBS_par)
# mc = min_rate(dat,theta,MBS_par)
# (r_eq-mc)/mc



# alpha_buy = np.sort(alpha_buy[alpha_buy<0.0])
# alpha_out = np.sort(alpha_out[alpha_out<0.0])

# alpha_all = np.sort(np.concatenate((alpha_buy,alpha_out)))


# buy_dist = sp.stats.gaussian_kde(alpha_buy)
# out_dist = sp.stats.gaussian_kde(alpha_out)
# all_dist = sp.stats.gaussian_kde(alpha_all)

# buy_obs = buy_dist(alpha_buy)
# out_obs = out_dist(alpha_out)
# all_obs = all_dist(alpha_all)

# buy_prob = len(alpha_buy)/len(alpha_all)

# approx_all_obs = buy_dist(alpha_all)*buy_prob + out_dist(alpha_all)*(1-buy_prob)

# plt.plot(alpha_buy,buy_obs)
# plt.plot(alpha_out,out_obs)
# plt.plot(alpha_all,all_obs)
# plt.show()

# plt.plot(alpha_all,approx_all_obs)
# plt.plot(alpha_all,all_obs)
# plt.show()


