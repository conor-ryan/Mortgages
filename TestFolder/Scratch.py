import pandas as pd
import numpy as np


from FirstStageFunctions import *
from ModelTypes import *
from EquilibriumFunctions import *
from ModelFunctions import *
from EstimationFunctions import *
from ParallelFunctions import *
from Derivatives import * 
from NumericalDerivatives import *
# File with Loan-Level Data
consumer_data_file = "consumer_data.csv"

#### First Stage Specification ####
# Hold/Sell (Defined as Sell=1)
sell_spec = "5"
# Interest Rates (Interacted with Time Fixed Effects)
# interest_rate_spec = ["x16","x17","x18","x19","x20"]
interest_rate_spec = ["16","17"]
mbs_price = "18" #"x21"
# Consumer/Loan Characteristics
consumer_spec = ["7","8","9","10","11","12","13","6"]
# Bank/Aggregate Characteristics - Cost
# bank_spec = ["x22","x23","x24","x25","x26","x27"]
bank_spec = ["19","20","21"]

# res, pars = run_first_stage(consumer_data_file,sell_spec,interest_rate_spec,mbs_price,consumer_spec,bank_spec)



#### Second Stage Specification ####
# Consumer/Loan Characteristics - Variables in Consumer Data
consumer_cost_spec =["7","8","9","10","11","12","13","6"]
# Bank Cost Covariates - Variables in Market Data
bank_cost_spec = ["7","8","9"]
# Bank Demand Covariates - Variables in Market Data
bank_dem_spec = ["2","3","4","5","6","10"]

# Discount Factor Covariates ("Time Dummies or Aggregate Data") - Consumer Data
discount_spec = ["14","15"]#,"x13","x14","x15"]

# MBS Price Function Specification
# Price variables in MBS data
mbs_spec = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.25, 3.5, 3.75, 4, 4.25, 4.5, 4.75, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7, 7.25, 7.5, 7.75, 8, 8.25, 8.5, 8.75, 9]
mbs_spec = [str(p) for p in mbs_spec]
# Corresponding coupon rates
mbs_coupons = np.arange(1,9.1,0.25)/100

# Selection and Interest Rate indices
rate_spec = "4"
lender_spec = "3"
market_spec = "0"
time_spec = "1"
outside_share_index = "1" # Index ranging 0,1,... that matches consumers to outside share data


## Read in Data 
consumer_data = pd.read_csv("consumer_data.csv") # Consumer/Loan level data
market_data = pd.read_csv("market_data.csv") # Market level data
mbs_data = pd.read_csv("mbs_data_real_subset.csv") # MBS coupon price data
share_data = pd.read_csv("share_data.csv") # Aggregate share data

true_first_stage = ParameterFirstStage(0.04,np.array([0.5,0.25]),
                                       np.array([-0.299,1.1e-4,3e-5,-7.2e-4,0,6.55e-3,7e-5,-0.161]),
                                       np.array([0.044,0.00094,-0.00514]))

# fs = ParameterFirstStage(0.04,np.array([0.5,0.25]),
#                                        np.array([-0.299]),
#                                        np.array([0.044,0.00094,-0.00514]))

theta = Parameters(bank_dem_spec,bank_cost_spec,consumer_cost_spec,discount_spec,
                   mbs_spec,mbs_coupons,
                   rate_spec,lender_spec,market_spec,time_spec,outside_share_index,
                   true_first_stage,
                   share_data["3"],share_data["4"])






true_parameters = np.array([2.3,2.1, 1.9, 1.7,1.5,200.0, # Beta_x
                   -0.01,-0.005,0.002, #Gamma_WH
                   0.32,-1e-4,-1e-4,1e-4,0.00,0.005,0.00,0.1]) # Gamma_ZH

start_parameters = np.array([0.1,0,0,0,0,0, # Beta_x
                   0,0,0, #Gamma_WH
                   0.0,0,0,0,0,0,0,0]) # Gamma_ZH


# i = 21
# theta.set(true_parameters)
# dat,mbs = consumer_subset(i,theta,consumer_data,market_data,mbs_data)
# alpha, r, itr = solve_eq_r_robust(dat.r_obs,dat.lender_obs,dat,theta,mbs)

# alpha = consumer_data.loc[i,"2"]
# r, itr = solve_eq_robust(alpha,dat,theta,mbs)

# r, itr = solve_eq(alpha,dat,theta,mbs)

f_val, pre_start = estimate_GA(start_parameters,theta,consumer_data,market_data,mbs_data)

# opt_res = py_optim(pre_start,theta,consumer_data,market_data,mbs_data)

ll2, grad2,hess2 = evaluate_likelihood_hessian(pre_start,theta,consumer_data,market_data,mbs_data)

# opt_res2 = py_optim2(pre_start,theta,consumer_data,market_data,mbs_data)


f_val, res = estimate_NR(pre_start,theta,consumer_data,market_data,mbs_data)

res2 = estimate_GA(res,theta,consumer_data,market_data,mbs_data)

evaluate_likelihood_gradient(pre_start,theta,consumer_data,market_data,mbs_data)
evaluate_likelihood_hessian(pre_start,theta,consumer_data,market_data,mbs_data)

mbsdf = mbs_data
cdf = consumer_data
mdf = market_data   
x = pre_start

clist = consumer_object_list(theta,cdf,mdf,mbsdf)
ll3 = evaluate_likelihood_parallel(x,theta,clist,3)
ll4, grad4 = evaluate_likelihood_gradient_parallel(x,theta,clist,3)
ll5, grad5,hess5 = evaluate_likelihood_hessian_parallel(x,theta,clist,3)


## Likelihood Derivatives
ll0 = evaluate_likelihood(true_parameters,theta,consumer_data,market_data,mbs_data)
ll1, grad1 = evaluate_likelihood_gradient(true_parameters,theta,consumer_data,market_data,mbs_data)
print(grad1)
ll2, grad2,hess2 = evaluate_likelihood_hessian(true_parameters,theta,consumer_data,market_data,mbs_data)
print(grad2)
t1 = deriv_test_likelihood(true_parameters,theta,consumer_data,market_data,mbs_data)
t2 = num_hessian_likelihood(true_parameters,theta,consumer_data,market_data,mbs_data)

print(np.max(np.abs((grad1-t1)/t1)))
print(np.max(np.abs((hess2-t2)/t2)))

test = np.array([31.26264496, 31.04771438, 30.7065461,  30.73619066, 31.11616083,  0.07386583,
  0.06117955 , 0.14697843,  0.20354394,  0.92667736, -3.24956762,  0.09978058,
 -0.09514625])

ll0 = evaluate_likelihood(test,theta,consumer_data,market_data,mbs_data)

# # true_parameters= np.copy(test)

# res[8] = 1.02
# f_val, res = estimate_NR(res,theta,consumer_data,market_data,mbs_data)
# ll_vec[i] = f_val
# LL = evaluate_likelihood(test,theta,consumer_data,market_data)
# for i in range(5):
#     start = time.process_time()
#     LL, grad,hess = evaluate_likelihood_hessian(true_parameters,theta,consumer_data,market_data,mbs_data)
#     end = time.process_time()
#     print(LL,end-start)


# N = 4
# object_list = consumer_object_list(theta,consumer_data,market_data)
# for i in range(5):
#     s1 = time.process_time()
#     s2 = time.time()
#     # LL = evaluate_likelihood_parallel(true_parameters,theta,consumer_data,market_data,N)
#     ll = eval_map(object_list,N)
#     e2 = time.time()
#     e1 = time.process_time()
#     print(LL,e1-s1,e2-s2)


# theta.set(true_parameters)


# test = np.array([ 0.67702366,  0.31699224,  0.03321853, -0.17520471,  0.94310762,  0.95894717,
#   0.99669461 , 1.01717964,  0.9,        -3.0,          0.03795377, -0.13243693])

# test = np.array([ 0.6333619 ,  0.29489977 , 0.04581763 ,-0.19886046  ,0.07027512 , 0.08091958,
#   0.11649356,  0.10968265 , 1.02    ,   -3.0,          0.1     ,   -0.1  ,     ])
# # test = np.array([ 0.76108574,  0.44078219,  0.19209339, -0.04355448,  1.09845783,  1.10860084,
# #   1.14630117,  1.14044716,  1.03,       -3.0,          0.1,        -0.1      ])



# test2 = np.copy(test)
# test2[8] = 1.025
# evaluate_likelihood(test2,theta,consumer_data,market_data,mbs_data)


# test3 = np.copy(test)
# test3[8] = 1.05
# evaluate_likelihood(test3,theta,consumer_data,market_data,mbs_data)


# for i in range(1500,1700):
#   theta.set(test)
#   alpha_true = consumer_data.loc[i,"x3"]
#   dat,mbs = consumer_subset(i,theta,consumer_data,market_data,mbs_data)
#   alpha, r, itr = solve_eq_r(dat.r_obs,dat.lender_obs,dat,theta,mbs)
#   q = market_shares(r,alpha,dat,theta)
#   mc = min_rate(dat,theta,mbs)
#   low_cost = np.where(mc==np.min(mc))
#   print(i,alpha,alpha_true,q[dat.lender_obs],q[low_cost])

#   theta.set(test)
#   ll1,alpha,r, itr = consumer_likelihood_eval(theta,dat,mbs)
#   theta.set(test3)
#   ll2,alpha2,r2, itr  = consumer_likelihood_eval(theta,dat,mbs)
#   q2= market_shares(r2,alpha2,dat,theta)
#   print(i, "Likelihood Change",ll2-ll1)
  
#   print(i,alpha2,alpha_true,q2[dat.lender_obs])


# test3 = np.copy(test2)
# test3[8]=1.028

# i = 0
# theta.set(test)
# dat,mbs = consumer_subset(i,theta,consumer_data,market_data,mbs_data)
# alpha1, r1, itr = solve_eq_r(dat.r_obs,dat.lender_obs,dat,theta,mbs)
# q1 = market_shares(r1,alpha1,dat,theta)
# np.log(q1[dat.lender_obs])

# theta.set(test2)
# dat,mbs = consumer_subset(i,theta,consumer_data,market_data,mbs_data)
# alpha2, r2, itr = solve_eq_r_robust(dat.r_obs,dat.lender_obs,dat,theta,mbs)
# q2 = market_shares(r2,alpha2,dat,theta)
# np.log(q2[dat.lender_obs])

# theta.set(test3)
# dat,mbs = consumer_subset(i,theta,consumer_data,market_data,mbs_data)
# alpha3, r3, itr = solve_eq_r_robust(dat.r_obs,dat.lender_obs,dat,theta,mbs)
# q3 = market_shares(r3,alpha3,dat,theta)
# np.log(q3[dat.lender_obs])

i =0
theta.set(true_parameters)
dat,mbs = consumer_subset(i,theta,consumer_data,market_data,mbs_data)

r0 = dat.r_obs
j = dat.lender_obs
d = dat
m = mbs

alpha, r, itr = solve_eq_r(r0,j,d,theta,m)

solve_eq_r_optim(r0,j,d,theta,m)

r = r + 0.01
t = deriv_test_foc(r,alpha,d,theta,m)
np.diag(t)
expected_foc_nonlinear(r,alpha,d,theta,m)

f,g,h = d2SaleProfit_dr2(r,d,theta,m)

t0  = deriv_test_alpha_share(r,alpha,dat,theta,mbs)
t1  = deriv_test_alpha(r,alpha,dat,theta,mbs)
t2 = deriv_test_rate(r,alpha,dat,theta,mbs)
t3 = deriv_test_beta_x(r,alpha,dat,theta,mbs)
t4 = deriv_test_gamma(r,alpha,dat,theta,mbs)

dalpha, dr, dbeta, dgamma = d_foc_all_parameters(r,alpha,dat,theta,mbs)
dalpha1, dr1,foc = d_foc(r,alpha,dat,theta,mbs)

#### Test First Derivatives ####
print(np.sum(np.abs(t1-dalpha)))
print(np.sum(np.abs(t2-dr)))
print(np.sum(np.abs(t3-dbeta)))
print(np.sum(np.abs(t4-dgamma)))

dalpha, dr, dbeta, dgamma,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,dat,theta,mbs)

#### Test First Derivatives ####
print(np.sum(np.abs(t1-dalpha)))
print(np.sum(np.abs(t2-dr)))
print(np.sum(np.abs(t3-dbeta)))
print(np.sum(np.abs(t4-dgamma)))

# #### Test Second Derivatives ####
t1 = deriv2_test_alpha(r,alpha,d,theta,m)
t2 = deriv2_test_rate(r,alpha,d,theta,m)
t3 = deriv2_test_alpha_rate(r,alpha,d,theta,m)
t4 = deriv2_test_beta(r,alpha,d,theta,m)
t5 = deriv2_test_alpha_beta(r,alpha,d,theta,m)
t6 = deriv2_test_r_beta(r,alpha,d,theta,m)
t7 = deriv2_test_gamma(r,alpha,d,theta,m)

print(np.sum(np.abs((t1-d2alpha))))
print(np.sum(np.abs((t2-d2r))))
print(np.sum(np.abs((t3-drdalpha))))
print(np.sum(np.abs((t4-d2beta))))
print(np.sum(np.abs((t5-dbetadalpha))))
print(np.sum(np.abs((t6-drdbeta))))

# l = 0
# print(d2r[l,l,l])
# print(t2[l,l,l])


## Implicit/Total Derivatives

endo_grad, grad0, outside_grad = share_parameter_derivatives(r,alpha,dat,theta,mbs)

grad1, hess1, d1, d2 = share_parameter_second_derivatives(r,alpha,dat,theta,mbs)
t1 = deriv_test_endo(dat,theta,mbs)
t2 = deriv2_test_endo(dat,theta,mbs)
t3 = deriv_test_total_shares(dat,theta,mbs)
t4 = deriv2_test_total_shares(dat,theta,mbs)


print(t1-dendo)
print(np.sum(np.abs(t1-dendo)))
print(np.sum(np.abs(t2-d2endo)))
print(np.sum(np.abs(t4-hess1)))
t4[10:11,10:11,:]
hess1[10:11,10:11,:]






np.max(np.abs((grad1-t1)/t1))
np.max(np.abs((hess2-t2)/t2))

k = 0
(t2[k,:] - hess2[k,:])/t2[k,:]
t2[k,:]
hess2[k,:]
# # Profit from an origination 
# pi_h = pi_hold(r,theta,dat) # HTM
# pi_s = pi_sell(r,theta,dat,mbs) # OTD

# # Derivative of origination profit
# dpi_h = dpidr_hold(r,theta,dat) #HTM
# dpi_s = dpidr_sell(r,mbs) #OTD

# # Market Shares and Derivatives
# q,dqdp = share_derivative(r,alpha,dat,theta)

# # Exponential of Expected Origination Profit (pre-computation)
# epi_h = np.exp(pi_h/theta.sigma)
# epi_s = np.exp(pi_s/theta.sigma)

# dEPidr = (dqdp*(pi_h*epi_h + pi_s*epi_s) + q*(dpi_h*epi_h + dpi_s*epi_s))/(epi_h + epi_s)

# print(dEPidr)

# prob_h = epi_h/(epi_h+epi_s)
# prob_s = 1 - prob_h

# 1/(alpha*(1-q)) - prob_h*pi_h - prob_s*pi_s

# theta.set(true_parameters)
theta.set(test)
alpha_vec = consumer_data["x3"].to_numpy()
alpha_test = np.zeros(len(alpha_vec))

for i in range(consumer_data.shape[0]):
  dat,mbs = consumer_subset(i,theta,consumer_data,market_data,mbs_data)
  ll_i,alpha,r_eq,itr = consumer_likelihood_eval(theta,dat,mbs)
  alpha_test[i] = alpha

i = 758

theta.set(true_parameters)
# theta.set(test)

dat,mbs = consumer_subset(i,theta,consumer_data,market_data,mbs_data)
alpha1, r, itr = solve_eq_r(dat.r_obs,dat.lender_obs,dat,theta,mbs)
alpha2, r, itr = solve_eq_r_robust(dat.r_obs,dat.lender_obs,dat,theta,mbs)

alpha = -800
r, itr = solve_eq_robust(alpha,dat,theta,mbs)
print(r)