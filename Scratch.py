import pandas as pd
import numpy as np
import numdifftools as nd

import cProfile
import matplotlib.pyplot as plt
import time 

from FirstStageFunctions import *
from ModelTypes import *
from EquilibriumFunctions import *
from ModelFunctions import *
from EstimationFunctions import *
from ParallelFunctions import *
from Derivatives import * 
from NumericalDerivatives import *
from CostEstFunctions import *
from KernelFunctions import *
from TestConditionalFunctions import *

# File with Loan-Level Data
consumer_data_file = "consumer_data_sim.csv"

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
consumer_cost_spec =["7", "8","9","10","11","12","13","6"]
# Bank Cost Covariates - Variables in Market Data
bank_cost_spec = ["7","8","9"]
# Bank Demand Covariates - Variables in Market Data
bank_dem_spec = ["2","3","4","5","6","0"]

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
consumer_data = pd.read_csv("consumer_data_sim.csv") # Consumer/Loan level data
market_data = pd.read_csv("market_data_sim.csv") # Market level data
mbs_data = pd.read_csv("mbs_data_real_subset.csv") # MBS coupon price data
share_data = pd.read_csv("share_data_sim.csv") # Aggregate share data

true_first_stage = ParameterFirstStage(0.04,np.array([0.5,0.25]),
                                       np.array([-0.299,1.1e-4,3e-5,-7.2e-4,0,6.55e-3,7e-5,-0.161]),
                                       np.array([0.044,0.00094,-0.00514]))

# fs = ParameterFirstStage(0.04,np.array([0.5,0.25]),
#                                        np.array([-0.299,-0.161]),
#                                        np.array([0.044,0.00094,-0.00514]))



# consumer_spec = ["7","6"]
# bank_spec = ["13","14","15"]
cost_spec = bank_spec + consumer_spec
# res,keep_index = estimate_costs(cost_spec,theta,consumer_data,mbs_data)

cost_res = estimate_costs(rate_spec,mbs_price,consumer_spec,bank_spec,discount_spec,
                                true_first_stage,consumer_data)


# consumer_data = consumer_data[keep_index]
# pd.DataFrame(consumer_data).to_csv("refresh.csv",index=False)
# consumer_data = pd.read_csv("refresh.csv")
# # true_parameters = np.array([2.3,2.1, 1.9, 1.7,1.5,200.0, # Beta_x
# #                    -0.01,-0.005,0.002, #Gamma_WH
# #                    0.32,-1e-4,-1e-4,1e-4,0.00,0.005,0.00,0.1]) # Gamma_ZH

theta = Parameters(bank_dem_spec,bank_cost_spec,consumer_cost_spec,discount_spec,
                   mbs_spec,mbs_coupons,
                   rate_spec,lender_spec,market_spec,time_spec,outside_share_index,
                   true_first_stage,
                   share_data["3"],share_data["4"])

cost_true = np.array([0,0,0.000,#Gamma_WH
                      0.4,0.0,0,0,0,0.00,0,0.00])
theta.set_cost(cost_res.x)
true_parameters = np.array([12.3,12.1, 11.9, 11.7,11.5,5])#, # Beta_x
                #    0,0,0, #Gamma_WH
                #    0.32,0]) # Gamma_ZH



# val,pre_start = estimate_GA(true_parameters,theta,consumer_data,market_data,mbs_data)
# f_val, res = estimate_NR(true_parameters,theta,consumer_data,market_data,mbs_data)
# theta.out_share
# theta.out_share = np.array([0.5,0.5])
# start = np.array([  5.57371928,   3.78605371,   2.70256459,   2.56466092 ,  3.21529397, 127.18751441])
# res = np.array([  16.6769731,    16.47231718 ,  15.8495233   , 16.13734006,   15.65849686,
#  -135.99767026])
res = np.zeros(len(true_parameters))
res[0:6] = [ 8.51718423,  6.92391885 , 6.45555841,  5.14300395,  3.91491964 ,15.14973622]

clist = consumer_object_list(theta,consumer_data,market_data,mbs_data)
ll0 = evaluate_likelihood(res,theta,clist)
ll1, grad1 = evaluate_likelihood_gradient(res,theta,clist)
print(grad1)


ll2, grad2,hess2,bfgs = evaluate_likelihood_hessian(res,theta,consumer_data,market_data,mbs_data)
print(grad2)
t1 = deriv_test_likelihood(res,theta,clist)
t2 = num_hessian_likelihood(res,theta,consumer_data,market_data,mbs_data)

print(np.max(np.abs((grad1-t1)/t1)))
print(np.max(np.abs((hess2-t2)/t2)))




# start = np.array([  9.44449934,   9.10344673,   9.15257351,   8.80094886,
#          8.59485071, -36.19692372])
# f_val, res = estimate_NR_parallel(true_parameters,theta,consumer_data,market_data,mbs_data,4,gtol=1e-6,xtol=1e-15)


a_vec, e_vec,flag_vec = predicted_elasticity(true_parameters,theta,consumer_data,market_data,mbs_data)


# print("Test Numerical Derivative At Optimum")
# clist = consumer_object_list(theta,consumer_data,market_data,mbs_data)
# ll1, grad1 = evaluate_likelihood_gradient(res,theta,clist,parallel=True,num_workers=NUM_WORKERS)
# print("Analytical Gradient")
# print(grad1)
# g_size = np.mean(np.sqrt(grad1[0:len(res)]**2))
# print(g_size)
# print("Numerical Gradient")
# g_test = deriv_test_ll_parallel(res,theta,clist,NUM_WORKERS)
# print(g_test)
# print(grad1[0:len(res)]-g_test)
# g_test_size =  np.mean(np.sqrt(g_test**2))
# print(g_test_size)


fval, res  = parallel_optim(start,theta,consumer_data,market_data,mbs_data,4)


# f_val, res = estimate_NR_parallel(true_parameters,theta,consumer_data,market_data,mbs_data,5)


# q0 = consumer_data.loc[:,"16"]

# [ 14.0866772   13.79543238  13.6109838   13.8024465   13.38834936
#  -29.83301558]


# test = np.array([  3.7456483 ,   3.44958674 ,  3.14584182,   3.02402567,   3.04680003,
#  110.11616341])


ll1, grad1 = evaluate_likelihood_gradient(res,theta,consumer_data,market_data,mbs_data)
print(ll1)

test2 = np.copy(res)
test2[:] = res[:] + 1e-3*grad1[0:len(res)]
ll2= evaluate_likelihood(test2,theta,consumer_data,market_data,mbs_data)
print(ll1)
print(ll2)
print(ll2-ll1)


print(grad1)
t1 = deriv_test_likelihood(res,theta,consumer_data,market_data,mbs_data)

test3 = res + 1e-8*t1[0:len(res)]
ll3= evaluate_likelihood(test3,theta,consumer_data,market_data,mbs_data)
print(ll3)


# test = np.array([12.36842681, 28.26226776, 30.13299811, 27.9180135 , 24.39588405,  6.3557909 ])
# ll1, grad1 = evaluate_likelihood_gradient(test,theta,consumer_data,market_data,mbs_data)


start_parameters = np.array([0.1,0,0,0,0,0, # Beta_x
                   0,0,0, #Gamma_WH
                   0.0,0,0,0,0,0,0,0]) # Gamma_ZH

f_val, pre_start = estimate_GA(true_parameters,theta,consumer_data,market_data,mbs_data)


res = np.array([ 7.16132169e+00,  5.30514782e+00,  5.44097803e+00,  4.70089354e+00,
        4.64211039e+00,  2.77454352e+01, -2.47232740e-02,  1.31937388e-03,
        1.94965492e-03,  3.03415882e-01,  1.46197121e-05,  1.58609983e-05,
        2.98193558e-05, -1.40130341e-04, -1.16898133e-03,  1.52767444e-06,
        1.79284318e-01])

res = np.array([ 2.50981210e+01,  2.49661655e+01,  2.43613460e+01,  2.45295427e+01,
        2.45145333e+01,  2.35863734e+02, -2.27147180e-02,  6.09173913e-04,
        4.10439363e-03,  3.55194739e-01, -7.75966670e-02])

res = py_optim(pre_start,theta,consumer_data,market_data,mbs_data,model="base")


# opt_res = py_optim(pre_start,theta,consumer_data,market_data,mbs_data)

ll2, grad2,hess2 = evaluate_likelihood_hessian(pre_start,theta,consumer_data,market_data,mbs_data)

# opt_res2 = py_optim2(pre_start,theta,consumer_data,market_data,mbs_data)




evaluate_likelihood_gradient(pre_start,theta,consumer_data,market_data,mbs_data)
evaluate_likelihood_hessian(pre_start,theta,consumer_data,market_data,mbs_data)

mbsdf = mbs_data
cdf = consumer_data
mdf = market_data   
x = true_parameters
model = "base"

o = 1 
ind = theta.out_vec==o
a_mkt = alpha_list[ind]#[theta.out_sample[o]]
q0_mkt = q0_list[ind]#[theta.out_sample[o]]
c_mkt_H = c_list_H[ind]
c_mkt_S = c_list_S[ind]
out_share = theta.out_share[0]



da_mkt = dalpha_list[ind,:]
dq0_mkt = dq0_list[ind,:]
d2a_mkt = d2alpha_list[ind,:,:]
d2q0_mkt = d2q0_list[ind,:,:]
vec = x + np.random.rand(len(x))

f, g1 = out_share_gradient(a_mkt,c_mkt_H,c_mkt_S,q0_mkt,
                        da_mkt,dq0_mkt,out_share,theta)



test = np.zeros(len(a_mkt))
for i in range(len(a_mkt)):
    test[i] = np.dot(np.transpose((vec-x)),np.dot(d2a_mkt[i,0:len(vec),0:len(vec)],(vec-x)))
    # test[i,:] = np.dot(d2a_mkt[i,0:len(vec),0:len(vec)],(vec-x))

test2 = np.tensordot(d2a_mkt[:,0:len(vec),0:len(vec)],vec-x,axes=1)
test2 = 0.5*np.dot(np.tensordot(d2a_mkt[:,0:len(vec),0:len(vec)],vec-x,axes=1),np.transpose((vec-x)))

def f_test(vec):
    a_1 = a_mkt + np.dot(da_mkt[:,0:len(vec)],(vec-x)) + 0.5*np.dot(np.tensordot(d2a_mkt[:,0:len(vec),0:len(vec)],vec-x,axes=1),np.transpose((vec-x)))
    q_1 = q0_mkt + np.dot(dq0_mkt[:,0:len(vec)],(vec-x)) + 0.5*np.dot(np.tensordot(d2q0_mkt[:,0:len(vec),0:len(vec)],vec-x,axes=1),np.transpose((vec-x)))
    f0 = outside_share(a_1,c_mkt_H,c_mkt_S,q_1,out_share)

    return f0

grad_func = nd.Gradient(f_test,step=1e-8,method="central")
hess_func = nd.Hessian(f_test,step=1e-8,method="central")

t = grad_func(x)
h = hess_func(x)


a_vec =a_mkt
c_h_vec =c_mkt_H
c_s_vec =c_mkt_S
q0_vec =q0_mkt


plt.scatter(-np.log(-a_vec),dist_cond_obs)
plt.show()

plt.scatter(-np.log(-a_vec),dist_outside_obs)
plt.show()

plt.scatter(-np.log(-a_vec),dist_uncond)
plt.show()

## Likelihood Derivatives


test = np.array([31.26264496, 31.04771438, 30.7065461,  30.73619066, 31.11616083,  0.07386583,
  0.06117955 , 0.14697843,  0.20354394,  0.92667736, -3.24956762,  0.09978058,
 -0.09514625])

# clist = consumer_object_list(theta,consumer_data,market_data,mbs_data)
# ll1, grad1 = evaluate_likelihood_gradient_parallel(res,theta,clist,3)
# print(grad1)
# ll2, grad2,hess2,bfgs = evaluate_likelihood_hessian_parallel(res,theta,clist,3)
# print(grad2)
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
# 193, 247, 385, 524, 541
error = np.zeros((consumer_data.shape[0],2))
# for i in range(consumer_data.shape[0]):
i = 0
# i = 98, 120, 226, 269,181, 123
# test = np.copy(res)
# test[0] = test[0] + 1e-6
theta.set_demand(res)
dat,mbs = consumer_subset(i,theta,consumer_data,market_data,mbs_data)
ll_i, dll_i, q0, dq0,alpha, da, sb = consumer_likelihood_eval_gradient(theta,dat,mbs)
# ll_i, dll_i, d2ll_i, q0, dq0, d2q0, alpha, da,d2a, sb,ab = consumer_likelihood_eval_hessian(theta,dat,mbs)
g_test = deriv_test_cons_ll(res,theta,dat,mbs)
# h_test = hess_test_cons_ll(res,theta,dat,mbs)
# error[i,0] = np.sum(np.abs(g_test- dll_i[0:6]))
# error[i,1] = np.sum(np.abs(h_test- d2ll_i[0:6,0:6]))  
# np.sum(np.abs(g_test- dll_i[0:6]))
# np.sum(np.abs(h_test- d2ll_i[0:6,0:6]))

# test = np.copy(res)
# # test[5]+= 1.0
# theta.set_demand(test)
r0 = dat.r_obs
j = dat.lender_obs
d = dat
m = mbs
r_test = min_rate(d,theta,m) + 1/150
alpha, r, itr, f = solve_eq_r_optim(r0,j,d,theta,m)
q =  market_shares(r,alpha,d,theta)
print(alpha,r,1-sum(q))
print(consumer_likelihood_eval(theta,d,m))
print(ll_test_function(theta,d,m))

dlogq1, d2logq1, dq01,d2q01, da1,d2a1  =share_parameter_second_derivatives(r,alpha,d,theta,m)
q0 = 1-np.sum(q)
dlogq2, d2logq2, dq02,d2q02, da2,d2a2  =conditional_parameter_second_derivatives(r,alpha,d,theta,m)

t1 = dlogq1[:,j] + dq01/(1-q0)
t2 = dlogq2[:,j]

# alpha_seq = np.linspace(-126,-127,10)
# t1 = np.zeros(len(alpha_seq))
# for i in range(len(alpha_seq)):
#         r, itr = solve_eq(alpha_seq[i],d,theta,m)
#         q =  market_shares(r,alpha_seq[i],d,theta)
#         t1[i] = 1-sum(q)

# test = np.copy(res)
# test[0] = test[0] + 1e-3

theta.set_demand(res+10000)
ll_i, dll_i, d2ll_i, q0, dq0, d2q0, alpha, da, itr = consumer_likelihood_eval_hessian(theta,d,m)
print(q0)
print(ll_i)

# theta.set_demand(res + dll_i[0:len(res)]*1e-3)
# ll_2, q0, alpha, itr = consumer_likelihood_eval(theta,d,m)
# print(ll_2-ll_i)


# ll_i - np.log(q[dat.lender_obs])

dll_i = dll_i[0:6]
g_test = deriv_test_cons_ll(res,theta,d,m)
# error[i,:] = dll_i[0:len(true_parameters)] - g_test
h_test = hess_test_cons_ll(res,theta,d,m)
d2q0[0:6,0:6]
d2ll_i[0:6,0:6]
h_test


# dlogq, dq0, dalpha = share_parameter_derivatives(r,alpha,dat,theta,mbs)
# t3 = deriv_test_total_shares(dat,theta,mbs)
# dll_i - dlogq[0:6,dat.lender_obs]


solve_eq_r_optim(r0,j,dat,theta,mbs)

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
print(np.sum(np.abs(t1-dalpha1)))
print(np.sum(np.abs((t2-dr1))))
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
# t1 = deriv_test_endo(dat,theta,mbs)
# t2 = deriv2_test_endo(dat,theta,mbs)
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