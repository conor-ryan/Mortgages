import pandas as pd
import numpy as np

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

cost_res,keep_index = estimate_costs(rate_spec,mbs_price,consumer_spec,bank_spec,discount_spec,
                                true_first_stage,consumer_data)


percent_drop = 1 - sum(keep_index)/len(keep_index)
print("Cost Estimate Drop Fraction:", percent_drop)

consumer_data = consumer_data[keep_index]
pd.DataFrame(consumer_data).to_csv("refresh.csv",index=False)
consumer_data = pd.read_csv("refresh.csv")

theta = Parameters(consumer_data,
                   bank_dem_spec,bank_cost_spec,consumer_cost_spec,discount_spec,
                   mbs_spec,mbs_coupons,
                   rate_spec,lender_spec,market_spec,time_spec,outside_share_index,
                   true_first_stage,
                   share_data["3"],share_data["4"])

cost_true = np.array([0,0,0.000,#Gamma_WH
                      0.4,0.0,0,0,0,0.00,0,0.00])
theta.set_cost(cost_res.x)
true_parameters = np.array([12.3,12.1, 11.9, 11.7,11.5,5.0])#, # Beta_x
                #    0,0,0, #Gamma_WH
                #    0.32,0]) # Gamma_ZH

#### Run Timing Tests ####
clist = consumer_object_list(theta,consumer_data,market_data,mbs_data)
# print("Timing 2 Cores")
# for i in range(5):
#     start = time.perf_counter()
#     res =  evaluate_likelihood_hessian_parallel(true_parameters,theta,clist,2)
#     end = time.perf_counter()
#     elapsed = end - start
#     print(f'Elapsed Time: {elapsed:.6f} seconds')

# print("Timing 4 Cores")
# for i in range(5):
#     start = time.perf_counter()
#     res =  evaluate_likelihood_hessian_parallel(true_parameters,theta,clist,4)
#     end = time.perf_counter()
#     elapsed = end - start
#     print(f'Elapsed Time: {elapsed:.6f} seconds')


print("Timing 8 Cores")
for i in range(5):
    start = time.perf_counter()
    res =  evaluate_likelihood_hessian_parallel(true_parameters,theta,clist,8)
    end = time.perf_counter()
    elapsed = end - start
    print(f'Elapsed Time: {elapsed:.6f} seconds')


print("Timing 16 Cores")
for i in range(5):
    start = time.perf_counter()
    res =  evaluate_likelihood_hessian_parallel(true_parameters,theta,clist,16)
    end = time.perf_counter()
    elapsed = end - start
    print(f'Elapsed Time: {elapsed:.6f} seconds')

NUM_WORKERS = 16
start_parameters = np.zeros(len(true_parameters))

# print("Test Serial Gradient Ascent")
# res = estimate_GA(true_parameters,theta,(consumer_data,market_data,mbs_data),parallel=False,itr_max=10)


# print("Estimate in Parallel")
# f_val, res = estimate_NR(start_parameters,theta,consumer_data,market_data,mbs_data,parallel=True,num_workers=NUM_WORKERS,gtol=1e-4)

print("Estimate in Parallel with precondition")
f_val, res = estimate_NR(start_parameters,theta,consumer_data,market_data,mbs_data,parallel=True,num_workers=NUM_WORKERS,gtol=1e-4,pre_condition=True)


print("Estimate without parallel")
f_val, res = estimate_NR(true_parameters,theta,consumer_data,market_data,mbs_data,parallel=False,gtol=1e-6,pre_condition=True)



