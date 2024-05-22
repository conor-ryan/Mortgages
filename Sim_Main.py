import pandas as pd
import numpy as np


from FirstStageFunctions import *
from ModelTypes import *
from EquilibriumFunctions import *
from ModelFunctions import *
from EstimationFunctions import *
from ParallelFunctions import *
from Derivatives import * 
from CostEstFunctions import *
from KernelFunctions import *

# File with Loan-Level Data
consumer_data_file = "consumer_data_sim.csv"

#### Load All Data ####
consumer_data = pd.read_csv("consumer_data_sim.csv") # Consumer/Loan level data
market_data = pd.read_csv("market_data_sim.csv") # Market level data
mbs_data = pd.read_csv("mbs_data_real_subset.csv") # MBS coupon price data
share_data = pd.read_csv("share_data_sim.csv") # Aggregate share data


#### First Stage Estimation ####
# Hold/Sell (Defined as Sell=1)
sell_spec = "5"
# Interest Rates (Interacted with Time Fixed Effects)
interest_rate_spec = ["16","17"]
# MBS Price Variable
mbs_price = "18"
# Consumer/Loan Characteristics
consumer_spec = ["7","8","9","10","11","12","13","6"]
# Bank/Aggregate Characteristics - Cost
bank_spec = ["19","20","21"]

# Doesn't estimate well on simulated data. 
#res, pars = run_first_stage(consumer_data_file,sell_spec,interest_rate_spec,mbs_price,consumer_spec,bank_spec)
# Save approximate results from real data

est_first_stage = ParameterFirstStage(0.04,np.array([0.5,0.25]),
                                       np.array([-0.299,1.1e-4,3e-5,-7.2e-4,0,6.55e-3,7e-5,-0.161]),
                                       np.array([0.044,0.00094,-0.00514]))





#### Estimate Costs ####
## Need two specifications in addition to those from the first stage
# - Disentangle interest rate and time variables
# - All still consumer data only

# Discount Factor Covariates (Time Dummies or Aggregate Data) 
discount_spec = ["14","15"]

# Interest rate of originated loan
rate_spec = "4"

# Estimate the costs that just-bind constraints
cost_res,keep_index = estimate_costs(rate_spec,mbs_price,consumer_spec,bank_spec,discount_spec,
                                est_first_stage,consumer_data)

## Drop observations where constraints are binding
consumer_data = consumer_data[keep_index]

## Pandas is being weird on the index line, so this refresh seems necessary?
pd.DataFrame(consumer_data).to_csv("refresh.csv",index=False)
consumer_data = pd.read_csv("refresh.csv")

#### Initialize Second Stage Parameter Object ####
# Bank Cost Covariates in the Market Data - Analogous to first stage variables in consumer-level data
bank_cost_mkt_spec = ["7","8","9"]
# Demand Covariates - Variables in Market Data
bank_dem_spec = ["2","3","4","5","6","10"]

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

theta = Parameters(consumer_data,
                   bank_dem_spec,bank_cost_mkt_spec,consumer_spec,discount_spec,
                   mbs_spec,mbs_coupons,
                   rate_spec,lender_spec,market_spec,time_spec,outside_share_index,
                   est_first_stage,
                   share_data["3"],share_data["4"])


#### Estimate Second Stage #### 

# ..... 