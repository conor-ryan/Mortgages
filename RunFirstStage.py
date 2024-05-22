import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit
from FirstStageFunctions import *


# File with Loan-Level Data
consumer_data_file = "consumer_data.csv"

#### First Stage Specification ####
# Hold/Sell (Defined as Sell=1)
sell_spec = "x6"
# Interest Rates (Interacted with Time Fixed Effects)
interest_rate_spec = ["x13","x14"]
mbs_price = "x15" #"x21"
# Consumer/Loan Characteristics
consumer_spec = ["x7","x8"]
# Bank/Aggregate Characteristics - Cost
bank_spec = ["x16","x17","x18","x19","x20","x21"]

res, pars = run_first_stage(consumer_data_file,sell_spec,interest_rate_spec,mbs_price,consumer_spec,bank_spec)


