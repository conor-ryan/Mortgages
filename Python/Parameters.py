#### Data Sets ####

# D - Covariates that we use to determine interest rate discount factor
# M - Linear Interpolation Points for MBS prices
# X - Demand Covariates
# WH - HTM Cost Covariates 
# WS - OTD Cost Difference Covariates
# ZH - Consumer-Specific HTM Costs
# ZS - Consumer-Specific OTD Costs

## Consumer Data = [ZH, ZS, M, D]
## Market Data = [X, WH, WS]

#### Class Variable: the object we are going to estimate
class ParameterFirstStage:

    def __init__(self,vec):
        self.beta_D
        self.sigma
        self.gamma_WS
        self.gamma_ZS



#### Class Variable: the object we are going to estimate
class Parameter:

    def __init__(self,df,dem_spec_list,cost_spec):
        self.beta_X
        self.gamma_WH
        self.gamma_ZH
