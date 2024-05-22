import numpy as np
import pandas as pd 
from statsmodels.discrete.discrete_model import Logit

#### Class: First Stage Variables - the structural parameters we are going to estimate
class ParameterFirstStage:

    def __init__(self,sigma,beta_d,gamma_ZS,gamma_WS):
        self.beta_d = beta_d
        self.sigma = sigma
        self.gamma_WS = gamma_WS
        self.gamma_ZS = gamma_ZS

    def summary(self):
        print(f"Sigma: {self.sigma:.3}") 
        print("Discount Par:", np.round(self.beta_d,3)) 
        print("Cons/Loan Par:", np.round(self.gamma_ZS,3)) 
        print("Bank Par:", np.round(self.gamma_WS,2)) 
    
    



def run_first_stage(file,sell_spec,int_spec,m_spec,Z_spec,W_spec):
    data = pd.read_csv(file)

    # Identify independent and dependent variables
    y = data[sell_spec]
    full_spec = int_spec + [m_spec] + Z_spec + W_spec
    X = data[full_spec]

    # Estimate Logit Regression

    model = Logit(y,X)
    result= model.fit()

    # Back out structural parameters
    sigma = 1/result.params[m_spec]
    beta_d = -1/(result.params[int_spec]*sigma)
    gamma_ZS = -result.params[Z_spec]*sigma
    gamma_WS = -result.params[W_spec]*sigma
    
    parameters = ParameterFirstStage(sigma,beta_d.to_numpy(),gamma_ZS.to_numpy(),gamma_WS.to_numpy())

    print(result.summary())
    parameters.summary()

    return result, parameters

