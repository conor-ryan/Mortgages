import numpy as np
import scipy as sp
import ModelFunctions
import EstimationFunctions
import ModelTypes


def consumer_revenue(cost_data,HTM_rate_rev,MBS_rate_rev,first_stage):
    # Re-compute estimated HTM - OTD cost difference
    diff_cost_pars = np.concatenate((first_stage.gamma_WS,first_stage.gamma_ZS))
    diff_costs = np.dot(cost_data,diff_cost_pars)

    # Compute origination method revenues, gross over HTM cost
    rev_h = HTM_rate_rev
    rev_s = MBS_rate_rev - diff_costs


    # Exponential of Expected Origination Profit (pre-computation)
    epi_h = np.exp((rev_h)/first_stage.sigma)
    epi_s = np.exp((rev_s)/first_stage.sigma)

    # Probability of Hold vs Sell
    prob_h = epi_h/(epi_h+epi_s)
    prob_s = 1 - prob_h

    # Gross revenue expectation w.r.t. balance sheet shock
    ERev = prob_h*rev_h + prob_s*rev_s

    # ERev = first_stage.sigma*(np.log(np.exp(rev_h/first_stage.sigma) + np.exp(rev_s/first_stage.sigma)))

    return ERev.to_numpy()



def estimate_costs(rate_spec,mbs_spec,cons_cost,bank_cost,discount_spec,first_stage,cdf):
    # Combine cost specifications 
    all_cost_spec = bank_cost + cons_cost
    # Combine all cost co-variates for originated loans
    cost_data = cdf[all_cost_spec].to_numpy()
    # Compute gross revenue per origination method
    HTM_rate_rev = cdf[rate_spec]/np.dot(cdf[discount_spec],first_stage.beta_d)
    MBS_rate_rev = cdf[mbs_spec]
    
    # Compute expected revenue of originated loan
    revenues = consumer_revenue(cost_data,HTM_rate_rev,MBS_rate_rev,first_stage)
    
    # Define constraints: 0 <= predicted marginal cost <= expected originated revenue
    lin_const = sp.optimize.LinearConstraint(cost_data,np.repeat(0.0,len(revenues)),revenues)

    # Define a wrapper for predicted marginal cost
    # Objective value: normalization of sum squared cost
    def f_obj(x):
        costs = np.dot(cost_data,x)
        return -sum(costs**2)/1e4

    # Starting parameter guess
    cost_parameters = np.zeros(cost_data.shape[1])
    # Set the guess on intercept term to be small, non-zero
    # cost_parameters[0] = 0.1 
    
    # Solve optimization problem
    res = sp.optimize.minimize(f_obj,cost_parameters,method='SLSQP',
                                constraints = lin_const,
                                options={'disp':True,'ftol':1e-9})
    
    # Create an index of all observations where estimated cost is not roughly equal to expected revenue.
    estimated_hold_costs = np.dot(cost_data,res.x)
    if any(estimated_hold_costs<=0):
        print("Warning: Negative Costs",np.mean(estimated_hold_costs<=0))

    return res

def drop_low_margins(theta,cdf,mdf,mbsdf):
    clist = EstimationFunctions.consumer_object_list(theta,cdf,mdf,mbsdf)
    N = len(clist)
    margins = np.zeros(N)
    for i in range(N):
        cons = clist[i]
        dat = cons['dat']
        J = dat.X.shape[0]
        prof, dprof = ModelFunctions.dSaleProfit_dr(np.repeat(dat.r_obs,J),dat,theta,cons['mbs'])
        margins[i] = prof[dat.lender_obs]/dprof[dat.lender_obs]
    
    keep_index = margins>(-1/theta.alpha_min)*1.5
    return keep_index


# def f_grad(x):
#     # cost_grad = np.sum(cost_data,0)
#     return -2*(np.dot(np.dot(cost_data,x),cost_data))/1e5

# def f_hess(x):
#     return np.zeros((len(x),len(x)))

# def test_deriv(x):
#     grad = np.zeros(len(x))
#     f0 = f_obj(x)
#     epsilon = 1e-6
#     for i in range(len(x)):
#         x_new = np.copy(x)
#         x_new[i] = x[i] + epsilon
#         f1 = f_obj(x_new)
#         grad[i] = (f1-f0)/epsilon
#     return grad

