import ModelTypes
import EquilibriumFunctions
import ModelFunctions
import Derivatives
import KernelFunctions
import numpy as np
import scipy as sp

### Consumer Data Subset Function
## Use the data frames, specification information, and consumer index i
## To create the relevant data objects for each specific consumer
## This is more about readability than speed, could be potential place to make code faster
# i - Consumer index, row number in loan level data
# theta - parameter object 
# cdf  - consumer/loan level data frame
# mdf  - market level data frame
# mbsdf - MBS coupon price data frame
## Output
# dat - consumer "Data" object
# mbs - MBS price interpolation object 

def consumer_subset(i,theta,cdf,mdf,mbsdf):
    ## Identify relevant market
    market = cdf[theta.market_spec][i]
    mkt_index = mdf[theta.market_spec]==market
    # Identify relevant time period
    time = int(cdf[theta.time_spec][i])


    # Subset appropriate data
    X_i = mdf.loc[mkt_index,theta.demand_spec].to_numpy() # Demand data
    W_i = mdf.loc[mkt_index,theta.cost_spec].to_numpy() # Lender cost data
    D_i = cdf.loc[i,theta.discount_spec].to_numpy() # Discount factor data
    Z_i = cdf.loc[i,theta.cons_spec].to_numpy() # Consumer cost data

    # Observed equilibrium outcomes 
    lender_obs = int(cdf.loc[i,theta.lender_spec]) # Chosen lender ID
    r_obs = cdf.loc[i,theta.rate_spec] # Observed interest rate
    # out = cdf.loc[i,theta.out_spec] # Outside option share (will be updated)
    out = int(cdf.loc[i,theta.out_index]) # Outside option share (will be updated)
    # Mortgage Backed Security interpolation
    prices = mbsdf.loc[time,theta.mbs_spec].to_numpy() # relevant MBS coupon prices
    mbs = ModelTypes.MBS_Func(theta.mbs_coupons,prices) # create MBS object

    # Create consumer data object
    dat = ModelTypes.Data(i,X_i,W_i,D_i,Z_i,lender_obs,r_obs,out) 

    return dat, mbs


###### Consumer Level Likelihood Evaluation Functions #######
# This section contains three functions, each of which evaluate the log likelihood of an individual observation. 
# The three functions evaluate likelihood only; likelihood & gradient; likelihood, gradient & hessian

### Consumer Likelihood Evaluation 
# Inputs
# theta - parameter object
# d - consumer data object
# m - MBS interpolation object
# Outputs
# ll_i - consumer contribution to log likelihood
# itr - number of interations to solve equilibrium 
def consumer_likelihood_eval(theta,d,m,model="base"):
    # Initial attempt to solve the equilibrium given current parameter guess
    alpha, r_eq, itr,success = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m,model=model)
    if not success: # Fast algorithm failed to converge, use robust algorithm
        # Robust equilibrium solution
        alpha, r_eq, itr = EquilibriumFunctions.solve_eq_r_robust(d.r_obs,d.lender_obs,d,theta,m,model=model)
        print("Robust Eq Method",d.i,itr) # An indication of the stability of the algorithm
 
    #Compute market shares
    q = ModelFunctions.market_shares(r_eq,alpha,d,theta)
    # Compute likelihood contribution
    ll_i = np.log(q[d.lender_obs]) - np.log(np.sum(q))
    # Compute macro likelihood contribution
    q0 = 1- np.sum(q) # Probability of selecting outside option
    # w = 1/(1-q0) # Model-implied probability of appearing in sample

    # Potential Violation: Paramaters imply that marginal cost exceed observed prices
    if itr==-1:
        ll_i = -1000 # large likelihood penalty if MC exceeds marginal cost
        q0 = 1.0
        # w = 1.0
        
    return ll_i, q0, alpha,itr

### Consumer Likelihood Evaluation with Gradient
# Inputs
# theta - parameter object
# d - consumer data object
# m - MBS interpolation object
# Outputs
# ll_i - consumer contribution to log likelihood
# dll_i - consumer contribution to gradient of the log likelihood
# itr - number of interations to solve equilibrium 

def consumer_likelihood_eval_gradient(theta,d,m,model="base"):
     # Initial attempt to solve the equilibrium given current parameter guess
    alpha, r_eq, itr,success = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m,model=model)
    if not success: # Fast algorithm failed to converge, use robust algorithm
        # Robust equilibrium solution
        alpha, r_eq, itr = EquilibriumFunctions.solve_eq_r_robust(d.r_obs,d.lender_obs,d,theta,m,model=model)
        print("Robust Eq Method",d.i,itr) # An indication of the stability of the algorithm


    # Compute market shares
    q,sb = ModelFunctions.market_shares(r_eq,alpha,d,theta,return_bound=True)
    # Compute likelihood contribution
    ll_i = np.log(q[d.lender_obs]) - np.log(np.sum(q))
    # Compute macro likelihood contribution
    q0 = 1- np.sum(q) # Probability of selecting outside option
    # w = 1/(1-q0) # Model-implied probability of appearing in sample

    
    # Potential Violation: Paramaters imply that marginal cost exceed observed prices
    if itr == -1:
        ll_i = -1000 # Large likelihood penalty
        # The rest of the output should be irrelevant
        q0 = 1.0
        # w = 1.0
        dll_i = np.zeros(len(theta.all()))  
        dq0 = np.zeros(len(theta.all()))
        da = np.zeros(len(theta.all()))

    # Potential Stability problem: one market share is close to 100% 
    elif itr == -2: 
        # Gradient is hard to evaluate and close to zero
        dll_i = np.zeros(len(theta.all())) 
        dq0 = np.zeros(len(theta.all()))
        da = np.zeros(len(theta.all()))
    else: # Everything is fine to compute gradient
        # Compute log share gradients
        dlogq, dq0, da = Derivatives.share_parameter_derivatives(r_eq,alpha,d,theta,m,model=model)
        # Compute likelihood and probability weight gradients
        dll_i = dlogq[:,d.lender_obs] + dq0/(1-q0)
        # dw = dq0/(1-q0)**2

    return ll_i, dll_i, q0, dq0, alpha, da, sb

### Consumer Likelihood Evaluation with Gradient and Hessian
# Inputs
# theta - parameter object
# d - consumer data object
# m - MBS interpolation object
# Outputs
# ll_i - consumer contribution to log likelihood
# dll_i - consumer contribution to gradient of the log likelihood
# d2ll_i - consumer contribution to hessian of the log likelihood
# itr - number of interations to solve equilibrium 
def consumer_likelihood_eval_hessian(theta,d,m,model="base"):
     # Initial attempt to solve the equilibrium given current parameter guess
    alpha, r_eq, itr,success,ab = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m,model=model,return_bound=True)
    if not success: # Fast algorithm failed to converge, use robust algorithm
        # Robust equilibrium solution
        ab = 0 
        alpha, r_eq, itr = EquilibriumFunctions.solve_eq_r_robust(d.r_obs,d.lender_obs,d,theta,m,model=model)
        print("Robust Eq Method",d.i,itr) # An indication of the stability of the algorithm

    # Compute market shares
    q,sb = ModelFunctions.market_shares(r_eq,alpha,d,theta,return_bound=True)
    # Compute likelihood contribution
    ll_i = np.log(q[d.lender_obs])- np.log(np.sum(q))
    # Compute macro likelihood contribution
    q0 = 1- np.sum(q) # Probability of selecting outside option
    # w = 1/(1-q0) # Model-implied probability of appearing in sample
    
    # Potential Violation: Paramaters imply that marginal cost exceed observed prices
    if itr == -1:
        ll_i = -1000 # Large likelihood penalty
        # The rest of the output should be irrelevant
        q0 = 1.0
        # w = 1.0
        dll_i = np.zeros(len(theta.all()))  
        dq0 = np.zeros(len(theta.all()))
        da = np.zeros(len(theta.all()))
        d2ll_i = np.zeros((len(theta.all()),len(theta.all())))
        d2q0 = np.zeros((len(theta.all()),len(theta.all())))
        d2a = np.zeros((len(theta.all()),len(theta.all())))
    # Potential Stability problem: one market share is close to 100% 
    elif itr == -2: 
        # Gradients and Hessians are hard to evaluate and close to zero
        dll_i = np.zeros(len(theta.all()))  
        dq0 = np.zeros(len(theta.all()))
        da = np.zeros(len(theta.all()))
        d2ll_i = np.zeros((len(theta.all()),len(theta.all())))
        d2q0 = np.zeros((len(theta.all()),len(theta.all())))
        d2a = np.zeros((len(theta.all()),len(theta.all())))
    else: # Everything is fine to evaluate gradient and hessian
        # Compute log share derivatives and second derivatives
        dlogq, d2logq, dq0,d2q0, da,d2a  = Derivatives.share_parameter_second_derivatives(r_eq,alpha,d,theta,m,model=model)
        # Compute likelihood and probability weight gradients
        dll_i = dlogq[:,d.lender_obs] + dq0/(1-q0)
        # dw = dq0/(1-q0)**2
        # Compute likelihood and probability weight hessians
        d2ll_i = d2logq[:,:,d.lender_obs] + d2q0/(1-q0) + np.outer(dq0,dq0)/(1-q0)**2
        # d2w = d2q0/(1-q0)**2 + 2*np.outer(dq0,dq0)/(1-q0)**3
         
    return ll_i, dll_i,d2ll_i, q0, dq0,d2q0, alpha, da,d2a,sb,ab

###### Functions to Evaluate Full Likelihood Function #####
## Similarly, three functions for objective, gradient, and hessian. 

### Evaluate Likelihood Function 
# Inputs
# x - Candidate parameter vector
# theta - parameter object (with specifications)
# cdf - consumer/loan level data frame
# mdf - market level data frame
# mbsdf - MBS coupon price data frame
# Outputs
# ll - log likelihood 
def evaluate_likelihood(x,theta,cdf,mdf,mbsdf,model="base"):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)
    # Initialize Aggregate Share Variables
    # pred_N = np.zeros(len(theta.out_share))
    # pred_N_out = np.zeros(len(theta.out_share))
    # mkt_Obs = np.zeros(len(theta.out_share))
    # q0_mkt = np.zeros(len(theta.out_share))

    # ll_market = np.zeros(len(theta.out_share))
    ll_micro = 0.0
    # Initialize log likelihood and iteration count
    ll = 0.0
    itr_avg = 0

    N = cdf.shape[0]
    alpha_list = np.zeros(cdf.shape[0])
    c_list_H = np.zeros(cdf.shape[0])
    c_list_S = np.zeros(cdf.shape[0])
    q0_list = np.zeros(cdf.shape[0])
    # Iterate over all consumers
    for i in range(cdf.shape[0]):
        # Subset data for consumer i
        dat, mbs = consumer_subset(i,theta,cdf,mdf,mbsdf)
        # Evaluate likelihood for consumer i 
        ll_i,q0_i,a_i,itr = consumer_likelihood_eval(theta,dat,mbs,model=model)
        # pred_N[dat.out] += w_i
        # pred_N_out[dat.out] += q0_i*w_i
        # q0_mkt[dat.out] +=q0_i
        # Add contribution to total likelihood
        # ll_market[dat.out] +=ll_i*w_i
        ll_micro += ll_i
        alpha_list[i] = a_i
        q0_list[i] = q0_i
        c_list_H[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZH)
        c_list_S[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZS)
        # Track iteration counts (for potential diagnostics)
        itr_avg += max(itr,0)
    # Combine Micro and Macro Likelihood Moments
    ll_macro = KernelFunctions.macro_likelihood(alpha_list,c_list_H,c_list_S,q0_list,theta)
    # pred_out_share = q0_mkt/mkt_Obs*(1-theta.out_share) + theta.out_share*(pred_N_out/pred_N)
    # ll_macro = np.sum(theta.N*(theta.out_share*np.log(pred_out_share)))# + (1-theta.out_share)*np.log(1-pred_out_share)))
    ll = ll_micro + ll_macro



    # Print and output likelihood value
    # print("Likelihood:",ll, "Macro Component:", ll_macro)
    return ll

### Evaluate Likelihood Function with Gradient
# Inputs
# x - Candidate parameter vector
# theta - parameter object (with specifications)
# cdf - consumer/loan level data frame
# mdf - market level data frame
# mbsdf - MBS coupon price data frame
# Outputs
# ll - log likelihood 
# dll - gradient of log likelihood 
def evaluate_likelihood_gradient(x,theta,cdf,mdf,mbsdf,model="base"):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)   
    K = len(theta.all())

    # Initialize Aggregate Share Variables
    # Necessary for matching outside option, and correcting selection into the sample
    # Predicted outside option takeup
    # pred_N_out = np.zeros(len(theta.out_share))
    # Adaptive weights to correct selection and derivatives
    # pred_N = np.zeros(len(theta.out_share))
    # dpred_N = np.zeros((len(theta.out_share),K))
    # mkt_Obs = np.zeros(len(theta.out_share))
    # q0_mkt = np.zeros(len(theta.out_share))
    # dq0_mkt = np.zeros((len(theta.out_share),K))
    
    # Likelihood value per market (defined by outside option) and gradient 
    # ll_market = np.zeros(len(theta.out_share))
    # dll_market = np.zeros((len(theta.out_share),K))

    ll_micro = 0.0
    dll_micro = np.zeros(K)

    
    alpha_list = np.zeros(cdf.shape[0])
    q0_list = np.zeros(cdf.shape[0])
    c_list_H = np.zeros(cdf.shape[0])
    c_list_S = np.zeros(cdf.shape[0])

    dalpha_list = np.zeros((cdf.shape[0],K))
    dq0_list = np.zeros((cdf.shape[0],K))

    # Iterate over all consumers
    for i in range(cdf.shape[0]):
        # Subset data for consumer i
        dat, mbs = consumer_subset(i,theta,cdf,mdf,mbsdf)
        # Evaluate likelihood and gradient for consumer i 
        ll_i,dll_i,q0_i,dq0_i,a_i,da_i,sb_i = consumer_likelihood_eval_gradient(theta,dat,mbs,model=model)
        # Add outside option, probability weights, and derivatives
        # pred_N[dat.out] += w_i
        # pred_N_out[dat.out] += q0_i*w_i
        # dpred_N[dat.out,:] += dw_i
        # q0_mkt[dat.out] += q0_i
        # dq0_mkt[dat.out,:] += dq0_i
        # mkt_Obs[dat.out] +=1
        # Add contribution to total likelihood, gradient and hessian (by outside option market)
        # Derivatives need to account for the fact that the likelihood is weighted
        # ll_market[dat.out] +=ll_i*w_i
        # dll_market[dat.out,:] += dll_i*w_i + ll_i*dw_i

        ll_micro += ll_i
        dll_micro += dll_i*(1-sb_i)

        alpha_list[i] = a_i
        q0_list[i] = q0_i
        c_list_H[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZH)
        c_list_S[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZS)

        dalpha_list[i,:] = da_i
        dq0_list[i,:] = dq0_i*(1-sb_i)

    
    # Compute Macro Likelihood Component and Gradient
    # pred_out_share = pred_N_out/pred_N # Predicted outside option share
    # pred_out_share = q0_mkt/mkt_Obs*(1-theta.out_share) + theta.out_share*(pred_N_out/pred_N)
    # ll_macro = np.sum(theta.N*(theta.out_share*np.log(pred_out_share)))
    # # Initialize objects to hold gradient
    # dll_macro = np.zeros(K)
    # # Sum across the outside option markets
    # for i in range(len(theta.out_share)):
    #     # Gradient of the log-weighted-outside option
    #     # d_log_out = dpred_N[i,:]*(1/pred_N_out[i] - 1/pred_N[i])
    #     d_log_out = (1/pred_out_share[i])*(1-theta.out_share[i])*dq0_mkt[i,:]/mkt_Obs[i] + theta.out_share[i]*dpred_N[i,:]/pred_N[i]*(1 - pred_N_out[i]/pred_N[i])
    #     # Weight by actual population size in macro component
    #     dll_macro += theta.N[i]*(theta.out_share[i]*d_log_out)

    # Compute total log likelihood
    ll_macro, dll_macro = KernelFunctions.macro_likelihood_grad(alpha_list,c_list_H,c_list_S,q0_list,
                                                                dalpha_list,dq0_list,theta)
    # Micro-likelihood is renormalized by sum of the weights, then weighted by actual population
    # ll = np.sum(theta.N*(1-theta.out_share)*ll_market/pred_N) + ll_macro
    ll = ll_micro + ll_macro
    
    # Initialize log likelihood gradient 
    # dll = dll_macro # Start with macro component
    # # Sum across outside option markets
    # for i in range(len(theta.out_share)):
    #     # Gradient of the micro-log likelihood
    #     dll += theta.N[i]*(1-theta.out_share[i])*(dll_market[i,:]/pred_N[i] +\
    #             - dpred_N[i,:]*ll_market[i]/pred_N[i]**2) 

    dll = dll_micro + dll_macro
        
    # Print and output likelihood value
    # print("Likelihood:",ll, "Macro Component:", ll_macro)
    return ll, dll

### Evaluate Likelihood Function with Gradient and Hessian
# Inputs
# x - Candidate parameter vector
# theta - parameter object (with specifications)
# cdf - consumer/loan level data frame
# mdf - market level data frame
# mbsdf - MBS coupon price data frame
# Outputs
# ll - log likelihood 
# dll - gradient of log likelihood 
# d2ll - hessian of log likelihood 
def evaluate_likelihood_hessian(x,theta,cdf,mdf,mbsdf,model="base",**kwargs):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)
    K = len(theta.all())

    # Initialize Aggregate Share Variables
    # Necessary for matching outside option, and correcting selection into the sample
    # Predicted outside option takeup
    # pred_N_out = np.zeros(len(theta.out_share))
    # Adaptive weights to correct selection and derivatives
    # pred_N = np.zeros(len(theta.out_share))
    # dpred_N = np.zeros((len(theta.out_share),K))
    # d2pred_N = np.zeros((len(theta.out_share),K,K))
    # mkt_Obs = np.zeros(len(theta.out_share))
    # q0_mkt = np.zeros(len(theta.out_share))
    # dq0_mkt = np.zeros((len(theta.out_share),K))
    # d2q0_mkt = np.zeros((len(theta.out_share),K))
    # Likelihood value per market (defined by outside option) and gradient 
    # ll_market = np.zeros(len(theta.out_share))
    # dll_market = np.zeros((len(theta.out_share),K))
    # d2ll_market = np.zeros((len(theta.out_share),K,K))

    ll_micro = 0.0
    dll_micro = np.zeros(K)
    d2ll_micro = np.zeros((K,K))

    
    alpha_list = np.zeros(cdf.shape[0])
    q0_list = np.zeros(cdf.shape[0])
    c_list_H = np.zeros(cdf.shape[0])
    c_list_S = np.zeros(cdf.shape[0])

    dalpha_list = np.zeros((cdf.shape[0],K))
    dq0_list = np.zeros((cdf.shape[0],K))
    
    d2q0_list = np.zeros((cdf.shape[0],K,K))
    d2alpha_list = np.zeros((cdf.shape[0],K,K))

    
    # Iterate over all consumers
    for i in range(cdf.shape[0]):
        # Subset data for consumer i
        dat, mbs = consumer_subset(i,theta,cdf,mdf,mbsdf)
        # Evaluate likelihood, gradient, and hessian for consumer i 
        ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i,d2a_i,sb,ab  = consumer_likelihood_eval_hessian(theta,dat,mbs,model=model)
        # Add outside option, probability weights, and derivatives
        # pred_N_out[dat.out] += q0_i*w_i
        # pred_N[dat.out] += w_i
        # dpred_N[dat.out,:] += dw_i
        # d2pred_N[dat.out,:,:] += d2w_i
        # q0_mkt[dat.out] += q0_i
        # dq0_mkt[dat.out,:] += dq0_i
        # d2q0_mkt[dat.out,:,:] += dq0_i
        # mkt_Obs[dat.out] +=1
        # Add contribution to total likelihood, gradient and hessian (by outside option market)
        # Derivatives need to account for the fact that the likelihood is weighted
        # ll_market[dat.out] +=ll_i*w_i
        # dll_market[dat.out,:] += dll_i*w_i + ll_i*dw_i
        # d2ll_market[dat.out,:,:] += d2ll_i*w_i + ll_i*d2w_i + np.outer(dll_i,dw_i) + np.outer(dw_i,dll_i)

        ll_micro += ll_i
        dll_micro += dll_i*(1-sb)
        d2ll_micro += d2ll_i*(1-sb)

        alpha_list[i] = a_i
        q0_list[i] = q0_i
        c_list_H[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZH)
        c_list_S[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZS)

        dalpha_list[i,:] = da_i
        dq0_list[i,:] = dq0_i*(1-sb)

        d2q0_list[i,:,:] = d2q0_i*(1-sb)
        d2alpha_list[i,:,:] = d2a_i

    
    # Compute Macro Likelihood Component and Gradient
    # pred_out_share = pred_N_out/pred_N # Predicted outside option share
    # pred_out_share = q0_mkt/mkt_Obs*(1-theta.out_share) + theta.out_share*(pred_N_out/pred_N)
    # ll_macro = np.sum(theta.N*(theta.out_share*np.log(pred_out_share)))
    # # Initialize objects to hold gradient and hessian
    # dll_macro = np.zeros(K)
    # d2ll_macro = np.zeros((K,K))
    # # Sum across the outside option markets
    # for i in range(len(theta.out_share)):
    #     # Gradient of the log-weighted-outside option
    #     # d_log_out = dpred_N[i,:]*(1/pred_N_out[i] - 1/pred_N[i])
    #     d_log_out = (1/pred_out_share[i])*(1-theta.out_share[i])*dq0_mkt[i,:]/mkt_Obs[i] + theta.out_share[i]*dpred_N[i,:]/pred_N[i]*(1 - pred_N_out[i]/pred_N[i])
    #     # Weight by actual population size in macro component
    #     dll_macro += theta.N[i]*(theta.out_share[i]*d_log_out)

    #     # Hessian of log-weighted-outside option 
    #     d2_log_out = d2pred_N[i,:,:]*(1/pred_N_out[i] - 1/pred_N[i]) - np.outer(dpred_N[i,:],dpred_N[i,:])*(1/pred_N_out[i]**2 - 1/pred_N[i]**2) 
    #     # Weight by actual population size in macro component
    #     d2ll_macro += theta.N[i]*(theta.out_share[i]*d2_log_out)

    # Compute total log likelihood
    ll_macro, dll_macro, d2ll_macro,BFGS_next = KernelFunctions.macro_likelihood_hess(alpha_list,c_list_H,c_list_S,q0_list,
                                                                                      dalpha_list,dq0_list,d2q0_list,theta,BFGS_prior=kwargs.get("BFGS_prior"))
    # Micro-likelihood is renormalized by sum of the weights, then weighted by actual population
    # ll = np.sum(theta.N*(1-theta.out_share)*ll_market/pred_N) + ll_macro
    ll = ll_micro + ll_macro
    # Initialize log likelihood gradient, hessian 
    # dll = dll_macro # Start with macro component
    # d2ll = d2ll_macro # Start with macro component
    # # Sum across outside option markets
    # for i in range(len(theta.out_share)):
    #     # Gradient of the micro-log likelihood
    #     dll += theta.N[i]*(1-theta.out_share[i])*(dll_market[i,:]/pred_N[i] +\
    #             - dpred_N[i,:]*ll_market[i]/pred_N[i]**2) 
    #     # Hessian of the micro-log likeliood
    #     d2ll += theta.N[i]*(1-theta.out_share[i])*(
    #         d2ll_market[i,:,:]/pred_N[i] +\
    #         -np.outer(dll_market[i,:],dpred_N[i,:])/pred_N[i]**2 +\
    #         -np.outer(dpred_N[i,:],dll_market[i,:])/pred_N[i]**2 + \
    #         -d2pred_N[i,:,:]*ll_market[i]/pred_N[i]**2 +\
    #         2*np.outer(dpred_N[i,:],dpred_N[i,:])*ll_market[i]/pred_N[i]**3
    #         ) 

    dll = dll_micro + dll_macro
    d2ll = d2ll_micro + d2ll_macro

    # Print and output likelihood value
    # print("Likelihood:",ll, "Macro ll component:", ll_macro)
    return ll, dll, d2ll, BFGS_next

##### Optimization Functions to Maximize Likelihood and Estimate Parameters #######

### Newton-Raphson Estimation 
## Use Newton-Raphson and the analytical hessian matrix to maximize likelihood function 
# Inputs 
# x - starting guess for parameter vector
# theta - parameter object with specification info
# cdf - consumer/loan level data frame
# mdf - market level data frame
# mbdsf - MBS coupon price data frame
# Outputs
# ll_k - maximized likelihood
# x - estimated parameter vector 
def estimate_NR(x,theta,cdf,mdf,mbsdf):
    # Testing Tool: A index of parameters to estimate while holding others constant
    # This can help identification. range(0,len(x)) will estimate all parameters 
    test_index = theta.beta_x_ind
    # Set candidate vector in parameter object
    theta.set_demand(x)

    # Initialize "new" candidate parameter vector
    x_new = np.copy(x)
    # Intitial evaluation of likelihood, gradient (f_k), and hessian (B_k)
    ll_k, f_k, B_k, bfgs_mem = evaluate_likelihood_hessian(x,theta,cdf,mdf,mbsdf)
    # Translate Hessian into a negative definite matrix for best ascent 
    # Also raises a warning when Hessian is not negitive definite, bad sign if it happens near convergence
    B_k = enforceNegDef(B_k[np.ix_(test_index,test_index)])

    # Initialize error and iteration count
    err = 1
    itr = 0
    # Iterate while error exceeds tolerance
    while err>1e-6 and itr<1000:
        # Compute newton step
        p_k = -np.dot(np.linalg.inv(B_k),f_k[test_index])
        # Initial line search value: full newton step
        alpha = 1.0
        # Bound the step size to be one in order to avoid model crashes on odd parameters
        largest_step = np.max(np.abs(p_k))
        alpha = np.minimum(10.0/largest_step,1.0)

        # Compute bounded step
        s_k = alpha*p_k
        # Update potential parameter vector
        x_new[test_index] = x[test_index] + s_k

        # Recompute likelihood, gradient and hessian
        ll_new, f_new, B_new, bfgs_mem_new = evaluate_likelihood_hessian(x_new,theta,cdf,mdf,mbsdf,BFGS_prior=bfgs_mem)
        
        # If the initial step leads to a much lower likelihood value
        # shrink the step size to search for a better step.
        line_search = 0  # Initialize line search flag
        while ll_new<ll_k*1.0005: # Allow a small step in the wrong direction
            line_search = 1 # Save that a line search happened
            alpha = alpha/2 # Shrink the step size
            print("Line Search Step Size:",alpha) # Print step size for search
            s_k = alpha*p_k # Compute new step size
            x_new[test_index] = x[test_index] + s_k # Update new candidate parameter vector

            ll_new = evaluate_likelihood(x_new,theta,cdf,mdf,mbsdf) # Check new value of the likelihood function

        # Update parameter vector after successful step
        x = np.copy(x_new)
        theta.set_demand(x)  
        
        # If no line search was done, update likelihood, gradient and hessian
        if line_search==0:
            ll_k = np.copy(ll_new)
            f_k = np.copy(f_new)
            B_k = np.copy(B_new)
            bfgs_mem = bfgs_mem_new
        # If there was a line search, need to evaluate the hessian again
        else:
            ll_k, f_k, B_k, bfgs_mem = evaluate_likelihood_hessian(x,theta,cdf,mdf,mbsdf,BFGS_prior=bfgs_mem)

        # Check that the hessian is negative definite and enforce it if necessary
        B_k = enforceNegDef(B_k[np.ix_(test_index,test_index)])

        # Evaluate the sum squared error of the gradient of the likelihood function
        err = np.sqrt(np.dot(f_k[test_index],f_k[test_index]))
        # Update iteration count and print error value
        itr+=1 
        print("#### Iteration",itr, "Error", err)

    # Print completion and output likelihood and estimated parameter vector
    print("Completed with Error", err)
    return ll_k, x

### Gradient Ascent Estimation 
## Use gradient only to maximize likelihood function 
## This method is very robust but very slow
## Useful for moving into a reasonable parameter space before applying NR method
# Inputs 
# x - starting guess for parameter vector
# theta - parameter object with specification info
# cdf - consumer/loan level data frame
# mdf - market level data frame
# mbdsf - MBS coupon price data frame
# Outputs
# ll_k - maximized likelihood
# x - estimated parameter vector 
def estimate_GA(x,theta,cdf,mdf,mbsdf):
    # Testing Tool: A index of parameters to estimate while holding others constant
    # This can help identification. range(0,len(x)) will estimate all parameters 
    test_index = theta.beta_x_ind
    # Set candidate vector in parameter object
    theta.set_demand(x)

    # Initialize "new" candidate parameter vector
    x_test = np.copy(x)


    # Initial evaluation of likelihood (ll_k) and gradient (g_k)
    ll_k, g_k = evaluate_likelihood_gradient(x,theta,cdf,mdf,mbsdf)

    #Initial step size
    alpha = 1e-6/np.max(np.abs(g_k[test_index]))

    # Initial evaluation of error and iteration count
    err = 1000
    itr = 0
    # Iterate until error becomes small or a small, maximum iteration count 
    while (err>1e-6) & (itr<30):
        # Asceding step is the step size times the gradient
        s_k = alpha*g_k[test_index]
        # Update candidate new parameter vector
        x_test[test_index] = x[test_index] + s_k


        # Evaluation of likelihood and gradient at new candidate vector
        ll_test, g_test = evaluate_likelihood_gradient(x_test,theta,cdf,mdf,mbsdf)
        
        # Theoretically, a small enough step should always increase the likelihood
        # If likelihood is lower at a given step, shrink until it is an ascending step
        while ll_test<ll_k:

            alpha = alpha/10 # Shrink step size
            print("Line Search Step",alpha) # Print the linesearch step size
            s_k = alpha*g_k[test_index] # New gradient ascent step
            x_test[test_index] = x[test_index] + s_k # Recompute a new candidate parameter vector

            # Re-evaluate likelihood and gradient
            ll_test,g_test = evaluate_likelihood_gradient(x_test,theta,cdf,mdf,mbsdf)

        # Update parameter vector after successful step
        x = np.copy(x_test)
        theta.set_demand(x)  

        # Update step size based on the change in the step size and gradient
        y_k = g_test[test_index] - g_k[test_index]
        # This is a very simple, quasi-newton approximation 
        alpha = np.abs(np.dot(s_k,s_k)/np.dot(s_k,y_k))

        # Update gradient and likelihood function
        g_k = np.copy(g_test)
        ll_k = np.copy(ll_test)
        
        # Compute the sum squared error of the likelihood gradient
        err = np.sqrt(np.dot(g_k[test_index],g_k[test_index]))
        # Update iteration count and print error value
        itr+=1 
        print("#### Iteration",itr, "Error", err)
   
    # Print completion and output likelihood and estimated parameter vector
    print("Completed with Error", err)
    return ll_k, x

### Enforce Negative Definite Matrix
# Takes a symmetric matrix and returns a negative definite version based on the eigen-value decomposition
# Useful for making sure that a newtown step still valid when hessian is not negative definite 
# See below reference 
#http://web.stanford.edu/class/cme304/docs/newton-type-methods.pdf
# Inputs
# H - a symmetric matrix
# Output
# H_new - a symmetric, negative defininte matrix with the same eigen vectors as H. 
def enforceNegDef(H):
    # First, attempt cholesky decomposition 
    # This is a trick to quickly verify if H is negative definite
    try:
        check = np.linalg.cholesky(-H)
        return H # If already negative definite, return H without doing anything
    # If cholesky decomposition returns an error, then H is not negative definite
    # Proceed with computing "direction of ascent"
    except:
        # Print a warning 
        print ("Hessian Is Not Concave: Compute Direction of Ascent") 
        # Compute eigen values (e) and eigen vectors (v)
        e, v = np.linalg.eig(H)
        # Output the largest eigen value (i.e. how far from negative definite is the matrix?)
        max_eig_val = max(e)
        print("Maximum Eigenvalue:", max_eig_val)
        # Construct a diagonal matrix of all negative eigen values 
        Lambda =-np.diag(np.abs(e))
        # Re-construct H from the eigen vectors and the negative eigen values
        H_new = np.dot(np.dot(v,Lambda),np.transpose(v))
        # Return newly constructed negative definite matrix
        return H_new
    

##### Wrapper for Likelihood Maximization #####
## A wrapper that compiles some steps in the maximization alogrithm.
# Inputs 
# x - starting guess for parameter vector
# theta - parameter object with specification info
# cdf - consumer/loan level data frame
# mdf - market level data frame
# mbdsf - MBS coupon price data frame
# Outputs
# f_val - maximized likelihood
# res - estimated parameter vector  
def maximize_likelihood(x,theta,cdf,mdf,mbsdf):
    print("Gradient Ascent Stage - Find Better Starting Parameter")
    # f_val, pre_start = estimate_GA(x,theta,cdf,mdf,mbsdf)
    print("Newton - Raphson Stage - Maximize Likelihood Function")
    f_val, res = estimate_NR(x,theta,cdf,mdf,mbsdf)
    return f_val, res


def py_optim(x,theta,cdf,mdf,mbsdf,model="base"):

    def f_obj(x):
        val, g = evaluate_likelihood_gradient(x,theta,cdf,mdf,mbsdf,model=model)
        err = np.sqrt(np.dot(g,g))
        print("Gradient Size:", err)
        return -val, -g
    
    def f_hess(x):
        ll, g,h = evaluate_likelihood_hessian(x,theta,cdf,mdf,mbsdf,model=model)
        return -h
    

    res = sp.optimize.minimize(f_obj,x,method="Newton-CG",jac=True,hess=f_hess,
                               options = {'xtol':1e-15})

  
    return res

def predicted_elasticity(x,theta,cdf,mdf,mbsdf,model="base"):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)
    # Initialize Aggregate Share Variables
    # pred_N = np.zeros(len(theta.out_share))
    # pred_N_out = np.zeros(len(theta.out_share))
    # mkt_Obs = np.zeros(len(theta.out_share))
    # q0_mkt = np.zeros(len(theta.out_share))

    alpha_list = np.zeros(cdf.shape[0])
    elas = np.zeros(cdf.shape[0])
    eq_flag = np.zeros(cdf.shape[0])
    # Iterate over all consumers
    for i in range(cdf.shape[0]):
        # Subset data for consumer i
        dat, mbs = consumer_subset(i,theta,cdf,mdf,mbsdf)
        # Evaluate likelihood for consumer i 
        ll_i,q0_i,a_i,itr = consumer_likelihood_eval(theta,dat,mbs,model=model)
        r, itr,flag= EquilibriumFunctions.solve_eq_optim(a_i,dat,theta,mbs)
        q = ModelFunctions.market_shares(r,a_i,dat,theta)
        alpha_list[i] = a_i
        elas[i] = a_i*dat.r_obs*(1-q[dat.lender_obs])
        eq_flag[i] = flag
    
    return alpha_list,elas,eq_flag