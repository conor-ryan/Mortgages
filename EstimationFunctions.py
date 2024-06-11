import ModelTypes
import EquilibriumFunctions
import ModelFunctions
import Derivatives
import KernelFunctions
import ParallelFunctions
import TestConditionalFunctions
import numpy as np
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

    # Skip consumers that are below a certain margin threshold
    prof, dprof = ModelFunctions.dSaleProfit_dr(np.repeat(r_obs,X_i.shape[0]),dat,theta,mbs)
    obs_margin = prof[lender_obs]/dprof[lender_obs]
    dat.skip = obs_margin<(-1/theta.alpha_min)

    return dat, mbs


### Consumer Data List Function
## This function pre-processes all of the consumer inputs into a list
# theta - parameter object 
# cdf  - consumer/loan level data frame
# mdf  - market level data frame
# mbsdf - MBS coupon price data frame
## Output
# A list of named lists where each item has two entries:
# --- dat - consumer "Data" object
# --- mbs - MBS price interpolation object 

def consumer_object_list(theta,cdf,mdf,mbsdf):
    consumer_list = list()
    skip_tracking = np.zeros(cdf.shape[0])
    for i in range(cdf.shape[0]):
        dat,mbs = consumer_subset(i,theta,cdf,mdf,mbsdf)
        consumer_list.append({'dat':dat,'mbs':mbs})
        skip_tracking[i] = dat.skip
    theta.construct_out_index(cdf)
    print("Fraction Dropped on Margin Threshold",np.mean(skip_tracking))
    return consumer_list


###### Function for skipped observations #######
def skipped_consumer_likelihood(theta,d,m):
        ll_i = 0.0
        dll_i = np.zeros(len(theta.all()))
        d2ll_i = 0.0
        # q0 = 1-1e-10
        q0 = 1-1e-10
        dq0 = np.zeros(len(theta.all()))
        da = np.zeros(len(theta.all()))
        d2q0 = np.zeros((len(theta.all()),len(theta.all())))
        d2a = np.zeros((len(theta.all()),len(theta.all())))
        prof, dprof = ModelFunctions.dSaleProfit_dr(np.repeat(d.r_obs,d.X.shape[0]),d,theta,m)
        alpha = -dprof[d.lender_obs]/prof[d.lender_obs]

        if alpha<-5000 or dprof[d.lender_obs]<0:
            alpha = -5000

        return ll_i, dll_i,d2ll_i, q0, dq0,d2q0, alpha, da,d2a,0,0

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
    if d.skip:
        ll_i, dll_i,d2ll_i, q0, dq0,d2q0, alpha, da,d2a,sb,ab = skipped_consumer_likelihood(theta,d,m)
        return ll_i, q0, alpha,1
    
    alpha, r_eq, itr,success = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m,model=model)
    if not success: # Fast algorithm failed to converge, use robust algorithm
        # Robust equilibrium solution
        alpha, r_eq, itr = EquilibriumFunctions.solve_eq_r_robust(d.r_obs,d.lender_obs,d,theta,m,model=model)
        print("Robust Eq Method",d.i,itr) # An indication of the stability of the algorithm
 
    #Compute market shares
    q,sb = ModelFunctions.market_shares(r_eq,alpha,d,theta,return_bound=True)
    ll_cond = ModelFunctions.conditional_likelihood(r_eq,alpha,d,theta)
    # Compute likelihood contribution
    if sb==0:
        ll_i = ll_cond[d.lender_obs]
    else:
        # ll_i = np.log(1/len(q))
        ll_i = ll_cond[d.lender_obs]
    # ll_i = np.log(q[d.lender_obs]) - np.log(np.sum(q))
    # Compute macro likelihood contribution
    q0 = 1- np.sum(q) # Probability of selecting outside option

    # Potential Violation: Paramaters imply that marginal cost exceed observed prices
    if itr==-1:
        ll_i = -1000 # large likelihood penalty if MC exceeds marginal cost
        q0 = 1.0
        
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
    if d.skip:
        ll_i, dll_i,d2ll_i, q0, dq0,d2q0, alpha, da,d2a,sb,ab = skipped_consumer_likelihood(theta,d,m)
        return ll_i, dll_i, q0, dq0, alpha, da, sb

     # Initial attempt to solve the equilibrium given current parameter guess
    alpha, r_eq, itr,success = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m,model=model)
    if not success: # Fast algorithm failed to converge, use robust algorithm
        # Robust equilibrium solution
        alpha, r_eq, itr = EquilibriumFunctions.solve_eq_r_robust(d.r_obs,d.lender_obs,d,theta,m,model=model)
        print("Robust Eq Method",d.i,itr) # An indication of the stability of the algorithm


    # Compute market shares
    q,sb = ModelFunctions.market_shares(r_eq,alpha,d,theta,return_bound=True)
    ll_cond = ModelFunctions.conditional_likelihood(r_eq,alpha,d,theta)
    # Compute likelihood contribution
    if sb==0:
        ll_i = ll_cond[d.lender_obs]
    else:
        # ll_i = np.log(1/len(q))
        ll_i = ll_cond[d.lender_obs]
    # ll_i = np.log(q[d.lender_obs]) - np.log(np.sum(q))
    # Compute macro likelihood contribution
    q0 = 1- np.sum(q) # Probability of selecting outside option

    
    # Potential Violation: Paramaters imply that marginal cost exceed observed prices
    if itr == -1:
        ll_i = -1000 # Large likelihood penalty
        # The rest of the output should be irrelevant
        q0 = 1.0
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
        dlogq_uncond, dq0, da = Derivatives.share_parameter_derivatives(r_eq,alpha,d,theta,m,model=model)
        dlogq = TestConditionalFunctions.conditional_parameter_derivatives(r_eq,alpha,d,theta,m,model=model)
        # Compute likelihood and probability weight gradients
        dll_i = dlogq[:,d.lender_obs] #+ dq0/(1-q0)
        dll_i = dll_i#*(1-sb)
        dq0 = dq0*(1-sb)

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
    if d.skip:
        ll_i, dll_i,d2ll_i, q0, dq0,d2q0, alpha, da,d2a,sb,ab = skipped_consumer_likelihood(theta,d,m)
        return ll_i, dll_i,d2ll_i, q0, dq0,d2q0, alpha, da,d2a,sb, ab

     # Initial attempt to solve the equilibrium given current parameter guess
    alpha, r_eq, itr,success,ab = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m,model=model,return_bound=True)
    if not success: # Fast algorithm failed to converge, use robust algorithm
        # Robust equilibrium solution
        ab = 0 
        alpha, r_eq, itr = EquilibriumFunctions.solve_eq_r_robust(d.r_obs,d.lender_obs,d,theta,m,model=model)
        print("Robust Eq Method",d.i,itr) # An indication of the stability of the algorithm

    # Compute market shares
    q,sb = ModelFunctions.market_shares(r_eq,alpha,d,theta,return_bound=True)
    ll_cond = ModelFunctions.conditional_likelihood(r_eq,alpha,d,theta)
    # Compute likelihood contribution
    if sb==0:
        ll_i = ll_cond[d.lender_obs]
    else:
        # ll_i = np.log(1/len(q))
        ll_i = ll_cond[d.lender_obs]
    # ll_i = np.log(q[d.lender_obs]) - np.log(np.sum(q))
    # Compute macro likelihood contribution
    q0 = 1- np.sum(q) # Probability of selecting outside option
    
    # Potential Violation: Paramaters imply that marginal cost exceed observed prices
    if itr == -1:
        ll_i = -1000 # Large likelihood penalty
        # The rest of the output should be irrelevant
        q0 = 1.0
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
        dlogq_uncond, d2logq_uncond, dq0,d2q0, da,d2a  = Derivatives.share_parameter_second_derivatives(r_eq,alpha,d,theta,m,model=model)
        dlogq, d2logq = TestConditionalFunctions.conditional_parameter_second_derivatives(r_eq,alpha,d,theta,m,model=model)
        # Compute likelihood and probability weight gradients
        dll_i = dlogq[:,d.lender_obs] #+ dq0/(1-q0)
        # Compute likelihood and probability weight hessians
        d2ll_i = d2logq[:,:,d.lender_obs] #+ d2q0/(1-q0) + np.outer(dq0,dq0)/(1-q0)**2
        dll_i = dll_i#*(1-sb)
        d2ll_i = d2ll_i#*(1-sb)
        dq0 = dq0*(1-sb)
        d2q0 = d2q0*(1-sb)
        
         
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
def evaluate_likelihood(x,theta,clist,parallel=False,num_workers=0,model="base"):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)

    # Parallel estimation can only estimate the base model at the moment
    if parallel and model!="base":
        raise Exception("WARNING: non-base model specified in parallel. Parallel can only estimate base model.")
    
    # Initialize log likelihood tracking variables
    N = len(clist)
    ll_micro = 0.0
    alpha_list = np.zeros(N)
    c_list_H = np.zeros(N)
    c_list_S = np.zeros(N)
    q0_list = np.zeros(N)
    skipped_list = np.zeros(N)


    ## Evaluate each consumer in parallel
    if parallel: 
        if num_workers<2:
            raise Exception("Number of workers not set (or less than one) in parallel estimation")
        args = [(theta,c_val['dat'],c_val['mbs']) for c_val in clist]
        res = ParallelFunctions.eval_map_likelihood(args,num_workers)

    # Iterate over all consumers to compute likelihood
    for i in range(N):
        # Get consumer level results
        if parallel:
            # Unpack previously estimated results
            dat= clist[i]['dat']
            ll_i,q0_i,a_i,itr = res[i]
        else:
            # Evaluate likelihood for consumer i 
            dat = clist[i]['dat']
            mbs = clist[i]['mbs']       
            ll_i,q0_i,a_i,itr = consumer_likelihood_eval(theta,dat,mbs,model=model)
        
        ll_micro += ll_i
        ## Collect consumer level info for implied outside option share
        alpha_list[i] = a_i
        q0_list[i] = q0_i
        c_list_H[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZH)
        c_list_S[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZS)
        skipped_list[i] = dat.skip

    # Combine Micro and Macro Likelihood Moments
    ll_macro = KernelFunctions.macro_likelihood(alpha_list,c_list_H,c_list_S,q0_list,theta)
    ll = ll_micro + ll_macro
    # Print and output likelihood value
    # print("Likelihood:",ll, "Macro Component:", ll_macro)
    return ll/N

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
def evaluate_likelihood_gradient(x,theta,clist,parallel=False,num_workers=0,model="base"):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)
    K = len(theta.all())
    N = len(clist)

    # Parallel estimation can only estimate the base model at the moment
    if parallel and model!="base":
        raise Exception("WARNING: non-base model specified in parallel. Parallel can only estimate base model.")
    

    # Initialize log likelihood tracking variables
    ll_micro = 0.0
    dll_micro = np.zeros(K)

    
    alpha_list = np.zeros(N)
    q0_list = np.zeros(N)
    c_list_H = np.zeros(N)
    c_list_S = np.zeros(N)

    dalpha_list = np.zeros((N,K))
    dq0_list = np.zeros((N,K))
    skipped_list = np.zeros(N)


    ## Evaluate each consumer in parallel
    if parallel: 
        if num_workers<2:
            raise Exception("Number of workers not set (or less than one) in parallel estimation")
        args = [(theta,c_val['dat'],c_val['mbs']) for c_val in clist]
        res = ParallelFunctions.eval_map_likelihood_gradient(args,num_workers)

    # Iterate over all consumers to compute likelihood
    for i in range(N):
        # Get consumer level results
        if parallel:
            # Unpack previously estimated results
            dat= clist[i]['dat']
            ll_i,dll_i,q0_i,dq0_i,a_i,da_i,sb_i = res[i]
        else:
            # Evaluate likelihood for consumer i 
            dat = clist[i]['dat']
            mbs = clist[i]['mbs']       
            ll_i,dll_i,q0_i,dq0_i,a_i,da_i,sb_i = consumer_likelihood_eval_gradient(theta,dat,mbs,model=model)

        ll_micro += ll_i
        dll_micro += dll_i

        ## Collect consumer level info for implied outside option share
        alpha_list[i] = a_i
        q0_list[i] = q0_i
        c_list_H[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZH)
        c_list_S[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZS)

        dalpha_list[i,:] = da_i
        dq0_list[i,:] = dq0_i
        skipped_list[i] = dat.skip
    
    # Compute Macro Likelihood Component and Gradient
    ll_macro, dll_macro = KernelFunctions.macro_likelihood_grad(alpha_list,c_list_H,c_list_S,q0_list,
                                                                dalpha_list,dq0_list,theta)

    ll = ll_micro + ll_macro
    dll = dll_micro + dll_macro

    # Print and output likelihood value
    # print("Likelihood:",ll, "Macro Component:", ll_macro)
    return ll/N, dll/N

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
def evaluate_likelihood_hessian(x,theta,clist,parallel=False,num_workers=0,model="base",**kwargs):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)
    K = len(theta.all())
    N = len(clist)

    # Parallel estimation can only estimate the base model at the moment
    if parallel and model!="base":
        raise Exception("WARNING: non-base model specified in parallel. Parallel can only estimate base model.")
    

    sbound_mean = np.zeros(N)
    abound_mean = np.zeros(N)
    skipped_list = np.zeros(N)

    # Initialize log likelihood tracking variables
    ll_micro = 0.0
    dll_micro = np.zeros(K)
    d2ll_micro = np.zeros((K,K))

    
    ## Collect consumer level info for implied outside option share
    alpha_list = np.zeros(N)
    q0_list = np.zeros(N)
    c_list_H = np.zeros(N)
    c_list_S = np.zeros(N)

    dalpha_list = np.zeros((N,K))
    dq0_list = np.zeros((N,K))
    
    d2q0_list = np.zeros((N,K,K))
    
        ## Evaluate each consumer in parallel
    if parallel: 
        if num_workers<2:
            raise Exception("Number of workers not set (or less than one) in parallel estimation")
        args = [(theta,c_val['dat'],c_val['mbs']) for c_val in clist]
        res = ParallelFunctions.eval_map_likelihood_hessian(args,num_workers)

    # Iterate over all consumers to compute likelihood
    for i in range(N):
        # Get consumer level results
        if parallel:
            # Unpack previously estimated results
            dat= clist[i]['dat']
            ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i,d2a_i,sb_i,ab_i = res[i]
        else:
            # Evaluate likelihood for consumer i 
            dat = clist[i]['dat']
            mbs = clist[i]['mbs']       
            ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i,d2a_i,sb_i,ab_i  = consumer_likelihood_eval_hessian(theta,dat,mbs,model=model)


        ll_micro += ll_i
        dll_micro += dll_i
        d2ll_micro += d2ll_i

        alpha_list[i] = a_i
        q0_list[i] = q0_i
        c_list_H[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZH)
        c_list_S[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZS)

        dalpha_list[i,:] = da_i
        dq0_list[i,:] = dq0_i

        d2q0_list[i,:,:] = d2q0_i

        sbound_mean[i] = sb_i
        abound_mean[i] = ab_i
        skipped_list[i] = dat.skip

    ll_macro, dll_macro, d2ll_macro,BFGS_next = KernelFunctions.macro_likelihood_hess(alpha_list,c_list_H,c_list_S,q0_list,
                                                                                      dalpha_list,dq0_list,d2q0_list,theta,
                                                                                      BFGS_prior=kwargs.get("BFGS_prior"))
    ll = ll_micro + ll_macro
    dll = dll_micro + dll_macro
    d2ll = d2ll_micro + d2ll_macro

    # Print and output likelihood value
    min_q0 = np.min(q0_list[skipped_list==False])
    max_q0 = 1-np.max(q0_list[skipped_list==False])
    print("Estimated purchase probability Bounds: ",min_q0,max_q0)
    # print("Likelihood:",ll, "Macro ll component:", ll_macro)
    if np.sum(sbound_mean)>0:
        print("Fraction on Share Bound",np.mean(sbound_mean))
    # print("Fraction below Alpha Bound",np.mean(abound_mean))
    return ll/N, dll/N, d2ll/N, BFGS_next

##### Optimization Functions to Maximize Likelihood and Estimate Parameters #######

### Newton-Raphson Estimation in Parallel
## Use Newton-Raphson and the analytical hessian matrix to maximize likelihood function 
# Inputs 
# x - starting guess for parameter vector
# theta - parameter object with specification info
# cdf - consumer/loan level data frame
# mdf - market level data frame
# mbdsf - MBS coupon price data frame
# num_workers - threads to use in parallel
# Outputs
# ll_k - maximized likelihood
# x - estimated parameter vector 
def estimate_NR(x,theta,cdf,mdf,mbsdf,
                parallel=False,num_workers=0,
                gtol=1e-6,xtol=1e-12,
                max_step_size = 10,
                pre_condition=False,pre_cond_itr=10):
    # Testing Tool: A index of parameters to estimate while holding others constant
    # This can help identification. range(0,len(x)) will estimate all parameters 
    test_index = theta.beta_x_ind

    # Create list of consumer data (could be memory issue, duplicating data)
    clist = consumer_object_list(theta,cdf,mdf,mbsdf)

    # Exit if number of workers hasn't been specified
    if parallel and (num_workers<2):
        print("ERROR: Number of workers has not been specified (or is fewer than 2)")
        return None, x

    if pre_condition:
        print("Run Gradient Ascent to pre-condition:")
        ll_pre,x = estimate_GA(x,theta,clist,
                               parallel=parallel,
                               num_workers=num_workers,
                               itr_max=pre_cond_itr)

    # Set candidate vector in parameter object
    theta.set_demand(x)  

    # Print initial parameters
    print("Starting Guess:",x)

    # Initialize "new" candidate parameter vector
    x_new = np.copy(x)
    # Intitial evaluation of likelihood, gradient (f_k), and hessian (B_k)
    ll_k, f_k, B_k,bfgs_mem = evaluate_likelihood_hessian(x,theta,clist,
                                                          parallel=parallel,
                                                          num_workers=num_workers)
    # Translate Hessian into a negative definite matrix for best ascent 
    # Also raises a warning when Hessian is not negitive definite, bad sign if it happens near convergence
    B_k = enforceNegDef(B_k[np.ix_(test_index,test_index)])
    print("Starting Eval:",ll_k)
    # Initialize error and iteration count
    err = 1e3
    itr = 0

    # Initialize best value
    ll_best = np.copy(ll_k)
    x_best = np.copy(x)
    f_best = np.copy(f_k)
    B_best = np.copy(B_k)
    bfgs_mem_best = bfgs_mem
    # Allow small backward movement, but only occasionally
    allowance = 1.01
    backward_tracker = 0 
    stall_count = 0

    # Iterate while error exceeds tolerance
    while (err>gtol) | (ll_k<ll_best):
        # Update best so far
        if ll_k>ll_best:
            ll_best = np.copy(ll_k)
            x_best = np.copy(x)
            f_best = np.copy(f_k)
            B_best = np.copy(B_k)
            bfgs_mem_best = bfgs_mem
            backward_tracker = 0
        else:
            backward_tracker +=1

        # If we have been behind the best for long, use a more strict search
        if (backward_tracker>1):
            stall_count +=1
            allowance = 1.00
            # Return to values at best evaluation
            ll_k = np.copy(ll_best)
            x = np.copy(x_best)
            f_k = np.copy(f_best)
            B_k = np.copy(B_best)
            bfgs_mem = bfgs_mem_best
            print("Stalled Progress. Previous best:",ll_k,"Return to best guess at:", x)
        elif stall_count>0:
            allowance = 1.00
        else:
            allowance = 1.01


        # Compute newton step
        p_k = -np.dot(np.linalg.inv(B_k),f_k[test_index])
        # Initial line search value: full newton step
        alpha = 1.0
        # Bound the step size to be one in order to avoid model crashes on odd parameters
        largest_step = np.max(np.abs(p_k))
        alpha = np.minimum(max_step_size/largest_step,1.0)

        # Compute bounded step
        s_k = alpha*p_k
        # Update potential parameter vector
        x_new[test_index] = x[test_index] + s_k
        print("Parameter Guess:",x_new)

        # Recompute likelihood, gradient and hessian
        ll_new, f_new, B_new, bfgs_mem_new = evaluate_likelihood_hessian(x_new,theta,clist,
                                                                         parallel=parallel,
                                                                         num_workers=num_workers,
                                                                         BFGS_prior=bfgs_mem)
        
        # If the initial step leads to a much lower likelihood value
        # shrink the step size to search for a better step.
        line_search = 0  # Initialize line search flag
        attempt_gradient_step = 0 
        while ll_new<(ll_k*allowance): # Allow a small step in the wrong direction
            line_search = 1 # Save that a line search happened
            alpha = alpha/10 # Shrink the step size
            print("Line Search Step Size:",alpha) # Print step size for search
            s_k = alpha*p_k # Compute new step size
            x_new[test_index] = x[test_index] + s_k # Update new candidate parameter vector
            
            # Check new value of the likelihood function
            ll_new = evaluate_likelihood(x_new,theta,clist,
                                            parallel=parallel,
                                            num_workers=num_workers)
            
            if (alpha<1e-3) & (attempt_gradient_step==0):
                stall_count +=1
                attempt_gradient_step = 1
                print("#### Begin Gradient Ascent")
                ll_new,x_new = estimate_GA(x,theta,clist,parallel=parallel,
                                           num_workers=num_workers,
                                           itr_max=4)
            elif (attempt_gradient_step==1):
                print("#### No Better Point Found")
                return ll_best, x_best
        if stall_count>3:
            print("#### No Better Point Found")
            err = np.mean(np.sqrt(f_best[test_index]**2))
            print("Completed with Error", err, "at Function Value",ll_best)
            return ll_best, x_best
        
        if (line_search == 0) & (backward_tracker==0):
            stall_count = 0
                
        # Update parameter vector after successful step
        final_step = np.abs(x-x_new)
        x = np.copy(x_new)
        theta.set_demand(x)  
        
        # If no gradient ascent was done, update likelihood, gradient and hessian
        if attempt_gradient_step==0:
            ll_k = np.copy(ll_new)
            f_k = np.copy(f_new)
            B_k = np.copy(B_new)
            bfgs_mem = bfgs_mem_new
        # If there was a line search, need to evaluate the hessian again
        else:
            ll_k, f_k, B_k, bfgs_mem = evaluate_likelihood_hessian(x,theta,clist,
                                                                    parallel=parallel,
                                                                    num_workers=num_workers,
                                                                    BFGS_prior=bfgs_mem)

        # Check that the hessian is negative definite and enforce it if necessary
        B_k = enforceNegDef(B_k[np.ix_(test_index,test_index)])
        # Evaluate the sum squared error of the gradient of the likelihood function
        err = np.mean(np.sqrt(f_k[test_index]**2))
        # Update iteration count and print error value
        itr+=1 
        print("#### Iteration",itr, "Evaluated at ",x)
        print("#### Iteration",itr, "Likelihood Value", ll_k, "Gradient Size", err)
        print("#### Iteration",itr,"Step Size (max, min):",np.max(final_step),np.min(final_step))
        if (np.max(final_step)<xtol) & (ll_k>=ll_best):
            print("#### Tolerance on Parameter Updated Magnitude Reached ####")
            break

    # Print completion and output likelihood and estimated parameter vector
    print("Completed with Error", err, "at Function Value",ll_best)
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


def estimate_GA(x,theta,clist,parallel=False,num_workers=0,gtol=1e-6,xtol=1e-15,itr_max=50):
    # Testing Tool: A index of parameters to estimate while holding others constant
    # This can help identification. range(0,len(x)) will estimate all parameters 
    test_index = theta.beta_x_ind

    # Setup the Parallel or Serial objective functions
    if parallel and (num_workers<2):
        raise Exception("GRADIENT ASCENT PARALLEL ERROR: Number of workers has not been specified (or is fewer than 2)")

    def f_grad(x):
        return evaluate_likelihood_gradient(x,theta,clist,parallel=parallel,num_workers=num_workers)
    


    # Set candidate vector in parameter object
    theta.set_demand(x)

    # Initialize "new" candidate parameter vector
    x_test = np.copy(x)


    # Initial evaluation of likelihood (ll_k) and gradient (g_k)
    ll_k, g_k = f_grad(x)
    print("Starting Parameters:",x[test_index])
    print("Starting Value:",ll_k)
    print("Starting Gradient:",g_k[test_index])
    
    #Initial step size
    alpha = 1e-3/np.max(np.abs(g_k[test_index]))

    # Initial evaluation of error and iteration count
    err = 1000
    itr = 0
    # Iterate until error becomes small or a small, maximum iteration count 
    while (err>gtol) & (itr<itr_max):
        # Asceding step is the step size times the gradient
        s_k = alpha*g_k[test_index]
        # Update candidate new parameter vector
        x_test[test_index] = x[test_index] + s_k


        # Evaluation of likelihood and gradient at new candidate vector
        ll_test, g_test = f_grad(x_test)
        
        print("Parameters:",x_test[test_index])
        print("Value:",ll_test)
        print("Gradient:",g_test[test_index])
        
        # Theoretically, a small enough step should always increase the likelihood
        # If likelihood is lower at a given step, shrink until it is an ascending step
        while ll_test<ll_k:

            alpha = alpha/10 # Shrink step size
            s_k = alpha*g_k[test_index] # New gradient ascent step
            x_test[test_index] = x[test_index] + s_k # Recompute a new candidate parameter vector

            # Re-evaluate likelihood and gradient
            ll_test,g_test = f_grad(x_test)
            print("Parameters:",x_test[test_index])
            print("Value:",ll_test)
            print("Gradient:",g_test[test_index])
            if np.max(np.abs(s_k))<xtol:
                print("GA: reached minimum step size")
                return ll_k,x

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
        err = np.mean(np.sqrt(g_k[test_index]**2))
        # Update iteration count and print error value
        itr+=1 
        print("GA Iteration:",itr, ", Likelihood:", ll_k,", Gradient Size:", err)
   
    # Print completion and output likelihood and estimated parameter vector
    # print("Completed with Error", err)
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



def predicted_elasticity(x,theta,cdf,mdf,mbsdf,model="base"):
    # Set parameters in the parameter object
    theta.set_demand(x)

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