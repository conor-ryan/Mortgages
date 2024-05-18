import EstimationFunctions as ef
import multiprocessing as mp
import numpy as np
import scipy as sp
import KernelFunctions

### Consumer Data List Function
## Parallel functions need a list of arguments.
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
    for i in range(cdf.shape[0]):
        dat,mbs = ef.consumer_subset(i,theta,cdf,mdf,mbsdf)
        consumer_list.append({'dat':dat,'mbs':mbs})
    return consumer_list

#### Worker Evaluation Wrappers ####
## Each worker will take a set of arguments and output the relevant values 
## These functions simply wrap the consumer likelihood evaluation functions in EstimationFunctions
## One function for likelihood, gradient, and hessian.
## Inputs
# theta - parameter object
# list_object - an item in the consumer object list
## Outputs
# Same as the associated consumer likelihood evaluation function (without iteration count)
def worker_likelihood(theta,list_object):
    ll_i,q0_i,a_i,itr  = ef.consumer_likelihood_eval(theta,list_object['dat'],list_object['mbs'])
    return ll_i,q0_i,a_i

def worker_likelihood_gradient(theta,list_object):
    ll_i,dll_i,q0_i,dq0_i,a_i,da_i,itr  = ef.consumer_likelihood_eval_gradient(theta,list_object['dat'],list_object['mbs'])
    return ll_i,dll_i,q0_i,dq0_i,a_i,da_i

def worker_likelihood_hessian(theta,list_object):
    ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i,itr   = ef.consumer_likelihood_eval_hessian(theta,list_object['dat'],list_object['mbs'])
    return ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i

#### Parallel Mapping Functions ####
## These functions implement python parallelization to the worker evaluation wrappers
# Inputs
# xlist - a list of arguments (theta,clist), where clist is an item in the consumer list object
# num_workers - number of threads to use in parallel
# Outputs
# res - a list of lists. 
# --- Each item in the outer list is a consumer
# --- Each item in the inner list is an output from the worker-level evaluation functoin
def eval_map_likelihood(xlist,num_workers):
    p = mp.Pool(num_workers) # Initialize parallel workers
    res = p.starmap(worker_likelihood, xlist) # Evaluate in parallel
    p.close() # Close parallel workers
    return res

def eval_map_likelihood_gradient(xlist,num_workers):
    p = mp.Pool(num_workers) # Initialize parallel workers
    res = p.starmap(worker_likelihood_gradient, xlist) # Evaluate in parallel
    p.close() # Close parallel workers
    return res

def eval_map_likelihood_hessian(xlist,num_workers):
    p = mp.Pool(num_workers) # Initialize parallel workers
    res = p.starmap(worker_likelihood_hessian, xlist) # Evaluate in parallel
    p.close() # Close parallel workers
    return res


###### Functions to Evaluate Full Likelihood Function in Parallel #####
## Similarly, three functions for objective, gradient, and hessian. 

### Evaluate Likelihood Function 
# Inputs
# x - Candidate parameter vector
# theta - parameter object (with specifications)
# clist - consumer data object list
# num_workers - threads to use in parallel
# Outputs
# ll - log likelihood 
def evaluate_likelihood_parallel(x,theta,clist,num_workers):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)


    
    # Initialize log likelihood tracking variables
    ll_micro = 0.0
    alpha_list = np.zeros(len(clist))
    c_list_H = np.zeros(len(clist))
    c_list_S = np.zeros(len(clist))
    q0_list = np.zeros(len(clist))


    ## Evaluate each consumer in parallel 
    args = [(theta,c_val) for c_val in clist]
    res = eval_map_likelihood(args,num_workers)

    # Iterate over all consumers to compute likelihood
    for i in range(len(res)):
        # Unpack parallel results
        dat= clist[i]['dat']
        ll_i,q0_i,a_i = res[i]
        
        ll_micro += ll_i
        alpha_list[i] = a_i
        q0_list[i] = q0_i
        c_list_H[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZH)
        c_list_S[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZS)

    # Combine Micro and Macro Likelihood Moments
    ll_macro = KernelFunctions.macro_likelihood(alpha_list,c_list_H,c_list_S,q0_list,theta)
    ll = ll_micro + ll_macro
    # Print and output likelihood value
    # print("Likelihood:",ll, "Macro Component:", ll_macro)
    return ll

### Evaluate Likelihood Function with Gradient
# Inputs
# x - Candidate parameter vector
# theta - parameter object (with specifications)
# clist - consumer data object list
# num_workers - threads to use in parallel
# Outputs
# ll - log likelihood 
# dll - gradient of log likelihood 
def evaluate_likelihood_gradient_parallel(x,theta,clist,num_workers):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)
    K = len(theta.all())

    # Initialize log likelihood tracking variables
    ll_micro = 0.0
    dll_micro = np.zeros(K)

    
    alpha_list = np.zeros(len(clist))
    q0_list = np.zeros(len(clist))
    c_list_H = np.zeros(len(clist))
    c_list_S = np.zeros(len(clist))

    dalpha_list = np.zeros((len(clist),K))
    dq0_list = np.zeros((len(clist),K))


    ## Evaluate each consumer in parallel 
    args = [(theta,c_val) for c_val in clist]
    res = eval_map_likelihood_gradient(args,num_workers)

    # Iterate over all consumers to compute likelihood
    for i in range(len(res)):
        # Unpack parallel results
        dat= clist[i]['dat']
        ll_i,dll_i,q0_i,dq0_i,a_i,da_i = res[i]

        ll_micro += ll_i
        dll_micro += dll_i

        alpha_list[i] = a_i
        q0_list[i] = q0_i
        c_list_H[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZH)
        c_list_S[i] = np.dot(np.transpose(dat.Z),theta.gamma_ZS)

        dalpha_list[i,:] = da_i
        dq0_list[i,:] = dq0_i
    
    # Compute Macro Likelihood Component and Gradient
    ll_macro, dll_macro = KernelFunctions.macro_likelihood_grad(alpha_list,c_list_H,c_list_S,q0_list,
                                                                dalpha_list,dq0_list,theta)

    ll = ll_micro + ll_macro
    dll = dll_micro + dll_macro

    # Print and output likelihood value
    # print("Likelihood:",ll, "Macro Component:", ll_macro)
    return ll, dll

### Evaluate Likelihood Function with Gradient and Hessian
# Inputs
# x - Candidate parameter vector
# theta - parameter object (with specifications)
# clist - consumer data object list
# num_workers - threads to use in parallel
# Outputs
# ll - log likelihood 
# dll - gradient of log likelihood 
# d2ll - hessian of log likelihood 
def evaluate_likelihood_hessian_parallel(x,theta,clist,num_workers,**kwargs):
    # Print candidate parameter guess 
    # print("Parameters:", x)
    # Set parameters in the parameter object
    theta.set_demand(x)
    K = len(theta.all())

    # Initialize log likelihood tracking variables
    ll_micro = 0.0
    dll_micro = np.zeros(K)
    d2ll_micro = np.zeros((K,K))

    
    alpha_list = np.zeros(len(clist))
    q0_list = np.zeros(len(clist))
    c_list_H = np.zeros(len(clist))
    c_list_S = np.zeros(len(clist))

    dalpha_list = np.zeros((len(clist),K))
    dq0_list = np.zeros((len(clist),K))
    
    d2q0_list = np.zeros((len(clist),K,K))
    
    ## Evaluate each consumer in parallel 
    args = [(theta,c_val) for c_val in clist]
    res = eval_map_likelihood_hessian(args,num_workers)

    # Iterate over all consumers to compute likelihood
    for i in range(len(res)):
        # Unpack parallel results
        dat= clist[i]['dat']
        ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i = res[i]

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

    ll_macro, dll_macro, d2ll_macro,BFGS_next = KernelFunctions.macro_likelihood_hess(alpha_list,c_list_H,c_list_S,q0_list,
                                                                                      dalpha_list,dq0_list,d2q0_list,theta,BFGS_prior=kwargs.get("BFGS_prior"))
    ll = ll_micro + ll_macro
    dll = dll_micro + dll_macro
    d2ll = d2ll_micro + d2ll_macro

    # Print and output likelihood value
    # print("Likelihood:",ll, "Macro ll component:", ll_macro)
    return ll, dll, d2ll, BFGS_next


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
def estimate_NR_parallel(x,theta,cdf,mdf,mbsdf,num_workers,gtol=1e-6,xtol=1e-12):
    # Testing Tool: A index of parameters to estimate while holding others constant
    # This can help identification. range(0,len(x)) will estimate all parameters 
    test_index = theta.beta_x_ind

    # Create list of consumer data (could be memory issue, duplicating data)
    clist = consumer_object_list(theta,cdf,mdf,mbsdf)

    # Set candidate vector in parameter object
    theta.set_demand(x) 

    # Print initial parameters
    print("Starting Guess:",x)

    # Initialize "new" candidate parameter vector
    x_new = np.copy(x)
    # Intitial evaluation of likelihood, gradient (f_k), and hessian (B_k)
    ll_k, f_k, B_k,bfgs_mem = evaluate_likelihood_hessian_parallel(x,theta,clist,num_workers)
    # Translate Hessian into a negative definite matrix for best ascent 
    # Also raises a warning when Hessian is not negitive definite, bad sign if it happens near convergence
    B_k = ef.enforceNegDef(B_k[np.ix_(test_index,test_index)])
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

    # Iterate while error exceeds tolerance
    while err>gtol:
        # Update best so far
        if ll_k>ll_best:
            ll_best = np.copy(ll_k)
            x_best = np.copy(x)
            backward_tracker = 0
        else:
            backward_tracker +=1

        # If we have been behind the best for long, use a more strict search
        if backward_tracker>1:
            allowance = 1.00
            # Return to values at best evaluation
            ll_k = np.copy(ll_best)
            x = np.copy(x_best)
            f_k = np.copy(f_best)
            B_k = np.copy(B_best)
            bfgs_mem = bfgs_mem_best
            print("Stalled Progress. Previous best:",ll_k,"Return to best guess at:", x)
        else:
            allowance = 1.01


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
        print("Parameter Guess:",x_new)

        # Recompute likelihood, gradient and hessian
        ll_new, f_new, B_new, bfgs_mem_new = evaluate_likelihood_hessian_parallel(x_new,theta,clist,num_workers,BFGS_prior=bfgs_mem)
        # If the initial step leads to a much lower likelihood value
        # shrink the step size to search for a better step.
        line_search = 0  # Initialize line search flag
        attempt_gradient_step = 0 
        while ll_new<(ll_k*allowance): # Allow a small step in the wrong direction
            line_search = 1 # Save that a line search happened
            alpha = alpha/10 # Shrink the step size
            print("Line Search Step Size:",attempt_gradient_step,alpha) # Print step size for search
            s_k = alpha*p_k # Compute new step size
            print(alpha,p_k,s_k)
            x_new[test_index] = x[test_index] + s_k # Update new candidate parameter vector

            ll_new = evaluate_likelihood_parallel(x_new,theta,clist,num_workers) # Check new value of the likelihood function
            if (alpha<1e-3) & (attempt_gradient_step==0):
                attempt_gradient_step = 1
                # alpha = 1e-3/np.max(np.abs(f_new[test_index]))
                # p_k = f_k[test_index]
                print("#### Begin Gradient Ascent")
                ll_new,x_new = estimate_GA_parallel(x,theta,clist,num_workers,itr_max=5)
            elif (attempt_gradient_step==1):
                print("#### No Better Point Found")
                return ll_best, x_best
            # elif (np.max(np.abs(s_k))<xtol) & (attempt_gradient_step==1):
            #     print("#### No Better Point Found")
            #     print(xtol)
            #     return ll_best, x_best

        # # Line Search for a larger step if this is a good direction
        # if (line_search== 0) & (ll_new>ll_best):
        #     ll_test = np.copy(ll_new)
        #     x_test = np.copy(x_new)
        #     fwd_ln_itr = 0
        #     while ll_test>=ll_new: # Allow a small step in the wrong direction
        #         ll_new = np.copy(ll_test)
        #         x_new = np.copy(x_test)
        #         print("Best so far", ll_new," at ",x_new)
        #         if fwd_ln_itr>=1: # Only if one forward search succeeded
        #             line_search = 1 # Save that a line search happened
        #         alpha = alpha*2.0 # Shrink the step size
        #         print("Still improving in this direction. Line Search Step Size:",alpha) # Print step size for search
        #         s_k = alpha*p_k # Compute new step size
        #         x_test[test_index] = x[test_index] + s_k # Update new candidate parameter vector
                
        #         # ll_prev = np.copy(ll_new)
        #         ll_test = evaluate_likelihood_parallel(x_test,theta,clist,num_workers) # Check new value of the likelihood function
        #         # x_prev = np.copy(x_test)
        #         fwd_ln_itr +=1
                
        # Update parameter vector after successful step
        final_step = np.abs(x-x_new)
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
            ll_k, f_k, B_k, bfgs_mem = evaluate_likelihood_hessian_parallel(x,theta,clist,num_workers,BFGS_prior=bfgs_mem)

        # Check that the hessian is negative definite and enforce it if necessary
        B_k = ef.enforceNegDef(B_k[np.ix_(test_index,test_index)])
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
    print("Completed with Error", err)
    return ll_k, x

def estimate_GA_parallel(x,theta,clist,num_workers,tol=1e-3,itr_max=50):
    # Testing Tool: A index of parameters to estimate while holding others constant
    # This can help identification. range(0,len(x)) will estimate all parameters 
    test_index = theta.beta_x_ind
    # Set candidate vector in parameter object
    theta.set_demand(x)

    # Initialize "new" candidate parameter vector
    x_test = np.copy(x)


    # Initial evaluation of likelihood (ll_k) and gradient (g_k)
    ll_k, g_k = evaluate_likelihood_gradient_parallel(x,theta,clist,num_workers)
    
    #Initial step size
    alpha = 1e-3/np.max(np.abs(g_k[test_index]))

    # Initial evaluation of error and iteration count
    err = 1000
    itr = 0
    # Iterate until error becomes small or a small, maximum iteration count 
    while (err>tol) & (itr<itr_max):
        # Asceding step is the step size times the gradient
        s_k = alpha*g_k[test_index]
        # Update candidate new parameter vector
        x_test[test_index] = x[test_index] + s_k


        # Evaluation of likelihood and gradient at new candidate vector
        ll_test, g_test = evaluate_likelihood_gradient_parallel(x_test,theta,clist,num_workers)
        
        # Theoretically, a small enough step should always increase the likelihood
        # If likelihood is lower at a given step, shrink until it is an ascending step
        while ll_test<ll_k:

            alpha = alpha/10 # Shrink step size
            s_k = alpha*g_k[test_index] # New gradient ascent step
            x_test[test_index] = x[test_index] + s_k # Recompute a new candidate parameter vector

            # Re-evaluate likelihood and gradient
            ll_test,g_test = evaluate_likelihood_gradient_parallel(x_test,theta,clist,num_workers)
            if np.max(np.abs(s_k))<1e-15:
                print("GA: stalled")
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



def parallel_optim(x,theta,cdf,mdf,mbsdf,num_workers):

    clist = consumer_object_list(theta,cdf,mdf,mbsdf)
    def f_obj(x):
        val, g = evaluate_likelihood_gradient_parallel(x,theta,clist,num_workers)
        g = g[0:len(x)]
        err = np.sqrt(np.mean(g**2))
        print("Parameter Guess: ", x)
        print("Likelihood Value: ", val)
        print("Gradient Size:", err)
        return -val, -g
    
    # def f_hess(x):
    #     ll, g,h = evaluate_likelihood_hessian(x,theta,cdf,mdf,mbsdf,model=model)
    #     return -h
    

    res = sp.optimize.minimize(f_obj,x,method="BFGS",jac=True),#hess=f_hess,
                            #    options = {'xtol':1e-15})

  
    return res

