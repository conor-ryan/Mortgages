import EstimationFunctions as ef
import multiprocessing as mp
import numpy as np
import scipy as sp
import KernelFunctions


#### Worker Evaluation Wrappers ####
## Each worker will take a set of arguments and output the relevant values 
## These functions simply wrap the consumer likelihood evaluation functions in EstimationFunctions
## One function for likelihood, gradient, and hessian.
## Inputs
# theta - parameter object
# list_object - an item in the consumer object list
## Outputs
# Same as the associated consumer likelihood evaluation function (without iteration count)
def worker_likelihood(theta,dat,mbs):
    ll_i,q0_i,a_i,itr  = ef.consumer_likelihood_eval(theta,dat,mbs)
    return ll_i,q0_i,a_i

def worker_likelihood_gradient(theta,dat,mbs):
    ll_i,dll_i,q0_i,dq0_i,a_i,da_i,sb_i   = ef.consumer_likelihood_eval_gradient(theta,dat,mbs)
    return ll_i,dll_i,q0_i,dq0_i,a_i,da_i,sb_i 

def worker_likelihood_hessian(theta,dat,mbs):
    ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i,d2a_i,sb_i,ab_i   = ef.consumer_likelihood_eval_hessian(theta,dat,mbs)
    return ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i,sb_i,ab_i

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
