import EstimationFunctions as ef
import multiprocessing as mp

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
    res = p.starmap(ef.consumer_likelihood_eval, xlist) # Evaluate in parallel
    p.close() # Close parallel workers
    return res

def eval_map_likelihood_gradient(xlist,num_workers):
    p = mp.Pool(num_workers) # Initialize parallel workers
    res = p.starmap(ef.consumer_likelihood_eval_gradient, xlist) # Evaluate in parallel
    p.close() # Close parallel workers
    return res

def eval_map_likelihood_hessian(xlist,num_workers):
    p = mp.Pool(num_workers) # Initialize parallel workers
    res = p.starmap(ef.consumer_likelihood_eval_hessian, xlist) # Evaluate in parallel
    p.close() # Close parallel workers
    return res
