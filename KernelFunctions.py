import scipy as sp
import numpy as np
import numdifftools as nd

import EstimationFunctions

## Use Kernel Density to estimate total purcahse share
def outside_share(a_vec,c_h_vec,c_s_vec,q0_vec,out_share):
    ### Three dimensional density function
    # cost_h x cost_s x alpha
    X = np.zeros((3,len(a_vec)))
    X[0,:] = a_vec
    X[1,:] = c_s_vec
    X[2,:] = c_h_vec

    # Potentially use for dropping observations from the kernel 
    ind = range(len(a_vec))
    
    #Estimate kernel density
    dist_cond = sp.stats.gaussian_kde(X[:,ind])
    dist_cond_obs = dist_cond(X)

    # Weight using model-implied bayes rule
    wgts = ( (1-out_share)/(1-q0_vec) - (1-out_share))*(1/out_share)

    # Estimate non-purchase kernel density
    dist_out = sp.stats.gaussian_kde(X[:,ind],weights=wgts[ind])
    dist_outside_obs = dist_out(X)

    # Implied total density 
    dist_uncond = dist_cond_obs*(1-out_share) + dist_outside_obs*out_share

    # Predicted outside option share
    pred_out=sum(dist_uncond*q0_vec)/sum(dist_uncond)
    return pred_out




def macro_likelihood(a_list,c_list_H,c_list_S,q0_list,theta,skip_vec):
    out_indices = [int(x) for x in np.unique(theta.out_vec)]
    pred_out = np.zeros(len(out_indices))
    for o in out_indices:
        ind = theta.out_vec==o
        a_mkt = a_list[ind]#[theta.out_sample[o]]
        q0_mkt = q0_list[ind]#[theta.out_sample[o]]
        c_mkt_H = c_list_H[ind]
        c_mkt_S = c_list_S[ind]

        skip_mkt = skip_vec[ind]

        # pred_out[o] = outside_share(a_mkt,c_mkt_H,c_mkt_S,q0_mkt,theta.out_share[o])
        pred_out[o] = np.mean(q0_mkt[skip_mkt==False])
    ll_macro = np.sum(theta.N*theta.out_share*np.log(pred_out)) + \
                    np.sum(theta.N*(1-theta.out_share)*np.log(1-pred_out))
    return ll_macro

def macro_likelihood_grad(a_list,c_list_H,c_list_S,q0_list,
                          da_list,dq0_list,theta,skip_vec):
    out_indices = [int(x) for x in np.unique(theta.out_vec)]
    pred_out = np.zeros(len(out_indices))
    grad = np.zeros(len(theta.all()))
    for o in out_indices:
        ind = theta.out_vec==o
        a_mkt = a_list[ind]#[theta.out_sample[o]]
        q0_mkt = q0_list[ind]#[theta.out_sample[o]]
        c_mkt_H = c_list_H[ind]
        c_mkt_S = c_list_S[ind]

        da_mkt = da_list[ind]#[theta.out_sample[o]]
        dq0_mkt = dq0_list[ind]#[theta.out_sample[o]]

        skip_mkt = skip_vec[ind]

        # out, g= out_share_gradient(a_mkt,c_mkt_H,c_mkt_S,q0_mkt,
        #                            da_mkt,dq0_mkt,theta.out_share[o],theta)
        out = np.mean(q0_mkt[skip_mkt==False])
        g = np.mean(dq0_mkt[skip_mkt==False,:],0)
        pred_out[o] = out
        x = theta.N[o]*(theta.out_share[o]*(g)/out - (1-theta.out_share[o])*(g)/(1-out) )
        grad += x
    
    if (any(pred_out>(1-1e-3))) or (any(pred_out<1e-3)):
        print("Outside Share Close to Corner Solution:",pred_out)
    ll_macro = np.sum(theta.N*theta.out_share*np.log(pred_out)) + \
                    np.sum(theta.N*(1-theta.out_share)*np.log(1-pred_out))
    return ll_macro, grad


def macro_likelihood_hess(a_list,c_list_H,c_list_S,q0_list,da_list,dq0_list,d2q0_list,theta,skip_vec,**kwargs):
    
    out_indices = [int(x) for x in np.unique(theta.out_vec)]
    pred_out = np.zeros(len(out_indices))
    grad = np.zeros(len(theta.all()))
    hess = np.zeros((len(theta.all()),len(theta.all())))
    for o in out_indices:
        ind = theta.out_vec==o
        a_mkt = a_list[ind]#[theta.out_sample[o]]
        q0_mkt = q0_list[ind]#[theta.out_sample[o]]
        c_mkt_H = c_list_H[ind]
        c_mkt_S = c_list_S[ind]

        da_mkt = da_list[ind]#[theta.out_sample[o]]
        dq0_mkt = dq0_list[ind]#[theta.out_sample[o]]

        skip_mkt = skip_vec[ind]

        d2q0_mkt = d2q0_list[ind]
        # x = theta.N[o]*theta.out_share[o]*(np.dot(grad_alpha,da_mkt) + np.dot(grad_q0,dq0_mkt))
        
        # out, grad= out_share_gradient(a_mkt,q0_mkt,da_mkt,dq0_mkt,theta.out_share[o])
        # log_pred_out[o] = log_out
        # x = theta.N[o]*theta.out_share[o]*(grad)

        # out, g= out_share_gradient(a_mkt,c_mkt_H,c_mkt_S,q0_mkt,
        #                            da_mkt,dq0_mkt,theta.out_share[o],theta)
    
        out = np.mean(q0_mkt[skip_mkt==False])
        g = np.mean(dq0_mkt[skip_mkt==False,:],0)

        pred_out[o] = out
        x = theta.N[o]*(theta.out_share[o]*(g)/out - (1-theta.out_share[o])*(g)/(1-out) )
        grad += x

        y = theta.N[o]*theta.out_share[o]*(np.mean(d2q0_mkt[skip_mkt==False,:,:],0)/out - np.outer(g,g)/out**2) - \
         theta.N[o]*(1-theta.out_share[o])*(np.mean(d2q0_mkt[skip_mkt==False,:,:],0)/(1-out) + np.outer(g,g)/(1-out)**2)
        hess += y

    ll_macro = np.sum(theta.N*theta.out_share*np.log(pred_out)) + \
                    np.sum(theta.N*(1-theta.out_share)*np.log(1-pred_out))
    
    if (any(pred_out>(1-1e-3))) or (any(pred_out<1e-3)):
        print("Outside Share Close to Corner Solution:",pred_out)
    print("Outside Share",pred_out)
    #Approximate Hessian (BFGS)
    BFGS_mem = kwargs.get("r_start")
    x_curr = theta.all()[theta.beta_x_ind]
    if BFGS_mem is None:
        H_new = -np.identity(len(grad))
        # H_new = np.zeros((len(grad),len(grad)))
    else:
        x0,g0,H0 = BFGS_mem
        dx = x_curr - x0
        dg = grad - g0
        H_new = H0 + (np.outer(dg,dg))/(np.dot(dg,dx)) - np.dot(np.dot(H0,np.outer(dx,dx)),np.transpose(H0))/np.dot(np.dot(np.transpose(dx),H0),dx)
    
    BFGS_next = (x_curr,grad,H_new)
    return ll_macro, grad, hess, BFGS_next
    # return ll_macro, grad, H_new, BFGS_next


def out_share_gradient(a_mkt,c_mkt_H,c_mkt_S,q0_mkt,
                        da_mkt,dq0_mkt,
                        out_share,theta):
    grad = np.zeros(len(theta.all()))
    x = theta.beta_x
    def f_obj(vec):
        a_1 = a_mkt + np.dot(da_mkt[:,0:len(x)],(vec-x)) 
        q_1 = q0_mkt + np.dot(dq0_mkt[:,0:len(x)],(vec-x)) 
        f0 = outside_share(a_1,c_mkt_H,c_mkt_S,q_1,out_share)
        return f0
    
    ## Step Size
    max_delt = np.max(dq0_mkt,1)
    min_delt = np.abs(np.min(dq0_mkt,1))
    min_epsilon1 = np.min(q0_mkt[max_delt>0]/max_delt[max_delt>0])
    min_epsilon2 = np.min((1-q0_mkt[min_delt>0])/min_delt[min_delt>0])
    epsilon = np.minimum(1e-10,np.minimum(min_epsilon1,min_epsilon2)/2)

    grad_func = nd.Gradient(f_obj,step=epsilon,method="central")
    g = grad_func(x)
    grad[0:len(x)] = g
    f = f_obj(x)
    return f,grad
    

# def macro_ll_test(x,theta,cdf,mdf,mbsdf):
#     # Print candidate parameter guess 
#     print("Parameters:", x)
#     # Set parameters in the parameter object
#     theta.set_demand(x)   

#     ll_micro = 0.0

    
#     alpha_list = np.zeros(cdf.shape[0])
#     q0_list = np.zeros(cdf.shape[0])
#     # Iterate over all consumers
#     itr_avg = 0
#     for i in range(cdf.shape[0]):
#         # Subset data for consumer i
#         dat, mbs = EstimationFunctions.consumer_subset(i,theta,cdf,mdf,mbsdf)
#         # Evaluate likelihood and gradient for consumer i 
#         ll_i,q0_i,a_i,itr = EstimationFunctions.consumer_likelihood_eval(theta,dat,mbs)
#         # Add outside option, probability weights, and derivatives
#         # pred_N[dat.out] += w_i
#         # pred_N_out[dat.out] += q0_i*w_i
#         # dpred_N[dat.out,:] += dw_i
#         # q0_mkt[dat.out] += q0_i
#         # dq0_mkt[dat.out,:] += dq0_i
#         # mkt_Obs[dat.out] +=1
#         # Add contribution to total likelihood, gradient and hessian (by outside option market)
#         # Derivatives need to account for the fact that the likelihood is weighted
#         # ll_market[dat.out] +=ll_i*w_i
#         # dll_market[dat.out,:] += dll_i*w_i + ll_i*dw_i

#         ll_micro += ll_i

#         alpha_list[i] = a_i
#         q0_list[i] = q0_i

#         # Track iteration counts (for potential diagnostics)
#         itr_avg += max(itr,0)
    
#     # Compute total log likelihood
#     ll_macro =macro_likelihood(alpha_list,q0_list,theta)

#     return ll_macro


# def test_macro_derivative(x,theta,cdf,mdf,mbsdf):
#     f0 = macro_ll_test(x,theta,cdf,mdf,mbsdf)
#     N= len(x)
#     epsilon = 1e-6
#     grad = np.zeros(N)
#     for i in range(N):
#         x_new = np.copy(x)
#         x_new[i] = x[i] + epsilon 
#         f1 = macro_ll_test(x_new,theta,cdf,mdf,mbsdf)
#         g = (f1-f0)/epsilon
#         print(i,g)
#         grad[i] = g
#     return grad
    

