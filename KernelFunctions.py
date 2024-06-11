import scipy as sp
import numpy as np
import numdifftools as nd
import ParallelFunctions
import EstimationFunctions

## Use Kernel Density to estimate total purcahse share
def outside_share(a_vec,c_h_vec,c_s_vec,q0_vec,out_share):
    ### Three dimensional density function
    # cost_h x cost_s x alpha
    X = np.zeros((3,len(a_vec)))
    X[0,:] = a_vec
    X[1,:] = c_s_vec
    X[2,:] = c_h_vec
    
    # X = np.zeros((1,len(a_vec)))
    # X[0,:] = a_vec

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

    # dist_outside_obs2 = dist_cond_obs*wgts

    # Implied total density 
    dist_uncond = dist_cond_obs*(1-out_share) + dist_outside_obs*out_share

    # sum(dist_uncond*q0_vec)/sum(dist_uncond)

    # sum(dist_cond_obs*q0_vec)/sum(dist_cond_obs)*(1-out_share) + sum(dist_outside_obs*q0_vec)/sum(dist_outside_obs)*(out_share)
    
    # sum(dist_cond_obs*q0_vec)/sum(dist_cond_obs)*(1-out_share) + sum(dist_outside_obs2*q0_vec)/sum(dist_outside_obs2)*(out_share)


    # Predicted outside option share
    pred_out=sum(dist_uncond*q0_vec)/sum(dist_uncond)
    return pred_out




def macro_likelihood(a_list,c_list_H,c_list_S,q0_list,theta):
    out_indices = [int(x) for x in np.unique(theta.out_vec)]
    pred_out = np.zeros(len(out_indices))
    for o in out_indices:
        ind = theta.out_vec==o
        a_mkt = a_list[ind]#[theta.out_sample[o]]
        q0_mkt = q0_list[ind]#[theta.out_sample[o]]
        c_mkt_H = c_list_H[ind]
        c_mkt_S = c_list_S[ind]

        pred_out[o] = outside_share(a_mkt,c_mkt_H,c_mkt_S,q0_mkt,theta.out_share[o])
        # pred_out[o] = np.mean(q0_mkt)
    print("Outside Share",pred_out)
    ll_macro = np.sum(theta.N*theta.out_share*np.log(pred_out)) + \
                    np.sum(theta.N*(1-theta.out_share)*np.log(1-pred_out))
    return ll_macro

def macro_likelihood_grad(a_list,c_list_H,c_list_S,q0_list,
                          da_list,dq0_list,theta):
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

        out, g= out_share_gradient(a_mkt,c_mkt_H,c_mkt_S,q0_mkt,
                                   da_mkt,dq0_mkt,theta.out_share[o],theta)
        # out = np.mean(q0_mkt)
        # g = np.mean(dq0_mkt,0)

        pred_out[o] = out
        x = theta.N[o]*(theta.out_share[o]*(g)/out - (1-theta.out_share[o])*(g)/(1-out) )
        grad += x
    
    if (any(pred_out>(1-1e-3))) or (any(pred_out<1e-3)):
        print("Outside Share Close to Corner Solution:",pred_out)
    ll_macro = np.sum(theta.N*theta.out_share*np.log(pred_out)) + \
                    np.sum(theta.N*(1-theta.out_share)*np.log(1-pred_out))
    return ll_macro, grad


def macro_likelihood_hess(a_list,c_list_H,c_list_S,q0_list,da_list,dq0_list,
                          d2a_list,d2q0_list,theta,**kwargs):
    
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

        d2a_mkt = d2a_list[ind]
        d2q0_mkt = d2q0_list[ind]

        out, g, h= out_share_hessian(a_mkt,c_mkt_H,c_mkt_S,q0_mkt,
                                   da_mkt,dq0_mkt,
                                   d2a_mkt,d2q0_mkt,theta.out_share[o],theta)
    
        # out = np.mean(q0_mkt)
        # g = np.mean(dq0_mkt,0)

        pred_out[o] = out
        x = theta.N[o]*(theta.out_share[o]*(g)/out - (1-theta.out_share[o])*(g)/(1-out) )
        grad += x

        # h = np.mean(d2q0_mkt,0)
        y = theta.N[o]*theta.out_share[o]*(h/out - np.outer(g,g)/out**2) - \
         theta.N[o]*(1-theta.out_share[o])*(h/(1-out) + np.outer(g,g)/(1-out)**2)
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
    epsilon = np.minimum(1e-8,np.minimum(min_epsilon1,min_epsilon2)/2)

    grad_func = nd.Gradient(f_obj,step=epsilon,method="central")
    g = grad_func(x)
    grad[0:len(x)] = g
    f = f_obj(x)
    return f,grad

def out_share_hessian(a_mkt,c_mkt_H,c_mkt_S,q0_mkt,
                        da_mkt,dq0_mkt,d2a_mkt,d2q0_mkt,
                        out_share,theta):
    grad = np.zeros(len(theta.all()))
    hess = np.zeros((len(theta.all()),len(theta.all())))
    x = theta.beta_x
    def f_obj(vec):
        a_1 = a_mkt + np.dot(da_mkt[:,0:len(x)],(vec-x)) #+ \
                #0.5*np.dot(np.tensordot(d2a_mkt[:,0:len(x),0:len(x)],vec-x,axes=1),np.transpose((vec-x)))
        q_1 = q0_mkt + np.dot(dq0_mkt[:,0:len(x)],(vec-x)) #+ \
                #0.5*np.dot(np.tensordot(d2q0_mkt[:,0:len(x),0:len(x)],vec-x,axes=1),np.transpose((vec-x)))
        f0 = outside_share(a_1,c_mkt_H,c_mkt_S,q_1,out_share)
        return f0
    
    ## Step Size
    max_delt = np.max(dq0_mkt,1)
    min_delt = np.abs(np.min(dq0_mkt,1))
    min_epsilon1 = np.min(q0_mkt[max_delt>0]/max_delt[max_delt>0])
    min_epsilon2 = np.min((1-q0_mkt[min_delt>0])/min_delt[min_delt>0])
    epsilon = np.minimum(1e-8,np.minimum(min_epsilon1,min_epsilon2)/2)

    f = f_obj(x)

    grad_func = nd.Gradient(f_obj,step=epsilon,method="central")
    g = grad_func(x)
    grad[0:len(x)] = g
    
    hess_func = nd.Hessian(f_obj,step=epsilon,method="central")
    h = hess_func(x)
    hess[0:len(x),0:len(x)] = h
    return f,grad, hess
    

def macro_ll_test(x,theta,clist,parallel=False,num_workers=0,model="base"):
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
    d2a_list = np.zeros((N,K,K))
    
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
            ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i,d2a_i,sb_i,ab_i  = EstimationFunctions.consumer_likelihood_eval_hessian(theta,dat,mbs,model=model)


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
        d2a_list[i,:,:] = d2a_i

        sbound_mean[i] = sb_i
        abound_mean[i] = ab_i
        skipped_list[i] = dat.skip

    ll_macro, dll_macro, d2ll_macro,BFGS_next = macro_likelihood_hess(alpha_list,c_list_H,c_list_S,q0_list,
                                                                                      dalpha_list,dq0_list,
                                                                                      d2a_list,d2q0_list,theta)


    return ll_macro, dll_macro, d2ll_macro


def macro_ll_test_2(x,theta,clist,parallel=False,num_workers=0,model="base"):
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
    d2a_list = np.zeros((N,K,K))
    
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
            ll_i,dll_i,d2ll_i,q0_i,dq0_i,d2q0_i,a_i,da_i,d2a_i,sb_i,ab_i  = EstimationFunctions.consumer_likelihood_eval_hessian(theta,dat,mbs,model=model)


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
        d2a_list[i,:,:] = d2a_i

        sbound_mean[i] = sb_i
        abound_mean[i] = ab_i
        skipped_list[i] = dat.skip

    ll_macro, dll_macro = macro_likelihood_grad(alpha_list,c_list_H,c_list_S,q0_list,
                                                                                      dalpha_list,dq0_list,theta)


    return ll_macro, dll_macro



def test_macro_derivative(x,theta,clist):
    f0, dump = macro_ll_test(x,theta,clist)
    N= len(x)
    epsilon = 1e-6
    grad = np.zeros(N)
    for i in range(N):
        x_new = np.copy(x)
        x_new[i] = x[i] + epsilon 
        f1, dump = macro_ll_test(x_new,theta,clist)
        g = (f1-f0)/epsilon
        print(i,g)
        grad[i] = g
    return grad
    

