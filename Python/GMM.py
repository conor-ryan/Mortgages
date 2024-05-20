import LinearModel as iv
import numpy as np
import pandas as pd

### Compute GMM Objective based on a weighting matrix W
def gmm_objective(moments,W):
    return np.matmul(np.transpose(moments),np.matmul(W,moments))

### Translate moment gradients into GMM objective function gradient
def gmm_gradient(moments,grad,W):
    gmm_grad = np.zeros(grad.shape[1])

    gmm_grad = np.matmul(np.transpose(grad),np.matmul(W,moments)) + np.matmul(np.transpose(moments),np.matmul(W,grad))

    return gmm_grad

### Translate moment gradients and hessian in to GMM objective function gradients and hessian
def gmm_hessian(moments,grad,hess,W):
    gmm_hess = np.zeros((grad.shape[1],grad.shape[1]))
    for k in range(grad.shape[1]):
        gmm_hess[k,:] = np.matmul(np.transpose(hess[:,k,:]),np.matmul(W,moments)) + np.matmul(np.transpose(grad),np.matmul(W,grad[:,k]))+np.matmul(np.transpose(moments),np.matmul(W,hess[:,k,:]))+np.matmul(np.transpose(grad[:,k]),np.matmul(W,grad))

    return gmm_hess

### Evaluate GMM Objective function based on data and parameter vector
def compute_gmm(data,par,W):
    # Cost moments
    moments_cost = iv.cost_moments(data,par)

    # Demand moments
    moments_iv = iv.demandIV(data,par)
    moments = np.concatenate((moments_iv,moments_cost),axis=0)

    # moments = iv.demandIV(data,par)
    # moments = iv.cost_moments(data,par)

    ## Temporary Monitoring
    total_val = gmm_objective(moments,W)
    # # IV Moment Component
    # idx = list(range(len(moments_iv)))
    # IV_comp = np.matmul(np.transpose(moments_iv),np.matmul(W[np.ix_(idx,idx)],moments_iv))
    # cost_comp = total_val - IV_comp
    # print('IV component is',"{:.5g}".format(IV_comp),'and cost component is',"{:.5g}".format(cost_comp))
    # # moments = iv.deposit_IV_moments(df,p)
    return total_val

def compute_gmm_gradient(data,par,W):

    moments_iv,grad_iv,hess_iv = iv.demandIV_moment_derivatives(data,par)
    moments_cost,grad_cost,hess_cost= iv.cost_moments_derivatives(data,par)

    moments = np.concatenate((moments_iv,moments_cost),axis=0)
    grad = np.concatenate((grad_iv,grad_cost),axis=0)
    # moments,grad,hess = iv.demandIV_moment_derivatives(data,par)
    # moments,grad,hess = iv.cost_moments_derivatives(data,par)

    G = gmm_gradient(moments,grad,W)
    return G

def compute_gmm_hessian(data,par,W):
    moments_iv,grad_iv,hess_iv = iv.demandIV_moment_derivatives(data,par)
    moments_cost,grad_cost,hess_cost = iv.cost_moments_derivatives(data,par)

    moments = np.concatenate((moments_iv,moments_cost),axis=0)
    grad = np.concatenate((grad_iv,grad_cost),axis=0)
    hess = np.concatenate((hess_iv,hess_cost),axis=0)
    # moments,grad,hess = iv.demandIV_moment_derivatives(data,par)
    # moments,grad,hess = iv.cost_moments_derivatives(data,par)

    f = gmm_objective(moments,W)
    G = gmm_gradient(moments,grad,W)
    H = gmm_hessian(moments,grad,hess,W)
    return f, G, H

def gmm_avar(data,par,W):
    moments_cost,grad_cost,hess_cost = iv.demandIV_moment_derivatives(data,par)
    moments_iv,grad_iv,hess_iv = iv.cost_moments_derivatives(data,par)
    G = np.concatenate((grad_iv,grad_cost),axis=0)

    # moments,G,H = iv.demandIV_moment_derivatives(data,par)

    # f = gmm_objective(moments,W)
    # G = gmm_gradient(moments,grad,W)
    # H = gmm_hessian(moments,grad,hess,W)
    S = iv.moment_longrun_variance(data,par)
    w, v = np.linalg.eig(S)
    print(w)
    print('Minimum Eigen Absolute Value',min(np.absolute(w)))
    low_vals  =np.where(w<1e-9)[0]
    for i in low_vals:
        check = np.where(np.abs(v[:,i])>0.99)
        print('Vector number ',i,'Problem ', np.where(np.abs(v[:,i])>0.1))

    # GWG = np.matmul(np.matmul(np.transpose(G),W),G)
    # GWSWG =np.matmul(np.matmul(np.matmul(np.transpose(G),W),S),np.matmul(W,G))

    # Avar = np.matmul(np.matmul(np.linalg.inv(GWG),GWSWG),np.linalg.inv(GWG))
    Avar = np.linalg.inv(np.transpose(G)@np.linalg.inv(S)@G)

    return Avar


## Numerical Derivative test functions
def numerical_gradient(data,par,W):
    tol =1e-6
    grad = np.zeros(len(par.param_vec))
    orig = compute_gmm(data,par,W)
    orig_vec = par.param_vec.copy()

    for i in range(len(par.param_vec)):
        update_vec = np.zeros(len(par.param_vec))
        update_vec[i]+=tol
        par.update(update_vec)
        new_val = compute_gmm(data,par,W)
        grad[i] = (new_val-orig)/tol
        par.set(orig_vec)
    return grad

def numerical_hessian(data,par,W):
    tol = 1e-6
    hess = np.zeros((len(par.param_vec),len(par.param_vec)))
    orig = compute_gmm_gradient(data,par,W)
    orig_vec = par.param_vec.copy()

    for i in range(len(par.param_vec)):
        update_vec = np.zeros(len(par.param_vec))
        update_vec[i]+=tol
        par.update(update_vec)
        new_val = compute_gmm_gradient(data,par,W)
        hess[i,] = (new_val-orig)/tol
        par.set(orig_vec)
    return hess
