import numpy as np
import EquilibriumFunctions
import EstimationFunctions
from Derivatives import *

def deriv_test_endo(d,theta,m):
    vec = theta.all()

    eps = 1e-5
    n = 0
    for i in range(len(vec)):
        theta.set(vec)
        a1, r1, itr,f = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m)
        f1 = r1
        f1[d.lender_obs] = a1
        vec_new = np.copy(vec)
        vec_new[i] = vec[i] + eps
        theta.set(vec_new)
        a2, r2, itr,f = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m)
        f2 = r2
        f2[d.lender_obs] = a2
        der = (f2-f1)/(eps)
        if n==0:
            d_init = der
        if n==1:
            D = np.vstack((d_init,der))
        if n>1:
            D = np.vstack((D,der))
        n+=1
        
        theta.set(vec)
    return D

def deriv2_test_endo(d,theta,m):
    vec = theta.all()
    eps = 1e-6
    K = len(vec)
    D = np.zeros((d.X.shape[0],K,K))
    a0, r0, itr = EquilibriumFunctions.solve_eq_r(d.r_obs,d.lender_obs,d,theta,m)
    f0  = cnst_mkup_endo(r0,d,theta,m)
    for i in range(K):
        k = i
        # k = theta.gamma_ind[i]
        vec_new = np.copy(vec)
        vec_new[k] = vec[k] + eps
        theta.set(vec_new)
        a1, r1, itr = EquilibriumFunctions.solve_eq_r(d.r_obs,d.lender_obs,d,theta,m)
        f1 = cnst_mkup_endo(r1,d,theta,m)
        der = (f1-f0)/(eps)
        for l in range(D.shape[0]):
            for j in range(K):
                D[l,j,i] = der[j,l]
    return D



def deriv_test_total_shares(d,theta,m):
    vec = theta.all()

    eps = 1e-5
    n = 0
    for i in range(len(vec)):
        theta.set(vec)
        a1, r1, itr, f = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m)
        q1 =  market_shares(r1,a1,d,theta)
        vec_new = np.copy(vec)
        vec_new[i] = vec[i] + eps
        theta.set(vec_new)
        a2, r2, itr, f = EquilibriumFunctions.solve_eq_r_optim(d.r_obs,d.lender_obs,d,theta,m)
        q2 =  market_shares(r2,a2,d,theta)

        der = (np.log(q2)-np.log(q1))/(eps)
        if n==0:
            d_init = der
        if n==1:
            D = np.vstack((d_init,der))
        if n>1:
            D = np.vstack((D,der))
        n+=1
        
        theta.set(vec)
    return D

def deriv2_test_total_shares(d,theta,m):
    vec = theta.all()
    eps = 1e-6
    D = np.zeros((len(vec),len(vec),d.X.shape[0]))
    a0, r0, itr = EquilibriumFunctions.solve_eq_r(d.r_obs,d.lender_obs,d,theta,m)
    f0, hess = share_parameter_second_derivatives(r0,a0,d,theta,m)
    for i in range(len(vec)):
        vec_new = np.copy(vec)
        vec_new[i] = vec[i] + eps
        theta.set(vec_new)
        a1, r1, itr = EquilibriumFunctions.solve_eq_r(d.r_obs,d.lender_obs,d,theta,m)
        f1, hess = share_parameter_second_derivatives(r1,a1,d,theta,m)
        der = (f1-f0)/(eps)
        D[i,:,:] = der
        # for l in range(D.shape[0]):
        #     for j in range(len(vec)):
        #         D[i,j,l] = der[j,l]
    return D

def deriv_test_likelihood(vec,theta,cdf,mdf,mbsdf):
    grad = np.zeros(len(vec))
    eps = 1e-8
    n = 0
    theta.set_demand(vec)
    ll1 = EstimationFunctions.evaluate_likelihood(vec,theta,cdf,mdf,mbsdf)
    for i in range(len(vec)):
        vec_new = np.copy(vec)
        vec_new[i] = vec[i] + eps
        ll2 = EstimationFunctions.evaluate_likelihood(vec_new,theta,cdf,mdf,mbsdf)

        vec_new[i] = vec[i] - eps
        ll3 = EstimationFunctions.evaluate_likelihood(vec_new,theta,cdf,mdf,mbsdf)

        der = (ll2-ll3)/(2*eps)
        print(der)
        grad[i] = der
        
        theta.set_demand(vec)
    return grad

def num_hessian_likelihood(vec,theta,cdf,mdf,mbsdf):

    eps = 1e-6
    n = 0
    theta.set_demand(vec)
    ll1, grad_1 = EstimationFunctions.evaluate_likelihood_gradient(vec,theta,cdf,mdf,mbsdf)
    D = np.zeros((len(vec),len(vec)))
    for i in range(len(vec)):
        vec_new = np.copy(vec)
        vec_new[i] = vec[i] + eps
        ll2, grad_2 = EstimationFunctions.evaluate_likelihood_gradient(vec_new,theta,cdf,mdf,mbsdf)

        D[:,i] = (grad_2[0:len(vec)]-grad_1[0:len(vec)])/(eps)
        
        theta.set_demand(vec)
    return D

def num_hessian_likelihood_small(theta,cdf,mdf,index):
    vec = theta.all()

    eps = 1e-6
    n = 0
    theta.set(vec)
    ll1, grad_1 = EstimationFunctions.evaluate_likelihood_gradient(vec,theta,cdf,mdf)
    for i in index:
        vec_new = np.copy(vec)
        vec_new[i] = vec[i] + eps
        ll2, grad_2 = EstimationFunctions.evaluate_likelihood_gradient(vec_new,theta,cdf,mdf)

        der = (grad_2[index]-grad_1[index])/(eps)
        if n==0:
            d_init = der
        if n==1:
            D = np.vstack((d_init,der))
        if n>1:
            D = np.vstack((D,der))
        n+=1
        
        theta.set(vec)
    return D

def deriv_test_alpha(r,alpha,d,theta,m):
    f1 = expected_foc(r,alpha,d,theta,m)
    eps = 1e-6
    f2 = expected_foc(r,alpha+eps,d,theta,m)
    f0 = expected_foc(r,alpha-eps,d,theta,m)
    der = (f2-f0)/(2*eps)
    return der

def deriv_test_alpha_share(r,alpha,d,theta,m):
    f1 = market_shares(r,alpha,d,theta)
    eps = 1e-6
    f2 = market_shares(r,alpha+eps,d,theta)
    f0 = market_shares(r,alpha-eps,d,theta)
    der = (f2-f0)/(2*eps)
    return der

def deriv2_test_alpha(r,alpha,d,theta,m):
    f1, df_dr, df_db, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,d,theta,m)
    eps = 1e-6
    f2, df_dr, df_db, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha+eps,d,theta,m)
    
    f0, df_dr, df_db, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha-eps,d,theta,m)
    
    der = (f2-f0)/(2*eps)
    return der

def deriv2_test_alpha_rate(r,alpha,d,theta,m):
    x, f1, df_db, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,d,theta,m)
    eps = 1e-6
    x, f2, df_db, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha+eps,d,theta,m)
    
    f0, f0, df_db, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha-eps,d,theta,m)
    
    der = (f2-f0)/(2*eps)
    return der

def deriv2_test_alpha_beta(r,alpha,d,theta,m):
    x, y, f1, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,d,theta,m)
    eps = 1e-6
    x, y, f2, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha+eps,d,theta,m)
    
    x,y, f0, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha-eps,d,theta,m)
    
    der = (f2-f0)/(2*eps)
    return der

def deriv2_test_rate(r,alpha,d,theta,m):
    eps = 1e-8
    D = np.zeros((len(r),len(r),len(r)))
    for i in range(len(r)):
        r_test_1 = np.copy(r)
        r_test_1[i] = r[i]-eps
        df_da, f1, df_db, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r_test_1,alpha,d,theta,m)
        r_test_2 = np.copy(r)
        r_test_2[i] = r[i]+eps
        df_da, f2, df_db, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r_test_2,alpha,d,theta,m)
        der = (f2-f1)/(2*eps)
        for f in range(len(r)):
            for j in range(len(r)):
                D[f,j,i] = der[j,f]
    return D


def deriv2_test_rate_share(r,alpha,d,theta,m):
    eps = 1e-8
    D = np.zeros((len(r),len(r),len(r)))
    for i in range(len(r)):
        r_test_1 = np.copy(r)
        r_test_1[i] = r[i]-eps
        q, f1, d2qdr2, dqdalpha,d2qdalpha2, dqdbeta_x, d2qdbeta_x = share_partial_deriv(r_test_1,alpha,d,theta,m)
        r_test_2 = np.copy(r)
        r_test_2[i] = r[i]+eps
        q, f2, d2qdr2, dqdalpha,d2qdalpha2, dqdbeta_x, d2qdbeta_x = share_partial_deriv(r_test_2,alpha,d,theta,m)
        der = (f2-f1)/(2*eps)
        for f in range(len(r)):
            for j in range(len(r)):
                D[f,j,i] = der[j,f]
    return D


def deriv_test_rate(r,alpha,d,theta,m):
    eps = 1e-6
    x,y,f0 = d_foc(r,alpha,d,theta,m,model="hold")
    D = np.zeros((len(r), len(f0)))
    for i in range(len(r)):
        r_test = np.copy(r)
        r_test[i] = r[i]+eps
        x,y,f1 = d_foc(r_test,alpha,d,theta,m)

        r_test = np.copy(r)
        r_test[i] = r[i]-eps
        x,y,f2 = d_foc(r_test,alpha,d,theta,m)

        der = (f1-f2)/(2*eps)
        D[i,:] = der
    return D

def deriv_test_rate_share(r,alpha,d,theta,m):
    eps = 1e-6
    f0 = market_shares(r,alpha,d,theta)
    D = np.zeros((len(r), len(f0)))
    for i in range(len(r)):
        r_test = np.copy(r)
        r_test[i] = r[i]+eps
        f1 = market_shares(r_test,alpha,d,theta)

        r_test = np.copy(r)
        r_test[i] = r[i]-eps
        f2 = market_shares(r_test,alpha,d,theta)

        der = (f1-f2)/(2*eps)
        D[i,:] = der
    return D

def deriv_test_beta_x(r,alpha,d,theta,m):
    eps = 1e-6
    for i in range(len(theta.beta_x)):
        f1 = expected_foc(r,alpha,d,theta,m)
        theta.beta_x[i] = theta.beta_x[i]+eps
        f2 = expected_foc(r,alpha,d,theta,m)
        theta.beta_x[i] = theta.beta_x[i]-eps
        der = (f2-f1)/(eps)
        if i==0:
            d_init = der
        if i==1:
            D = np.vstack((d_init,der))
        if i>1:
            D = np.vstack((D,der))
    return D

def deriv2_test_beta(r,alpha,d,theta,m):
    eps = 1e-8
    D = np.zeros((len(r),len(theta.beta_x),len(theta.beta_x)))
    for i in range(len(theta.beta_x)):
        theta.beta_x[i] = theta.beta_x[i]+eps
        df_da, x, f1, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,d,theta,m)
        theta.beta_x[i] = theta.beta_x[i]-eps
        df_da, x, f2, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,d,theta,m)
        der = (f1-f2)/(eps)
        for f in range(len(r)):
            for j in range(len(theta.beta_x)):
                D[f,j,i] = der[j,f]
    return D

def deriv2_test_r_beta(r,alpha,d,theta,m):
    eps = 1e-8
    D = np.zeros((len(r),len(r),len(theta.beta_x)))
    for i in range(len(theta.beta_x)):
        theta.beta_x[i] = theta.beta_x[i]+eps
        df_da, f1, x, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,d,theta,m)
        theta.beta_x[i] = theta.beta_x[i]-eps
        df_da, f2, x, df_dg,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,d,theta,m)
        der = (f1-f2)/(eps)
        for f in range(len(r)):
            for j in range(len(r)):
                D[f,j,i] = der[j,f]
    return D


def deriv_test_gamma(r,alpha,d,theta,m):
    vec = theta.all()
    start_ind = len(theta.beta_x)

    eps = 1e-6
    n = 0
    for i in range(start_ind,len(vec)):
        theta.set(vec)
        f1 = expected_foc(r,alpha,d,theta,m)
        vec_new = np.copy(vec)
        vec_new[i] = vec[i] + eps
        theta.set(vec_new)
        f2 = expected_foc(r,alpha,d,theta,m)
        der = (f2-f1)/(eps)
        if n==0:
            d_init = der
        if n==1:
            D = np.vstack((d_init,der))
        if n>1:
            D = np.vstack((D,der))
        n+=1
    return D

def deriv2_test_gamma(r,alpha,d,theta,m):
    eps = 1e-6
    D = np.zeros((len(r),len(r),len(theta.gamma_ZH)))
    for i in range(len(theta.gamma_ZH)):
        theta.gamma_ZH[i] = theta.gamma_ZH[i]+eps
        df_da, f1, x, y,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,d,theta,m)
        theta.gamma_ZH[i] = theta.gamma_ZH[i]-eps
        df_da, f2, x, y,d2alpha,d2r,drdalpha,d2beta,dbetadalpha,drdbeta = d2_foc_all_parameters(r,alpha,d,theta,m)
        der = (f1-f2)/(eps)
        for f in range(der.shape[1]):
            for j in range(der.shape[0]):
                D[f,j,i] = der[j,f]
    return D

def deriv_test_foc(r,alpha,d,theta,m):
    eps = 1e-6
    f0,dump = dSaleProfit_dr(r,d,theta,m)
    q = market_shares(r,alpha,d,theta)
    f0 = f0*q
    D = np.zeros((len(r), len(f0)))
    for i in range(len(r)):
        r_test = np.copy(r)
        r_test[i] = r[i]+eps
        f1,dump = dSaleProfit_dr(r_test,d,theta,m)
        q = market_shares(r_test,alpha,d,theta)
        f1 = f1*q
        der = (f1-f0)/(eps)
        D[i,:] = der
    return D

def deriv_test_cons_ll(vec,theta,d,m):
    grad = np.zeros(len(vec))
    eps = 1e-5
    theta.set_demand(vec)
    f0, x1, x2, x3, x4, x5, x6 = EstimationFunctions.consumer_likelihood_eval_gradient(theta,d,m)
    for i in range(len(vec)):
        vec_new = np.copy(vec)
        vec_new[i] = vec[i] + eps
        theta.set_demand(vec_new)
        f1, x0, x2, x3, x4, x5, x6 = EstimationFunctions.consumer_likelihood_eval_gradient(theta,d,m)

        der = (f1-f0)/(eps)
        grad[i] = der        
        theta.set_demand(vec)
    return grad

def hess_test_cons_ll(vec,theta,d,m):
    hess = np.zeros((len(vec),len(vec)))
    eps = 1e-5
    theta.set_demand(vec)
    x0, f0, q, dq, a, da, i = EstimationFunctions.consumer_likelihood_eval_gradient(theta,d,m)
    for i in range(len(vec)):
        vec_new = np.copy(vec)
        vec_new[i] = vec[i] + eps
        theta.set_demand(vec_new)
        x0, f1, x2, x3, x4, x5, x6 = EstimationFunctions.consumer_likelihood_eval_gradient(theta,d,m)
        
        vec_new[i] = vec[i] - eps
        theta.set_demand(vec_new)
        x0, f2, x2, x3, x4, x5, x6 = EstimationFunctions.consumer_likelihood_eval_gradient(theta,d,m)

        der = (f1-f2)/(2*eps)
        hess[:,i] = der[0:len(vec)]    
        theta.set_demand(vec)
    return hess

