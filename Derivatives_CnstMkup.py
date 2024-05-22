from ModelFunctions import *
from Derivatives import *


def cnst_mkup_second_derivatives(r,alpha,d,theta,m):

    q, dqdr, d2qdr2, dqdalpha,d2qdalpha2,d2qdrdalpha, dqdbeta_x, d2qdbeta_x,d2qdbetadr, d2qdbetadalpha = share_partial_deriv(r,alpha,d,theta,m)

    prof = expected_sale_profit(d.r_obs,d,theta,m)[d.lender_obs]
    dc_dgamma = np.transpose(np.concatenate((d.W,np.tile(d.Z,(5,1))),axis=1))
    d2c_dgamma2 = np.zeros((dc_dgamma.shape[0],dc_dgamma.shape[0],dc_dgamma.shape[1]))
    dPi_dr = dSaleProfit_dr(r,d,theta,m)
    d2Pi_dr2 = d2SaleProfit_dr2(r,d,theta,m)
    

    dalpha_dgamma= -dc_dgamma[:,d.lender_obs]/(prof)**2
    d2alpha_dgamma2 = -2*np.outer(dc_dgamma[:,d.lender_obs],dc_dgamma[:,d.lender_obs])/prof**3

    dr_dgamma = (dc_dgamma - np.transpose(np.tile(dc_dgamma[:,d.lender_obs],reps=(dc_dgamma.shape[1],1))))/np.tile(dPi_dr,reps=(dc_dgamma.shape[0],1))
    dr_dgamma[:,d.lender_obs] = dalpha_dgamma

    d2r_dgamma2 = d2c_dgamma2
    for j in range(len(q)):
        if j == d.lender_obs:
            d2r_dgamma2[:,:,j] = d2alpha_dgamma2
        else:
            d2r_dgamma2[:,:,j] = -np.outer(dr_dgamma[:,j],dr_dgamma[:,j])*d2Pi_dr2[j]/dPi_dr[j]

    dendo_dtheta = np.zeros((len(theta.all()),len(q)))
    dendo_dtheta[theta.gamma_ind,:] = dr_dgamma

    K,L = dendo_dtheta.shape
    K_beta = len(theta.beta_x)
    d2endo_dtheta2 = np.zeros((L,K,K))

    for l in range(L):
        ind_min = min(theta.gamma_ind)
        ind_max = max(theta.gamma_ind)+1
        d2endo_dtheta2[l,ind_min:ind_max,ind_min:ind_max] = d2r_dgamma2[:,:,l]


    
    
    ## Derivative of shares w.r.t. cost parameters is 0.
    dqdtheta = np.zeros((K,L))
    dqdtheta[theta.beta_x_ind,:] = dqdbeta_x

    d2qdtheta2 = np.zeros((K,K,L))
    d2qdtheta2[0:K_beta,0:K_beta,:] = d2qdbeta_x

    # Share Derivatives w.r.t. Endogenous variables (col: s_j, row: endo_k)
    dqdendo = np.copy(dqdr)
    dqdendo[d.lender_obs,:] = dqdalpha

    # Second derivative, dims: endo_k x endo_l x s_j 
    d2qdendo2 = np.copy(d2qdr2)
    d2qdendo2[:,d.lender_obs,:] = d2qdrdalpha
    d2qdendo2[d.lender_obs,:,:] = d2qdrdalpha
    d2qdendo2[d.lender_obs,d.lender_obs,:] = d2qdalpha2

    # second derivative endo and parameters
    d2qdthetadendo = np.zeros((K,L,L))
    d2qdthetadendo[0:K_beta,:,:] = d2qdbetadr
    d2qdthetadendo[0:K_beta,d.lender_obs,:] = d2qdbetadalpha

    # Share Derivatives w.r.t. parameters (col: s_j, row: theta_k)
    dqdtheta_total = dqdtheta + np.dot(dendo_dtheta,dqdendo)

    # Share Second Derivatives (theta_k x theta_l x s_j)
    d2qdtheta2_total = np.zeros((K,K,L))

    for l in range(L):
        C = np.zeros((K,K))
        for k in range(K):
            C[k,:] = np.dot(dqdendo[:,l],d2endo_dtheta2[:,:,k])

        d2qdtheta2_total[:,:,l] = d2qdtheta2[:,:,l] + \
            np.dot(d2qdthetadendo[:,:,l],np.transpose(dendo_dtheta)) +\
            np.dot(dendo_dtheta,np.transpose(d2qdthetadendo[:,:,l])) +\
            np.dot(np.dot(dendo_dtheta,d2qdendo2[:,:,l]),np.transpose(dendo_dtheta)) +\
            C

    dlogq_dtheta = dqdtheta_total/np.tile(q,(dqdtheta_total.shape[0],1))
    dq0_dtheta = -np.sum(dqdtheta_total,axis=1)
    # dlogOut_dtheta = -(1/(1-sum(q)))*dSumdtheta

    d2logq_dtheta2 = np.zeros((K,K,L))
    for j in range(L):
        d2logq_dtheta2[:,:,j] = d2qdtheta2_total[:,:,j]/q[j] -\
              (1/q[j]**2)*np.outer(dqdtheta_total[:,j],dqdtheta_total[:,j]) 

    d2q0_dtheta2 = -np.sum(d2qdtheta2_total,axis=2)
    # d2logOut_dtheta2 = -(1/(1-sum(q)))*d2Sumdtheta2 - (1/(1-sum(q))**2)*np.outer(dSumdtheta,dSumdtheta)


    return dlogq_dtheta, d2logq_dtheta2, dq0_dtheta, d2q0_dtheta2


def cnst_mkup_endo(r,d,theta,m):
    prof = expected_sale_profit(d.r_obs,d,theta,m)[d.lender_obs]
    dc_dgamma = np.transpose(np.concatenate((d.W,np.tile(d.Z,(5,1))),axis=1))
    dPi_dr = np.tile(dSaleProfit_dr(r,d,theta,m),reps=(dc_dgamma.shape[0],1))
    
    dalpha_dgamma= -dc_dgamma[:,d.lender_obs]/(prof)**2
    dr_dgamma = (dc_dgamma - np.transpose(np.tile(dc_dgamma[:,d.lender_obs],reps=(dc_dgamma.shape[1],1))))/dPi_dr
    dr_dgamma[:,d.lender_obs] = dalpha_dgamma

    return dr_dgamma



def cnst_mkup_derivatives(r,alpha,d,theta,m):
    prof = expected_sale_profit(d.r_obs,d,theta,m)[d.lender_obs]
    dc_dgamma = np.transpose(np.concatenate((d.W,np.tile(d.Z,(5,1))),axis=1))
    dPi_dr = np.tile(dSaleProfit_dr(r,d,theta,m),reps=(dc_dgamma.shape[0],1))
    
    dalpha_dgamma= -dc_dgamma[:,d.lender_obs]/(prof)**2
    dr_dgamma = (dc_dgamma - np.transpose(np.tile(dc_dgamma[:,d.lender_obs],reps=(dc_dgamma.shape[1],1))))/dPi_dr
    dr_dgamma[:,d.lender_obs] = dalpha_dgamma

    q, dqdr, d2qdr2, dqdalpha,d2qdalpha2,d2qdrdalpha, dqdbeta_x, d2qdbeta_x,d2qdbetadr, d2qdbetadalpha = share_partial_deriv(r,alpha,d,theta,m)


    ## Derivative of shares w.r.t. cost parameters is 0.
    dqdtheta = np.zeros((len(theta.all()),len(q)))
    dqdtheta[theta.beta_x_ind,:] = dqdbeta_x


    ## Combine ``endogenous'' variables
    dendo_dtheta = np.zeros((len(theta.all()),len(q)))
    dendo_dtheta[theta.gamma_ind,:] = dr_dgamma
    # # Market Share Derivatives w.r.t. interest rates
    # dqdr = -alpha*q_mat*q_mat_t
    # np.fill_diagonal(dqdr,alpha*q*(1-q)) 

    # Replace observed rate with derivative of share w.r.t. alpha (col: s_j, row: r,alpha)
    # r_avg = sum(r*q)
    # dqdalpha = q*(r-r_avg)
    dqdendo = np.copy(dqdr)
    dqdendo[d.lender_obs,:] = dqdalpha

    # Share Derivatives w.r.t. parameters (col: s_j, row: theta_k)
    dqdtheta_total = dqdtheta + np.dot(dendo_dtheta,dqdendo)
    dq0_dtheta = -np.sum(dqdtheta_total,axis=1)
    dlogq_dtheta = dqdtheta_total/np.tile(q,(dqdtheta_total.shape[0],1))
    # dlogOut_dtheta = -(1/(1-sum(q)))*dSumdtheta

    return dlogq_dtheta, dq0_dtheta

