from ModelFunctions import *
import EquilibriumFunctions
import EstimationFunctions
## Derivative of First Order Condition
# Second order conditions that will help find the nash equilibrium 
# Derivative computed with respect to alpha and all unknown prices

# r - interest rate vector
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
# m - MBS pricing function (may change this when using MBS data)

def d_foc(r,alpha,d,theta,m,model="base"):

    # Profit and Derivative
    if model=="base":
        pi,dpi_dr,d2pi_dr2 = d2SaleProfit_dr2(r,d,theta,m)
    elif model=="hold":
        pi,dpi_dr,d2pi_dr2 = d2HoldOnly_dr2(r,d,theta)

    # Market Shares and Derivatives
    q =  market_shares(r,alpha,d,theta)
    

    ## Derivative of q w.r.t. alpha
    r_avg = sum(r*q)
    dqdalpha = q*(r-r_avg)


    # Linearized to isolate alpha and q
    dEPidr = (dpi_dr)/(alpha*(1-q)) + pi

    dFOC_dalpha = ( -(dpi_dr)/(alpha*(1-q))**2 ) * ( 1-q-alpha*dqdalpha )


    q_mat = np.tile(q,(len(q),1))
    q_mat_t = np.transpose(q_mat)

    ## Derivative of FOC_j w.r.t. r_k - Columns: FOC_j, Rows: r_k (der ) 
    infra_mat = np.tile((dpi_dr),(len(q),1))
    dFOC_dr = -(infra_mat/(1-q_mat)**2)*(q_mat*q_mat_t)

    diag_vals = (dpi_dr)*q/(1-q) + dpi_dr + d2pi_dr2/(alpha*(1-q))
    np.fill_diagonal(dFOC_dr,diag_vals) 

    return dFOC_dalpha, dFOC_dr, dEPidr


def d_foc_all_parameters(r,alpha,d,theta,m,model="base"):

    # Profit and Derivative
    # Profit and Derivative
    if model=="base":
        pi,dpi_dr,d2pi_dr2 = d2SaleProfit_dr2(r,d,theta,m)
    elif model=="hold":
        pi,dpi_dr,d2pi_dr2 = d2HoldOnly_dr2(r,d,theta)


    # Market Shares and Derivatives
    q =  market_shares(r,alpha,d,theta)

    q_mat = np.tile(q,(len(q),1))
    q_mat_t = np.transpose(q_mat)
    
    ## Derivative of q w.r.t. alpha
    r_avg = sum(r*q)
    dqdalpha = q*(r-r_avg)

    ## Derivative of q w.r.t. beta_x, col: q_j, row: beta_xk
    q_mat_x = np.tile(q,(d.X.shape[1],1))
    dqdbeta_x = q_mat_x*(np.transpose(d.X) - np.transpose(np.tile(np.dot(np.transpose(d.X),q),(d.X.shape[0],1))) )

    # Linearized FOC to isolate alpha and q
    # dEPidr = (dpi_dr)/(alpha*(1-q)) + pi
    
    ## Derivative of FOC w.r.t. Alpha
    dFOC_dalpha = ( -(dpi_dr)/(alpha*(1-q))**2 ) * ( 1-q-alpha*dqdalpha )


    ## Derivative of FOC w.r.t. beta_x - Columns: FOC_j, Rows: beta_xk
    infra_mat = np.tile((dpi_dr),(dqdbeta_x.shape[0],1))
    dFOC_dbeta_x = (infra_mat/(alpha*(1-q_mat_x)**2))*(dqdbeta_x)

    ## Derivative of FOC w.r.t. gamma - Columns: FOC_j, Rows: gamma_k
    dFOC_dgamma = -np.transpose(np.concatenate((d.W,np.tile(d.Z,(5,1))),axis=1))

    ## Derivative of FOC_j w.r.t. r_k - Columns: FOC_j, Rows: r_k (der ) 
    infra_mat = np.tile((dpi_dr),(len(q),1))
    dFOC_dr = -(infra_mat/(1-q_mat)**2)*(q_mat*q_mat_t)

    diag_vals = (dpi_dr)*q/(1-q) + dpi_dr + d2pi_dr2/(alpha*(1-q))
    np.fill_diagonal(dFOC_dr,diag_vals) 

    return dFOC_dalpha, dFOC_dr, dFOC_dbeta_x, dFOC_dgamma

def d2_foc_all_parameters(r,alpha,d,theta,m,model="base"):

    # Profit and Derivative
    if model=="base":
        pi,dpi_dr,d2pi_dr2,d3pi_dr3 = d3SaleProfit_dr3(r,d,theta,m)
    elif model=="hold":
        pi,dpi_dr,d2pi_dr2,d3pi_dr3 = d3HoldOnly_dr3(r,d,theta)

    
    # # Market Shares and Derivatives
    q =  market_shares(r,alpha,d,theta)
    J = len(q)
    K = d.X.shape[1]

    q_mat = np.tile(q,(len(q),1))
    q_mat_t = np.transpose(q_mat)
    

    ## Derivative of q w.r.t. alpha
    r_avg = sum(r*q)
    r2_avg = sum((r**2)*q)
    dqdalpha = q*(r-r_avg)
    d2qdalpha2 = dqdalpha*(r-r_avg) - q*(r2_avg - r_avg**2)

    # ## Derivative of q w.r.t. beta_x, col: q_j, row: beta_xk
    q_mat_x = np.tile(q,(K,1))
    x_avg = np.dot(np.transpose(d.X),q)
    xr_avg = np.dot(np.transpose(d.X),q*r)
    x_mat = np.transpose(d.X - np.tile(x_avg,(J,1)))
    xr_mat = np.transpose(np.tile(xr_avg - x_avg*r_avg,(J,1)))
    dqdbeta_x = q_mat_x*(x_mat)
    d2qdbeta_x = np.zeros((K,K,J))
    for l in range(K):
        x_avg_kl = np.dot(d.X[:,l]*np.transpose(d.X),q)
        d2qdbeta_x[l,:,:] = dqdbeta_x*x_mat[l,:] -q_mat_x*(np.transpose(np.tile(x_avg_kl,(J,1))) -np.transpose(np.tile(x_avg,(J,1)))*x_avg[l])
            
    

    ## Derivative of FOC w.r.t. Alpha
    dFOC_dalpha = ( -(dpi_dr)/(alpha*(1-q))**2 ) * ( 1-q-alpha*dqdalpha )
    d2FOC_dalpha2 = -(dpi_dr)*( (1/(alpha*(1-q))**2 )*( -dqdalpha - (dqdalpha + alpha*d2qdalpha2) ) + 
      (-2/(alpha*(1-q))**3)*( 1-q-alpha*dqdalpha )**2)
    
    ## Derivative of FOC w.r.t. beta_x - Columns: FOC_j, Rows: beta_xk
    infra_mat = np.tile((dpi_dr),(dqdbeta_x.shape[0],1))
    dFOC_dbeta_x = (infra_mat/(alpha*(1-q_mat_x)**2))*(dqdbeta_x)

    ## Second Derivative of FOC w.r.t. beta_xk & alpha -  Columns: FOC_j, Rows: beta_xk
    r_mat = np.tile(r-r_avg,(K,1))
    d2FOC_dbetadalpha =infra_mat*( (1/(alpha*(1-q_mat_x)**2))*(r_mat*dqdbeta_x - xr_mat*q_mat_x ) - 
                             (1/(alpha**2*(1-q_mat_x)**3)*(dqdbeta_x)*(1-q_mat_x-2*alpha*r_mat*q_mat_x)) ) 


    ##Second Derivative of FOC w.r.t. beta_xk & beta_xl - dim: (J x K x K)
    d2FOC_dbeta_x2 = np.zeros((J,K,K))
    infra = dpi_dr
    for j in range(J):
        d2FOC_dbeta_x2[j,:,:] = infra[j]/(alpha*(1-q[j])**2)*d2qdbeta_x[:,:,j] + \
                                        2*infra[j]/(alpha*(1-q[j])**3)* np.outer(dqdbeta_x[:,j],dqdbeta_x[:,j])
        
    
    ## Derivative of FOC w.r.t. gamma - Columns: FOC_j, Rows: gamma_k
    dFOC_dgamma = -np.transpose(np.concatenate((d.W,np.tile(d.Z,(5,1))),axis=1))

    ## Derivative of FOC_j w.r.t. r_k - Columns: FOC_j, Rows: r_k (der ) 
    infra_mat = np.tile((dpi_dr),(len(q),1))
    dFOC_dr = -(infra_mat/(1-q_mat)**2)*(q_mat*q_mat_t)

    diag_vals = (dpi_dr)*q/(1-q) + dpi_dr + d2pi_dr2/(alpha*(1-q))
    np.fill_diagonal(dFOC_dr,diag_vals) 

    ## Derivative of FOC_j w.r.t. r_k & alpha - Columns: FOC_j, Rows: r_k (der )
    r_mat = np.tile(r-r_avg,(J,1)) 
    r_mat_t = np.transpose(r_mat)
    d2FOC_drdalpha = infra_mat*(-(1/(1-q_mat)**2)*(q_mat*q_mat_t*(r_mat + r_mat_t)) -
                                (2/(1-q_mat)**3)*(q_mat**2*q_mat_t*r_mat))

    diag_vals = ((dpi_dr)*(dqdalpha/(1-q) + q*dqdalpha/(1-q)**2) - 
                    (1/(alpha*(1-q) )**2)*(1-q-alpha*dqdalpha)*(d2pi_dr2 ) )
    np.fill_diagonal(d2FOC_drdalpha,diag_vals) 

    ## Derivative of FOC_j w.r.t. r_k & r_l - dim: (FOC x r_k x r_l)
    d2FOC_dr2 = np.zeros((J,J,J))
    infra = dpi_dr
    for j in range(J):
        d2FOC_dr2[j,:,:] = infra[j]*( (2*alpha*q[j]*q_mat*q_mat_t)/(1-q[j])**2 + 
                                     (2*alpha*q[j]**2*q_mat*q_mat_t)/(1-q[j])**3 )
        


        j_sym = infra[j]*(-alpha*q[j]*q/(1-q[j])-alpha*q[j]**2*q/(1-q[j])**2) - \
                            (q[j]*q/(1-q[j])**2)*(d2pi_dr2[j])
        

        d2FOC_dr2[j,j,:] = j_sym
        d2FOC_dr2[j,:,j] = j_sym

        r_diag_vals = -infra[j]*( (alpha*q[j]*q*(1-2*q))/(1-q[j])**2 - 
                                (2*alpha*(q[j]*q)**2)/(1-q[j])**3 )
        np.fill_diagonal(d2FOC_dr2[j,:,:],r_diag_vals) 


        d2FOC_dr2[j,j,j] = infra[j]*(alpha*q[j]+alpha*q[j]**2/(1-q[j]) ) +\
                            2*q[j]/(1-q[j])*(d2pi_dr2[j]) +\
                            d2pi_dr2[j] +\
                            d3pi_dr3[j]/(alpha*(1-q[j]))
        
    ## Derivative of FOC_j w.r.t. r_k & beta_xl - dim: (FOC x r_k x beta_xl)
    d2FOC_drdbeta = np.zeros((J,J,K))
    infra = dpi_dr
    # x_mat_t = np.transpose(x_mat)
    q_mat_x_t = np.transpose(q_mat_x)
    for j in range(J):
        # x_mat_k = np.tile(x_mat[:,j],(J,1))
        dqdb_j = np.tile(dqdbeta_x[:,j],(J,1))
        
        # d2FOC_drdbeta[j,:,:] = -infra[j]*( (1/(1-q[j])**2)*q[j]*q_mat_x_t*(x_mat_t + x_mat_k) + 
        #                                     (2/(1-q[j]**3)*q[j]**2*q_mat_x_t*x_mat_k))
        d2FOC_drdbeta[j,:,:] = -infra[j]*( (1/(1-q[j])**2)*(dqdb_j*q_mat_x_t + np.transpose(dqdbeta_x)*q[j]  ) + 
                                            (2/(1-q[j])**3)*q[j]*q_mat_x_t*dqdb_j)
        
        d2FOC_drdbeta[j,j,:] = infra[j]*( dqdbeta_x[:,j]/(1-q[j])*(1 + q[j]/(1-q[j])) ) +\
                                dqdbeta_x[:,j]/(alpha*(1-q[j])**2)*(d2pi_dr2[j])

    d2FOC_dendo2 = np.copy(d2FOC_dr2)
    d2FOC_dendo2[:,d.lender_obs,:] = np.transpose(d2FOC_drdalpha)
    d2FOC_dendo2[:,:,d.lender_obs] = np.transpose(d2FOC_drdalpha)
    d2FOC_dendo2[:,d.lender_obs,d.lender_obs] = d2FOC_dalpha2

    L = d.W.shape[1] + d.Z.shape[0]
    d2FOC_dpar2 = np.zeros((J,K+L,K+L))
    d2FOC_dpar2[:,0:K,0:K] = d2FOC_dbeta_x2
    
    d2FOC_dendodpar = np.zeros((J,J,K+L))
    d2FOC_dendodpar[:,:,0:K] = d2FOC_drdbeta
    d2FOC_dendodpar[:,d.lender_obs,0:K] = np.transpose(d2FOC_dbetadalpha)

    dFOC_dendo = np.transpose(np.copy(dFOC_dr))
    dFOC_dendo[:,d.lender_obs] = dFOC_dalpha

    dFOC_dpar = np.transpose(np.concatenate((dFOC_dbeta_x,dFOC_dgamma),axis=0))
    # return dFOC_dalpha, dFOC_dr, dFOC_dbeta_x,dFOC_dgamma, d2FOC_dalpha2, d2FOC_dr2,d2FOC_drdalpha,d2FOC_dbeta_x2,d2FOC_dbetadalpha,d2FOC_drdbeta
    return dFOC_dendo, dFOC_dpar, d2FOC_dendo2,d2FOC_dendodpar,d2FOC_dpar2


def share_parameter_second_derivatives(r,alpha,d,theta,m,model="base"):
    ## Evaluate all FOC derivatives
    df_dendo,df_dtheta, d2f_dendo2, d2f_dendodtheta, d2f_dtheta2 = d2_foc_all_parameters(r,alpha,d,theta,m,model=model)

    dendo_dtheta = -np.transpose(np.dot(np.linalg.inv(df_dendo),df_dtheta))

    L, K, K2 = d2f_dtheta2.shape
    K_beta = len(theta.beta_x)
    d2endo_dtheta2 = np.zeros((L,K,K))

    for k in range(K):
        A = np.zeros((L,K))
        A_t = np.zeros((L,K))
        B = np.zeros((L,K))

        for l in range(L):
            A[l,:] =  np.dot(dendo_dtheta[k,:],d2f_dendodtheta[l,:,:])
            A_t[l,:] = np.dot(d2f_dendodtheta[l,:,k],np.transpose(dendo_dtheta))
            B[l,:] = np.dot(np.dot(dendo_dtheta[k,:],d2f_dendo2[l,:,:]),np.transpose(dendo_dtheta))

        d2endo_dtheta2[:,:,k] = -np.dot(np.linalg.inv(df_dendo),(d2f_dtheta2[:,:,k] + A + A_t + B))

    q, dqdr, d2qdr2, dqdalpha,d2qdalpha2,d2qdrdalpha, dqdbeta_x, d2qdbeta_x,d2qdbetadr, d2qdbetadalpha = share_partial_deriv(r,alpha,d,theta,m)

    
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

    # return alpha derivative for macro gradient
    dalpha_dtheta = dendo_dtheta[:,d.lender_obs]

    return dlogq_dtheta, d2logq_dtheta2, dq0_dtheta, d2q0_dtheta2, dalpha_dtheta



def share_parameter_derivatives(r,alpha,d,theta,m,model="base"):
    ## Evaluate all FOC derivatives
    dfdalph, df_dr, df_db, df_dg = d_foc_all_parameters(r,alpha,d,theta,m,model=model)

    q, dqdr, d2qdr2, dqdalpha,d2qdalpha2,d2qdrdalpha, dqdbeta_x, d2qdbeta_x,d2qdbetadr, d2qdbetadalpha = share_partial_deriv(r,alpha,d,theta,m)

    ## Replace the observed interest rate with the alpha parameter
    df_dendo = np.transpose(np.copy(df_dr))
    df_dendo[:,d.lender_obs] = dfdalph

    ## Compute implicit derivatives of non-observed rates and alpha
    df_dtheta = np.transpose(np.concatenate((df_db,df_dg),axis=0))
    dendo_dtheta = -np.transpose(np.dot(np.linalg.inv(df_dendo),df_dtheta))

    # # Market Shares
    # q =  market_shares(r,alpha,d,theta)
    # q_mat = np.tile(q,(len(q),1))
    # q_mat_t = np.transpose(q_mat)

    # # Market Share Parameter Derivatives
    # ## Derivative of q w.r.t. beta_x, col: q_j, row: beta_xk
    # q_mat_x = np.tile(q,(d.X.shape[1],1))
    # dqdbeta_x = q_mat_x*(np.transpose(d.X) - np.transpose(np.tile(np.dot(np.transpose(d.X),q),(d.X.shape[0],1))) )


    ## Derivative of shares w.r.t. cost parameters is 0.
    dqdtheta = np.zeros((len(theta.all()),len(q)))
    dqdtheta[theta.beta_x_ind,:] = dqdbeta_x

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

    # return alpha derivative for macro gradient
    dalpha_dtheta = dendo_dtheta[:,d.lender_obs]

    return dlogq_dtheta, dq0_dtheta, dalpha_dtheta


def share_partial_deriv(r,alpha,d,theta,m):
     # Market Shares and Derivatives
    q =  market_shares(r,alpha,d,theta)
    J = len(q)
    K = d.X.shape[1]

    # Market Share Derivatives w.r.t. interest rates    # Market Share Derivatives w.r.t. interest rates
    q_mat = np.tile(q,(len(q),1))
    q_mat_t = np.transpose(q_mat)
    dqdr = -alpha*q_mat*q_mat_t
    np.fill_diagonal(dqdr,alpha*q*(1-q)) 

    d2qdr2 = np.zeros((J,J,J))
    for j in range(J):
        d2qdr2[j,:,:] = 2*alpha**2*q[j]*q_mat*q_mat_t
        diag_val = 2*alpha**2*q[j]*q**2 - alpha**2*q[j]*q
        np.fill_diagonal(d2qdr2[j,:,:],diag_val)
        j_sym = 2*alpha**2*q[j]**2*q - alpha**2*q[j]*q
        d2qdr2[j,j,:] = j_sym
        d2qdr2[j,:,j] = j_sym
        d2qdr2[j,j,j] = 2*alpha**2*q[j]**3 - 3*alpha**2*q[j]**2 + alpha**2*q[j]


    ## Derivative of q w.r.t. alpha
    r_avg = sum(r*q)
    r2_avg = sum((r**2)*q)
    dqdalpha = q*(r-r_avg)
    d2qdalpha2 = dqdalpha*(r-r_avg) - q*(r2_avg - r_avg**2)

    dqda_mat = np.tile(dqdalpha,(len(q),1))
    dqda_mat_t = np.transpose(dqda_mat)
    d2qdrdalpha = -q_mat*q_mat_t - alpha*(dqda_mat*q_mat_t + dqda_mat_t*q_mat)
    diag_val = q*(1-q) + alpha*dqdalpha*(1-2*q)
    np.fill_diagonal(d2qdrdalpha,diag_val)

    ## Derivative of q w.r.t. beta_x, col: q_j, row: beta_xk
    q_mat_x = np.tile(q,(K,1))
    x_avg = np.dot(np.transpose(d.X),q)
    xr_avg = np.dot(np.transpose(d.X),q*r)
    x_mat = np.transpose(d.X - np.tile(x_avg,(J,1)))
    dqdbeta_x = q_mat_x*(x_mat)
    # Second Derivative dims: (theta_k x theta_l x q_j)
    d2qdbeta_x = np.zeros((K,K,J))
    for l in range(K):
        x_avg_kl = np.dot(d.X[:,l]*np.transpose(d.X),q)
        d2qdbeta_x[l,:,:] = dqdbeta_x*x_mat[l,:] -q_mat_x*(np.transpose(np.tile(x_avg_kl,(J,1))) -np.transpose(np.tile(x_avg,(J,1)))*x_avg[l])
    # Second Der w.r.t. beta_x and r_l: beta_k x r_l x s_j
    d2qdbetadr = np.zeros((K,J,J))
    for j in range(J):
        dqdb = np.transpose(np.broadcast_to(dqdbeta_x[:,j],(len(q),dqdbeta_x.shape[0])))
        d2qdbetadr[:,:,j] = -alpha*(dqdb*q_mat_x + dqdbeta_x*q[j])
        d2qdbetadr[:,j,j] = alpha*dqdbeta_x[:,j]*(1- 2*q[j])

    # Second Der w.r.t. beta_x and alpha: (K x J)
    xr_mat = np.transpose(np.tile(xr_avg - x_avg*r_avg,(J,1)))
    r_mat = np.tile(r-r_avg,(K,1))
    d2qdbetadalpha = r_mat*dqdbeta_x - xr_mat*q_mat_x

    
    return q, dqdr, d2qdr2, dqdalpha,d2qdalpha2, d2qdrdalpha, dqdbeta_x, d2qdbeta_x, d2qdbetadr, d2qdbetadalpha


