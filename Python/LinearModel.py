import statsmodels
import numpy as np

### General Prediction Error from a Linear Model
def lm_residual(data,par,spec):
    Y = data[spec['index'],spec['dep_var']]
    X = data[np.ix_(spec['index'],spec['ind_var'])]
    beta = par.param_vec[spec['param']]
    residuals =  np.matmul(X,beta) - Y
    return residuals
### Prediction from a model with distinct endogenous & exogenous parameters
def lm_residual_endoexo(data,par,spec):
    Y = data[spec['index'],spec['dep_var']]
    X_1 = data[np.ix_(spec['index'],spec['ind_var_endo'])]
    X_2 = data[np.ix_(spec['index'],spec['ind_var_exo'])]
    beta_1 = par.param_vec[spec['param_endo']]
    beta_2 = par.param_vec[spec['param_exo']]
    residuals =  np.matmul(X_1,beta_1) + np.matmul(X_2,beta_2) - Y
    return residuals



### Interact Residual with Insturments: IV Moments, or OLS if Z = X
# Implement for each demand specification
def demandIV(data,par):
    K = len(par.dem_spec_list)
    N = par.rownum
    mom = np.empty(shape=(0,))
    # For each Specification
    for k in range(K):
        spec = par.dem_spec_list[k]
        res = lm_residual(data,par,spec)
        Z = data[np.ix_(spec['index'],spec['inst_var'])]
        # print(spec['dep_var'],sum(spec['index']))
        mom = np.append(mom,np.matmul(res,Z)/sum(spec['index']))
    # Return moments
    # Dimension is number of total instruments across all specs
    return mom

### Output residuals only for each demand specification
def demand_residuals(data,par):
    K = len(par.dem_spec_list)
    N = par.rownum
    # Residual Matrix
    # Residuals are 0 for observations where a product isn't offered
    dem_residual_matrix = np.zeros((N,K))
    # For each specification
    for k in range(K):
        spec = par.dem_spec_list[k]
        res = lm_residual(data,par,spec)
        dem_residual_matrix[spec['index'],k] = res
    return dem_residual_matrix

### Compute 1st and 2nd derivatives of IV moments (Hessian is 0)
# Concatenate across each specification
def demandIV_moment_derivatives(data,par):
        K = len(par.dem_spec_list)
        N = par.rownum
        moments = demandIV(data,par)
        grad = np.empty(shape = (0,par.parnum,))
        hess = np.zeros((len(moments),par.parnum,par.parnum))
        # For each specification
        for k in range(K):
            spec = par.dem_spec_list[k]
            grad_residual = np.zeros((sum(spec['index']),par.parnum))
            X = data[np.ix_(spec['index'],spec['ind_var'])]
            # Derivative of the residual is just X
            # This matrix is X plus 0's for parameters that aren't used in that specification
            grad_residual[:,spec['param']] = X
            Z = data[np.ix_(spec['index'],spec['inst_var'])]
            grad = np.concatenate((grad,np.matmul(np.transpose(Z),grad_residual)/sum(spec['index'])),axis=0)
        return moments, grad, hess

### Cost Equation Moments
#interaction of cost prediction residuals and demand residuals
def cost_moments(data,par):
    # Demand residuals
    res_mat = np.transpose(demand_residuals(data,par))
    # Cost residuals
    cost_res = lm_residual_endoexo(data,par,par.cost_spec)


    # Residual Interaction moments
    moments = np.matmul(res_mat,cost_res)
    #Scale by number of observations
    K = len(par.dem_spec_list)
    for k in range(K):
        spec = par.dem_spec_list[k]
        moments[k] = moments[k]/sum(spec['index'])

    # Exogenous regressor moments
    X_2 = data[np.ix_(par.cost_spec['index'],par.cost_spec['ind_var_exo'])]
    moments = np.append(moments,np.matmul(cost_res,X_2)/sum(par.cost_spec['index']))
    #Dimension is number of product demand specifications
    return moments

### Compute Cost Moment 1st and 2nd Derivatives
def cost_moments_derivatives(data,par):
    res_mat = np.transpose(demand_residuals(data,par))
    cost_res = lm_residual_endoexo(data,par,par.cost_spec)
    moments = cost_moments(data,par)

    grad = np.zeros((len(moments),par.parnum))
    hess = np.zeros((len(moments),par.parnum,par.parnum))
    K = len(par.dem_spec_list)
    cost_ind_vars = par.cost_spec['ind_var_endo']+par.cost_spec['ind_var_exo']
    cost_ind_params =np.append(par.cost_spec['param_endo'],par.cost_spec['param_exo'])


    # For each demand specification
    for k in range(K):
        spec = par.dem_spec_list[k]
        X_beta = data[np.ix_(spec['index'],spec['ind_var'])]
        X_gamma = data[np.ix_(spec['index'],cost_ind_vars)]
        grad[k,spec['param']] = np.matmul(np.transpose(cost_res[spec['index']]),X_beta)/sum(spec['index'])
        grad[k,cost_ind_params] = np.matmul(np.transpose(data[:,cost_ind_vars]),np.transpose(res_mat[k,:]))/sum(spec['index'])

        # Not positive if the indexing on the Hessian is correct, but seems to match numerical hessian
        hess[np.ix_([k],cost_ind_params,spec['param'])] = np.matmul(np.transpose(X_gamma),X_beta)/sum(spec['index'])
        hess[np.ix_([k],spec['param'],cost_ind_params)] =np.matmul(np.transpose(X_beta),X_gamma)/sum(spec['index'])

    X_gamma = data[np.ix_(par.cost_spec['index'],cost_ind_vars)]
    X_2 = data[np.ix_(par.cost_spec['index'],par.cost_spec['ind_var_exo'])]
    grad[K:,cost_ind_params] = np.matmul(np.transpose(X_2),X_gamma)/sum(par.cost_spec['index'])

    return moments, grad, hess


def moment_longrun_variance(data,par):
    K = len(par.dem_spec_list)
    N = par.rownum

    momiv = demandIV(data,par)
    res_mat = np.transpose(demand_residuals(data,par))
    # S = np.zeros((len(momiv),len(momiv)))
    momcost = cost_moments(data,par)
    cost_res = lm_residual_endoexo(data,par,par.cost_spec)

    S = np.zeros((len(momiv)+len(momcost),len(momiv)+len(momcost)))

    cost_ind_vars = par.cost_spec['ind_var_endo']+par.cost_spec['ind_var_exo']
    cost_ind_params =np.append(par.cost_spec['param_endo'],par.cost_spec['param_exo'])

    for n in range(N):
        moment_obs = np.empty(shape = (0,1,))
        # IV Moments
        # For each specification
        for k in range(K):
            spec = par.dem_spec_list[k]
            if spec['index'][n]==True:
                Z = data[n,spec['inst_var']]
            else:
                Z = np.zeros(len(spec['inst_var']))

            moment_obs = np.append(moment_obs,Z*res_mat[k,n])


        #Cost Moments
        #For each demand specification
        momcost_obs = np.zeros((len(momcost)))
        for k in range(K):
            spec = par.dem_spec_list[k]
            momcost_obs[k] = res_mat[k,n]*cost_res[n]

        # Exogenous moments
        X_2 = data[n,par.cost_spec['ind_var_exo']]
        momcost_obs[K:] =cost_res[n]*X_2

        moment_obs = np.append(moment_obs,momcost_obs)

        moment_obs = np.expand_dims(moment_obs,axis=1)

        S += np.matmul(moment_obs,np.transpose(moment_obs))

    return S/N
