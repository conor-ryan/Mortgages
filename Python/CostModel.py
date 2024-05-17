import ProfitFunctions as pf
import DemandIV as iv
import numpy as np
import pandas as pd

def predicted_expenses(data,par):
    obs_num = data.shape[0]
    pred_expenses = np.zeros(obs_num)
    for i in range(obs_num):
        mc_assets = (data[i,par.data_asset_p_index] - data[i,par.data_dep_p_index])*(1-1/par.param_vec[par.par_dep_index])
        pred_expenses[i] = mc_assets*data[i,par.data_asset_q_index]
        for j in range(len(par.par_prod_index)):
            mc_prod = (data[i,par.data_prod_p_index[j]])*(1-1/par.param_vec[par.par_prod_index[j]])
            pred_expenses[i] += mc_prod*data[i,par.data_prod_q_index[j]]
    return pred_expenses

# def predicted_expenses(data,par):
#     prediction = np.matmul(par.X_dep,par.beta_dep)
#     return pred_expenses

def pred_exp_moments(data,par):
    moments = predicted_expenses(data,par) - data[:,par.expenses_target]
    return moments


def gradient_pred_exp(data,par):
    obs_num = data.shape[0]
    pred_expenses = np.zeros(obs_num)
    d_exp = np.zeros((obs_num,len(par.par_prod_index)+1))
    for i in range(obs_num):
        # Deposits and Assets
        spread = data[i,par.data_asset_p_index] - data[i,par.data_dep_p_index]
        assets = data[i,par.data_asset_q_index]
        #Gradient
        dmc = (spread)/(par.param_vec[par.par_dep_index])**2
        d_exp[i,par.par_dep_index] = dmc*data[i,par.data_asset_q_index]

        # Cost Prediction
        mc_assets = (spread)*(1-1/par.param_vec[par.par_dep_index])
        pred_expenses[i] = mc_assets*assets

        # Non Interest Products
        for j in range(len(par.par_prod_index)):
            price = data[i,par.data_prod_p_index[j]]
            quantity = data[i,par.data_prod_q_index[j]]

            #Gradient
            dmc = (price)/(par.param_vec[par.par_prod_index[j]])**2
            d_exp[i,par.par_prod_index[j]] = dmc*quantity

            # Cost Prediction
            mc_prod = (price)*(1-1/par.param_vec[par.par_prod_index[j]])
            pred_expenses[i] += mc_prod*quantity

    cost_moments = pred_expenses - data[:,par.expenses_target]

    return cost_moments,d_exp



def hessian_pred_exp(data,par):
    obs_num = data.shape[0]
    pred_expenses = np.zeros(obs_num)
    d_exp = np.zeros((obs_num,len(par.par_prod_index)+1))
    d2_exp = np.zeros((obs_num,len(par.par_prod_index)+1,len(par.par_prod_index)+1))
    for i in range(obs_num):
        # Deposits and Assets
        spread = data[i,par.data_asset_p_index] - data[i,par.data_dep_p_index]
        assets = data[i,par.data_asset_q_index]

        #Hessian
        d2mc = -2*(spread)/(par.param_vec[par.par_dep_index])**3
        d2_exp[i,par.par_dep_index,par.par_dep_index] = d2mc*assets

        #Gradient
        dmc = (spread)/(par.param_vec[par.par_dep_index])**2
        d_exp[i,par.par_dep_index] = dmc*assets

        # Cost Prediction
        mc_assets = (spread)*(1-1/par.param_vec[par.par_dep_index])
        pred_expenses[i] = mc_assets*assets

        # Non Interest Products
        for j in range(len(par.par_prod_index)):
            price = data[i,par.data_prod_p_index[j]]
            quantity = data[i,par.data_prod_q_index[j]]

            #Hessian
            d2mc = -2*(price)/(par.param_vec[par.par_prod_index[j]])**3
            d2_exp[i,par.par_prod_index[j],par.par_prod_index[j]] = d2mc*quantity

            #Gradient
            dmc = (price)/(par.param_vec[par.par_prod_index[j]])**2
            d_exp[i,par.par_prod_index[j]] = dmc*quantity

            # Cost Prediction
            mc_prod = (price)*(1-1/par.param_vec[par.par_prod_index[j]])
            pred_expenses[i] += mc_prod*quantity

    cost_moments = pred_expenses - data[:,par.expenses_target]

    return cost_moments,d_exp, d2_exp
