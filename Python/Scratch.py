import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

## Custom Code
from Parameters import *
# from CostModel import *
from GMM import *
from Estimate import *
from LinearModel import *

plt.style.use('seaborn')

#Import dataframe
#os.chdir('/home/pando004/Desktop/BankData/FRY9')
#df = pd.read_csv('frdata.csv')
os.chdir('G:/Shared drives/BankBusinessLines')
dem_spec_list = [{'dep_var': 'log_q_deposit',
                  'ind_var': [ 'log_p_deposit',
                  'bankFactor1025608' , 'bankFactor1026632' , 'bankFactor1036967' , 'bankFactor1037003' ,
                  'bankFactor1039502' , 'bankFactor1068025' , 'bankFactor1068191' , 'bankFactor1069778' ,
                   'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1074156' , 'bankFactor1078529' ,
                   'bankFactor1111435' , 'bankFactor1119794' , 'bankFactor1120754' , 'bankFactor1132449' ,
                   'bankFactor1199611' , 'bankFactor1199844' , 'bankFactor1245415' , 'bankFactor1275216' ,
                   'bankFactor1378434' , 'bankFactor1562859' , 'bankFactor1574834' , 'bankFactor1575569' ,
                   'bankFactor1951350' , 'bankFactor2162966' , 'bankFactor2277860' , 'bankFactor2380443' ,
                    'bankFactor2816906' , 'bankFactor3232316' , 'bankFactor3242838' , 'bankFactor3587146' ,
                     'bankFactor3606542' , 'bankFactor3846375' , 'bankFactor3981856' , 'bankFactor4504654' ,
                     'bankFactor4846998' , 'bankFactor5006575' , 'bankFactor5280254' ,
                     'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
                      'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
                      'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
                      'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
                      'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
                    'inst_var': [ 'FEDFUNDS',
                    'bankFactor1025608' , 'bankFactor1026632' , 'bankFactor1036967' , 'bankFactor1037003' ,
                    'bankFactor1039502' , 'bankFactor1068025' , 'bankFactor1068191' , 'bankFactor1069778' ,
                     'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1074156' , 'bankFactor1078529' ,
                     'bankFactor1111435' , 'bankFactor1119794' , 'bankFactor1120754' , 'bankFactor1132449' ,
                     'bankFactor1199611' , 'bankFactor1199844' , 'bankFactor1245415' , 'bankFactor1275216' ,
                     'bankFactor1378434' , 'bankFactor1562859' , 'bankFactor1574834' , 'bankFactor1575569' ,
                     'bankFactor1951350' , 'bankFactor2162966' , 'bankFactor2277860' , 'bankFactor2380443' ,
                      'bankFactor2816906' , 'bankFactor3232316' , 'bankFactor3242838' , 'bankFactor3587146' ,
                       'bankFactor3606542' , 'bankFactor3846375' , 'bankFactor3981856' , 'bankFactor4504654' ,
                       'bankFactor4846998' , 'bankFactor5006575' , 'bankFactor5280254' ,
                       'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
                        'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
                        'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
                        'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
                        'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
                      'flag_var': 'flag_deposit'},
{'dep_var': 'log_q_propundwrt',
                  'ind_var': [ 'log_p_propundwrt',
                  'bankFactor1036967' , 'bankFactor1037003' , 'bankFactor1039502' , 'bankFactor1068191' ,
                   'bankFactor1069778' , 'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1074156' ,
                    'bankFactor1120754' , 'bankFactor1245415' , 'bankFactor1275216' , 'bankFactor1562859' ,
                     'bankFactor1951350' ,
                     'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
                      'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
                      'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
                      'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
                      'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
                    'inst_var': [ 'log_p_propundwrt',
                    'bankFactor1036967' , 'bankFactor1037003' , 'bankFactor1039502' , 'bankFactor1068191' ,
                     'bankFactor1069778' , 'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1074156' ,
                      'bankFactor1120754' , 'bankFactor1245415' , 'bankFactor1275216' , 'bankFactor1562859' ,
                       'bankFactor1951350' ,
                       'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
                        'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
                        'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
                        'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
                        'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
                      'flag_var': 'flag_propundwrt'},
{'dep_var': 'log_q_lifeundwrt',
                  'ind_var': [ 'log_p_lifeundwrt',
                   'bankFactor1037003' , 'bankFactor1039502' , 'bankFactor1068025' , 'bankFactor1068191' ,
                    'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1119794' , 'bankFactor1120754' ,
                     'bankFactor1951350' , 'bankFactor2380443' , 'bankFactor2816906' , 'bankFactor3242838' ,
                      'bankFactor3587146' , 'bankFactor4846998',
                      'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
                       'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
                       'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
                       'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
                       'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
                    'inst_var': [ 'log_p_lifeundwrt',
                     'bankFactor1037003' , 'bankFactor1039502' , 'bankFactor1068025' , 'bankFactor1068191' ,
                      'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1119794' , 'bankFactor1120754' ,
                       'bankFactor1951350' , 'bankFactor2380443' , 'bankFactor2816906' , 'bankFactor3242838' ,
                        'bankFactor3587146' , 'bankFactor4846998',
                        'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
                         'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
                         'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
                         'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
                         'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
                      'flag_var': 'flag_lifeundwrt'},
{'dep_var': 'log_q_annuity',
                  'ind_var': [ 'log_p_annuity',
                  'bankFactor1025608' , 'bankFactor1026632' , 'bankFactor1036967' , 'bankFactor1037003' ,
                   'bankFactor1039502' , 'bankFactor1068025' , 'bankFactor1068191' , 'bankFactor1069778' ,
                    'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1074156' , 'bankFactor1078529' ,
                     'bankFactor1111435' , 'bankFactor1119794' , 'bankFactor1120754' , 'bankFactor1132449' ,
                      'bankFactor1199611' , 'bankFactor1199844' , 'bankFactor1245415' , 'bankFactor1275216' ,
                       'bankFactor1378434' , 'bankFactor1574834' , 'bankFactor1575569' , 'bankFactor1951350' ,
                        'bankFactor2162966' , 'bankFactor2277860' , 'bankFactor2380443' , 'bankFactor2816906' ,
                         'bankFactor3232316' , 'bankFactor3242838' , 'bankFactor3587146' , 'bankFactor3606542' ,
                          'bankFactor3981856' , 'bankFactor4504654' , 'bankFactor4846998' , 'bankFactor5280254' ,
                          'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
                           'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
                           'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
                           'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
                           'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
                    'inst_var': [ 'log_p_annuity',
                    'bankFactor1025608' , 'bankFactor1026632' , 'bankFactor1036967' , 'bankFactor1037003' ,
                     'bankFactor1039502' , 'bankFactor1068025' , 'bankFactor1068191' , 'bankFactor1069778' ,
                      'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1074156' , 'bankFactor1078529' ,
                       'bankFactor1111435' , 'bankFactor1119794' , 'bankFactor1120754' , 'bankFactor1132449' ,
                        'bankFactor1199611' , 'bankFactor1199844' , 'bankFactor1245415' , 'bankFactor1275216' ,
                         'bankFactor1378434' , 'bankFactor1574834' , 'bankFactor1575569' , 'bankFactor1951350' ,
                          'bankFactor2162966' , 'bankFactor2277860' , 'bankFactor2380443' , 'bankFactor2816906' ,
                           'bankFactor3232316' , 'bankFactor3242838' , 'bankFactor3587146' , 'bankFactor3606542' ,
                            'bankFactor3981856' , 'bankFactor4504654' , 'bankFactor4846998' , 'bankFactor5280254' ,
                            'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
                             'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
                             'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
                             'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
                             'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
                      'flag_var': 'flag_annuity'}]#,
# {'dep_var': 'log_q_inv',
#                   'ind_var': [ 'log_p_inv',
#                    'bankFactor1025608' , 'bankFactor1026632' , 'bankFactor1036967' , 'bankFactor1037003' ,
#                     'bankFactor1039502' , 'bankFactor1068025' , 'bankFactor1068191' , 'bankFactor1069778' ,
#                      'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1074156' , 'bankFactor1078529' ,
#                       'bankFactor1111435' , 'bankFactor1119794' , 'bankFactor1120754' , 'bankFactor1132449' ,
#                        'bankFactor1199611' , 'bankFactor1199844' , 'bankFactor1245415' , 'bankFactor1275216' ,
#                         'bankFactor1378434' , 'bankFactor1562859' , 'bankFactor1574834' , 'bankFactor1575569' ,
#                          'bankFactor1951350' , 'bankFactor2162966' , 'bankFactor2277860' , 'bankFactor2380443' ,
#                           'bankFactor2816906' , 'bankFactor3232316' , 'bankFactor3242838' , 'bankFactor3587146' ,
#                            'bankFactor3606542' , 'bankFactor3846375' , 'bankFactor3981856' , 'bankFactor4504654' ,
#                             'bankFactor4846998' , 'bankFactor5006575' , 'bankFactor5280254' ,
#                             'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
#                              'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
#                              'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
#                              'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
#                              'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
#                     'inst_var': [ 'log_p_inv',
#                      'bankFactor1025608' , 'bankFactor1026632' , 'bankFactor1036967' , 'bankFactor1037003' ,
#                       'bankFactor1039502' , 'bankFactor1068025' , 'bankFactor1068191' , 'bankFactor1069778' ,
#                        'bankFactor1070345' , 'bankFactor1073757' , 'bankFactor1074156' , 'bankFactor1078529' ,
#                         'bankFactor1111435' , 'bankFactor1119794' , 'bankFactor1120754' , 'bankFactor1132449' ,
#                          'bankFactor1199611' , 'bankFactor1199844' , 'bankFactor1245415' , 'bankFactor1275216' ,
#                           'bankFactor1378434' , 'bankFactor1562859' , 'bankFactor1574834' , 'bankFactor1575569' ,
#                            'bankFactor1951350' , 'bankFactor2162966' , 'bankFactor2277860' , 'bankFactor2380443' ,
#                             'bankFactor2816906' , 'bankFactor3232316' , 'bankFactor3242838' , 'bankFactor3587146' ,
#                              'bankFactor3606542' , 'bankFactor3846375' , 'bankFactor3981856' , 'bankFactor4504654' ,
#                               'bankFactor4846998' , 'bankFactor5006575' , 'bankFactor5280254' ,
#                               'dateFactor2016-06-30' , 'dateFactor2016-09-30' , 'dateFactor2016-12-31' , 'dateFactor2017-03-31' ,
#                                'dateFactor2017-06-30' , 'dateFactor2017-09-30' , 'dateFactor2017-12-31' , 'dateFactor2018-03-31' ,
#                                'dateFactor2018-06-30' , 'dateFactor2018-09-30' , 'dateFactor2018-12-31' , 'dateFactor2019-03-31' ,
#                                'dateFactor2019-06-30' , 'dateFactor2019-09-30' , 'dateFactor2019-12-31' , 'dateFactor2020-03-31' ,
#                                'dateFactor2020-06-30' , 'dateFactor2020-09-30'],# , 'dateFactor2020-12-31'],
#                       'flag_var': 'flag_inv'}   ]

cost_spec = {'dep_var':'cost_dep_var',
            'ind_var_endo':['rev_deposit_tilde' , 'rev_propundwrt' , 'rev_lifeundwrt' , 'rev_annuity'],# , 'rev_inv'],
            'ind_var_exo': ['constant']}








### Market Data
df = pd.read_csv('Data/GMMSample.csv')
data = df.to_numpy()
p = Parameter(df,dem_spec_list,cost_spec)
p.check_full_rank(data)

# print('IV Moments')
# print(demandIV(data,p))
# m, g, h = demandIV_moment_derivatives(data,p)
# print(m.shape)
# print(g.shape)
# print(h.shape)
# print('Cost Moments')
# print(cost_moments(data,p))
# m, g, h = cost_moments_derivatives(data,p)
# print(m.shape)
# print(g.shape)
# print(h.shape)



# # Single Equation 2SLS W
X = data[np.ix_(p.dem_spec_list[0]['index'],p.dem_spec_list[0]['ind_var'])]


Z = data[np.ix_(p.dem_spec_list[0]['index'],p.dem_spec_list[0]['inst_var'])]
W = np.linalg.inv(np.matmul(np.transpose(Z),Z))
Y = data[p.dem_spec_list[0]['index'],p.dem_spec_list[0]['dep_var']]

Szx = np.matmul(np.transpose(Z),X)/p.rownum
Szz = np.matmul(np.transpose(Z),Z)/p.rownum
Szy = np.matmul(np.transpose(Z),Y)/p.rownum
#
est = np.matmul(np.linalg.inv(np.matmul(np.matmul(np.transpose(Szx),W),Szx)),np.matmul(np.matmul(np.transpose(Szx),W),Szy))
print(est)
print(len(est))

print('Standard Errors')

Avar = np.linalg.inv(np.transpose(Szx)@np.linalg.inv(Szz)@Szx)
print(np.diag(Avar))

res = Y - X @ est

print('Residual Variance')
print(res.shape)
print(np.var(res)/p.rownum)
# Only possible because it is specified as exactly identified
W = np.identity(p.parnum)


f,G,H = compute_gmm_hessian(data,p,W)
print('Evaluated at ',f)
grad_size = np.sqrt(np.dot(G,G))
print('Gradient Size ', grad_size)

deviation = np.random.rand(len(p.param_vec))*0.1 - 0.05
p.update(deviation)
f,G,H = compute_gmm_hessian(data,p,W)
print('Evaluated at ',f)
grad_size = np.sqrt(np.dot(G,G))
print('Gradient Size ', grad_size)




# G = compute_gmm_gradient(data,p,W)
# print(G.shape)
# f,G,H = compute_gmm_hessian(data,p,W)
# print('Evaluated at ',f)
# print(G.shape)
# print(H.shape)
# test = np.linalg.inv(H)
# print(test.shape)


# newton_raphson(data,p,W,1e-6)
# p.output(data,W)

#
# ## Exogenous Deposit Demand Covariates
# X = pd.read_csv('Data/ExogenousDemandCovariates.csv').to_numpy()
# M = annihilator_matrix(X)
# ## Deposit Demand Instruments
# Z = pd.read_csv('Data/DemandInstruments.csv').to_numpy()
#
# # ### Weighting Matrix
# cost_moment_length = df.to_numpy().shape[0]
# IV_moment_length = Z.shape[1]
# # print(cost_moment_length)
# # print(IV_moment_length)
#
#
#
#
# W = np.identity(cost_moment_length+IV_moment_length)
# diagonal = np.concatenate((np.ones(IV_moment_length),df['total_cost'].to_numpy()),axis=0)
# diagonal = np.sqrt(diagonal)*1e-18
# np.fill_diagonal(W,diagonal)
# # W = np.identity(IV_moment_length)
# Szz = np.matmul(np.transpose(Z),Z)
# W[0:IV_moment_length,0:IV_moment_length] = np.linalg.inv(Szz)
# print(W)
#
# parameter_vector = np.array([3.166537e+01,-100.,-100.,-100.,-100.,0.9,0.9,0.9,0.9])
# parameter_vector = np.concatenate((parameter_vector,np.zeros(X.shape[1])),axis=0)
#
# p = Parameter(parameter_vector,X_dep= X,Z_dep = Z)
#
# # df = simulate(df,parameter_vector,X_dep = X,Z_dep = Z)
#
# #
# deviations = np.concatenate(([10.,100.,100.,100.,100.,0.0,0.0,0.0,0.0],np.zeros(X.shape[1])),axis=0)
#
# # p0 = parameter_vector + np.random.rand(len(parameter_vector))*deviations - deviations
# p0 = parameter_vector.copy()
# p_est = newton_raphson(df,p0,W,X_dep = X,Z_dep=Z)
#
# df = predict(df,p_est,X,Z)
# df.to_csv('EstimationPrediction.csv')
#
#
print("Gradient Test")
fval, grad, hess =  compute_gmm_hessian(data,p,W)
print(grad)

grad_test = numerical_gradient(data,p,W)
print(grad_test)
print(np.mean((grad-grad_test)/grad))
print(np.max((grad-grad_test)))
print(np.min((grad-grad_test)))
print(np.mean((grad-grad_test)))

print("Hessian Test")
print(hess)

hess_test = numerical_hessian(data,p,W)
print(hess_test)
print(hess-hess_test)
# print((hess-hess_test)/hess)
print(np.mean((hess-hess_test)/(hess+0.0001)))
print(np.max((hess-hess_test)))
print(np.min((hess-hess_test)))

# #
# # moments_cost = calc_cost_moments(df,p)
# # print("Cost Moment Starting Value")
# # W_cost = np.identity(cost_moment_length)
# # diagonal = df['total_cost'].to_numpy()
# # diagonal = np.sqrt(diagonal)*1e-18
# # np.fill_diagonal(W_cost,diagonal)
# #
# # compute_gmm(df,parameter_vector,W,X_dep=X,Z_dep=Z)
# # compute_gmm_gradient(df,parameter_vector,W,X_dep=X,Z_dep=Z)
# # val, grad, hess = compute_gmm_hessian(df,parameter_vector,W,X_dep=X,Z_dep=Z)
# # print(grad)
#
#
# #
# # val = np.matmul(np.transpose(moments_cost),np.matmul(W_cost,moments_cost))
# # print(val)
# #
# # moments_iv = IV_moments(df,p)
# # print("IV Moment Starting Value")
# # val = np.matmul(np.transpose(moments_iv),np.matmul(np.linalg.inv(Szz),moments_iv))
# # print(val)
#
# # print(IV_mom)
# # p = Parameter(parameter_vector,X,Z)
# # IV_mom = deposit_IV_moments(df,p)
# # grad = dep_IV_mom_derivatives(df,p)
# # print("Residual Stats")
# # print(np.mean(IV_mom))
# # print(np.median(IV_mom))
# # print(np.max(IV_mom))
# # print(np.min(IV_mom))
#
#
#
#
# ## Add Predicted Cost to the
#
# # #
# # p_idx = [0]
# # p_idx.extend(list(range(9,(9+X.shape[1]))))
# # print("Index",p_idx)
# # p_est = newton_raphson(df,X,Z,p0,W,p_idx=p_idx)
#
#
# # np.set_printoptions(precision=2,suppress=True)
# # ### Initial Evaluation
# # fval, G, H = gmm.compute_gmm_hessian(df,p0)
# # # G = G[0:5]
# # # H = H[0:5,0:5]
# # grad_size = np.sqrt(np.dot(G,G))
# # param_vec = p0.copy()
# # itr=0
# # # Initialize Gradient Increment
# # alpha = 0.1
#
#
#
#
# # p_est = newton_raphson(df,M,Z,parameter_vector)
