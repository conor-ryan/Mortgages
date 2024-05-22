import os
os.chdir('/classified/jordan_shared/cHMDA/model')    

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.discrete.discrete_model import Logit
from FirstStageFunctions import *
from ModelTypes import *
from EquilibriumFunctions import *
from ModelFunctions import *
from EstimationFunctions import *
from ParallelFunctions import *
from Derivatives import * 

"""
    Import data
"""
# File with Loan-Level Data
consumer_data_file = "df_loan.csv"
data = pd.read_csv(consumer_data_file)
data = data.replace([np.inf,-np.inf], np.nan)

# drop shadow banks
data = data[ data.shadow == 0 ]
data = data[ (data['tier1_rwcr'].notna()) & (data['tier1_rwcr'] != 0) ]


# adjust units 
data['interest_rate'] = data['interest_rate']/100 
data['ffr']  = data['ffr']/100 

data['mbs_price'] = data['mbs_price']/100  

"""
    Some variable construction/filtering
"""
# create time FEs
df_dummies = pd.get_dummies( data['month'], columns=['month'], prefix = 'dummy_month_level', drop_first=False )*1
data = pd.concat([data,df_dummies],axis=1)

# create intercept 
data['intercept'] = 1

# sell dummy
# create time FE interacted with mortgage rate
df_dummies = pd.get_dummies( data[['interest_rate','month']], columns=['month'], prefix = 'dummy_month_int', drop_first=False )
for col in df_dummies.columns:
    if 'dummy_month_int' in col:
        df_dummies[col] = df_dummies[col]*(df_dummies['interest_rate'] - .0025 )

df_dummies.drop('interest_rate',axis=1,inplace=True)
data = pd.concat([data,df_dummies],axis=1)


# create federal funds rate interaction with mortgage rate
data['ffr_rate_interaction']     = (data['interest_rate']-.0025)/data['ffr']
data['ffr_rate_interaction_doh'] = (data['interest_rate']-.0025)*data['ffr_doh']

# Bank/Aggregate Characteristics - Cost
data['tier2_rwcr'] = data['total_rwcr'] - data['tier1_rwcr']
data['tier1_rwcr'] = 100*data['tier1_rwcr']
data['tier2_rwcr'] = 100*data['tier2_rwcr']

# truncating 
data = data[ data.applicant_income > 0 ]
data = data[ data.debt_to_income_ratio > 0 ]
data = data[ data.liq_ratio > 0 ]

data.loc[ data.liq_ratio > data.liq_ratio.quantile(.99), 'liq_ratio'] = data.liq_ratio.quantile(.99)

"""
    Loan level data
"""
# restrict to one market
data_loan = data[ (data.joint_code.isin( [ 29095, 20091] ) ) & (data.month.isin([ '2018-06','2019-06' ]) ) ]

# if interest rates are below fed funds rate, truncate
for at,t in enumerate(data_loan.month.unique()):
    
    # prevailing fed funds rate
    ffr_rate  = data_loan[ data_loan.month == t].ffr.mean() 
    ffr_quant = data_loan[ data_loan.month == t].interest_rate.quantile(.05)
    
    data_loan = data_loan[ ~( (data_loan.month == t) & (data_loan.interest_rate < ffr_quant) )]
        
"""
    Market level data
"""

#joint_df = data.groupby([ 'joint_code','month' ]).agg( names = ('lei','unique')).reset_index() 
#joint_df = joint_df.explode('names')
#joint_df['market_id'] = joint_df.joint_code + '-' + joint_df.mdate
#joint_df = joint_df.rename( columns={'names':'lei'})

market_df = data_loan[ ['market_id','month','lei','joint_code','dep_spread','tier1_rwcr','tier2_rwcr','liq_ratio','branch_num'] ].groupby([ 'lei','market_id','month']).mean().reset_index() 

loan_count = data_loan.groupby([ 'lei','market_id' ]).size()
loan_count = loan_count.rename('loan_counts')

market_df = pd.merge( market_df, loan_count.to_frame(), on=['lei','market_id'] )

market_df['bank_id'] = market_df['lei']

market_df =  market_df[market_df.loan_counts > 5 ] 
#market_df.loc[ market_df.loan_counts <= 2, 'bank_id'] = 'small'

df_dummies = pd.get_dummies( market_df['bank_id'], columns=['bank_id'], prefix = 'dum', drop_first=False )*1
market_df = pd.concat([market_df,df_dummies],axis=1)

# create lender id (within market) and market id
market_df['row_idx'] = 0
market_df['market_idx'] = 0

for idx,name in enumerate(market_df.market_id.unique()):
    J = market_df[ market_df.market_id == name ].lei.nunique()
    
    # lender IDs
    market_df.loc[ (market_df.market_id == name),'row_idx'] = np.arange(0,J)

    # market IDs
    market_df.loc[ market_df.market_id == name, 'market_idx' ] = idx

# create time index
market_df['time_id'] = 0

for at,t in enumerate( market_df.month.unique() ):
    market_df.loc[ (market_df.month == t),'time_id'] = at

# merge with loan level
data_loan = pd.merge( data_loan, market_df[['lei','market_id','row_idx','market_idx','time_id']], on =['lei','market_id'], how='right' )

# replace nans
market_df = market_df.replace(np.nan,0)



"""
    MBS Data
"""

os.chdir('/classified/jordan_shared/cHMDA/mbs_data')    

df_fred = pd.read_csv('Freddie_MBS_Price.csv') 
df_fan  = pd.read_csv('Fannie_MBS_Price.csv') 

df_fred['date'] = pd.to_datetime( df_fred['Date'] )
df_fan['date']  = pd.to_datetime( df_fan['Date'] )

from scipy.interpolate import interp1d 

# linear interpolation/extrapolation 
mbs_price_spec = ['1', '1.25', '1.5', '1.75', '2', '2.25', '2.5', '2.75', '3', '3.25', '3.5', '3.75',
                   '4', '4.25', '4.5', '4.75', '5', '5.25', '5.5', '5.75', '6', '6.25', '6.5', '6.75',
                    '7', '7.25', '7.5', '7.75', '8', '8.25', '8.5', '8.75', '9']
mbs_coupon_spec = np.asarray(( [float(x) for x in mbs_price_spec] ))


for i in range(len(df_fred)):
    
    # FRED MBS Prices 
    y_arr = np.array( df_fred.iloc[i][ list(df_fred)[1:-1]  ] )
    valid_idx = np.where( 0*y_arr == 0 )[0]
    
    f = interp1d( mbs_coupon_spec[valid_idx], y_arr[valid_idx] , fill_value='extrapolate' )
    
    df_fred.iloc[i, df_fred.columns.isin(list(df_fred)[1:-1])] =  f( mbs_coupon_spec )/100 - 1

    # FRED MBS Prices 
    y_arr = np.array( df_fan.iloc[i][ list(df_fan)[1:-1]  ] )
    valid_idx = np.where( 0*y_arr == 0 )[0]
    
    f = interp1d( mbs_coupon_spec[valid_idx], y_arr[valid_idx] , fill_value='extrapolate' )
    
    df_fan.iloc[i, df_fan.columns.isin(list(df_fan)[1:-1])] =  f( mbs_coupon_spec )/100 - 1


# combines them
df_mbs_com = df_fan.copy()
#df_mbs_com = df_fred.copy()


# for i in range(len(df_fan)):
    
#     df_mbs_com.loc[i,list(df_fred)[1:-1] ] = np.maximum( df_fan[ list(df_fred)[1:-1] ].iloc[i], 
#                                                             df_fred[ list(df_fred)[1:-1] ].iloc[i]   )/100 - 1
    
df_mbs_com['month']  = df_mbs_com['date'].dt.strftime('%Y-%m')
df_mbs_com_sub = df_mbs_com.groupby('month')[ list(df_fan)[1:-1] ].mean()
df_mbs_com_sub.reset_index(inplace=True)

# smooth prices
for i in range(len(df_mbs_com_sub)):
    
    #coeffs      = np.polyfit( mbs_coupon_spec, np.array( list(df_mbs_com_sub.loc[i,mbs_price_spec]) ), 2 )
    coeffs      = np.polyfit( mbs_coupon_spec, np.array( list(df_mbs_com_sub.loc[i,mbs_price_spec]) ), 1 )
    interp_func = np.poly1d(  coeffs ) 
    
    df_mbs_com_sub.loc[i, mbs_price_spec ] = interp_func( mbs_coupon_spec )

df_mbs_com_sub_market = df_mbs_com_sub[df_mbs_com_sub['month'].isin( ['2018-06','2019-06'] ) ]

df_mbs_com_sub_market = pd.merge( df_mbs_com_sub_market, market_df[['month','time_id']].drop_duplicates(), on = 'month', how='left' )
df_mbs_com_sub_market = df_mbs_com_sub_market.reset_index()

# df_mbs_fake = df_mbs_com_sub_market.copy()
# df_mbs_fake[ list(df_fan)[1:-1] ] = mbs_coupon_spec/(0.496*100)#-1

# outside option dataset
outside_df = pd.DataFrame({ 'time_id': list(data_loan.time_id.unique()), 'outside_share': np.ones(data_loan.time_id.nunique() )*0.05,'market_size':np.array(( data_loan.groupby('time_id').size()/.95 )) })

"""
    Run First Stage 
"""
sell_spec = "hold_dummy"  
consumer_spec = ["intercept","applicant_credit_score","applicant_income",
                 "combined_loan_to_value_ratio","debt_to_income_ratio","refi_dummy","loan_amount","ffr"]
bank_spec = ["dep_spread","tier1_rwcr","tier2_rwcr","liq_ratio","branch_num"] 
mbs_price = "mbs_price" 
interest_rate_spec = [col for col in data.columns if 'dummy_month_int' in col ]        

#other_soec = [col for col in data.columns if 'dummy_month_level' in col ]     
#other_spec = [ 'ffr' ]
#other_spec = [ 'tract_medfaminc', 'tract_pctminority', 'tract_med_age_housing_units', 'active_banks', 'active_nonbanks' ]

# VERSION 1: baseline specification 
res1, pars1 = run_first_stage(data,sell_spec,interest_rate_spec,mbs_price,consumer_spec,bank_spec) #,other_spec)

# bank dummy variables/FEs
demand_spec =  [col for col in market_df.columns if 'dum_' in col ]    

#MBS spec
mbs_price_spec = ['1', '1.25', '1.5', '1.75', '2', '2.25', '2.5', '2.75', '3', '3.25', '3.5', '3.75',
                   '4', '4.25', '4.5', '4.75', '5', '5.25', '5.5', '5.75', '6', '6.25', '6.5', '6.75',
                    '7', '7.25', '7.5', '7.75', '8', '8.25', '8.5', '8.75', '9']
mbs_coupon_spec = np.asarray(( [float(x) for x in mbs_price_spec] ))/100

# Observed Choices
rate_spec = 'interest_rate' # Paid interest rate
lender_spec = 'row_idx' # Lender choice
market_spec = 'market_id'
time_spec= 'time_id'
outside_spec = 'time_id'
discount_spec = [col for col in data.columns if 'dummy_month_level' in col ] 

"""
    Create parameter specification
"""
theta = Parameters(demand_spec,bank_spec,consumer_spec,discount_spec,
                   mbs_price_spec,mbs_coupon_spec,
                   rate_spec,lender_spec,market_spec,time_spec,outside_spec,
                   pars1,
                   outside_df['outside_share'],outside_df['market_size'])

# df_mbs_com_sub.to_csv("/classified/jordan_shared/cHMDA/model/mbs_smoothed_data.csv", index=False)
# data_test = data[ consumer_spec + bank_spec + ['interest_rate']]
# stats_df = pd.DataFrame({
#     'Min':data_test.min(),
#     'Max':data_test.max(),
#     'Mean':data_test.mean(),
#     'Vol':data_test.std()
#     })
# stats_df = stats_df.transpose()
# stats_df.to_csv("/classified/jordan_shared/cHMDA/model/summary_data.csv", index=False)


""" 
    Starting Parameter Guess
"""
# K = 10000
# parameter_guess = np.zeros((K,len(demand_spec) + len(consumer_spec) + len(bank_spec)))
# for i in range(K):
#     parameter_guess[i,len(demand_spec)] = -.05 + np.random.uniform()*0.1
#     # for j in range(len(demand_spec),len(demand_spec) + len(consumer_spec) + len(bank_spec)):
#     #     parameter_guess[i,j] = -0.5 + np.random.uniform()


init_guess = np.zeros(len(demand_spec) + len(consumer_spec) + len(bank_spec))
init_guess[len(demand_spec):(len(demand_spec) + len(bank_spec) )] = -theta.gamma_WS
init_guess[(len(demand_spec) + len(bank_spec)):(len(demand_spec) + len(consumer_spec) + len(bank_spec))] = -theta.gamma_ZS#*(theta.gamma_ZS<0)
# init_guess[len(demand_spec) + len(bank_spec)] = .289
theta.set(init_guess)

parm_func(init_guess,theta,data_loan,market_df,df_mbs_com_sub_market)
fval = evaluate_likelihood(init_guess,theta,data_loan,market_df,df_mbs_com_sub_market)

for i in range(len(data_loan)):
    dat, mbs = consumer_subset(i,theta,data_loan,market_df,df_mbs_com_sub_market)
    mc_s = np.dot(np.transpose(dat.Z),theta.gamma_ZH + theta.gamma_ZS)
    mc_h = np.dot(np.transpose(dat.Z),theta.gamma_ZH)
    print(i,mc_s,mc_h)

for i in range(len(data_loan)):
    dat, mbs = consumer_subset(i,theta,data_loan,market_df,df_mbs_com_sub_market)
    mc_s = np.dot(dat.W,(theta.gamma_WH + theta.gamma_WS) ) 
    mc_h = np.dot(dat.W,theta.gamma_WH)
    print(i,mc_s,mc_h)


def parm_func(p_guess,theta,data_loan,market_df,df_mbs_com_sub_market):

    theta.set(p_guess) 
    
    # initialize function value 
    invalid_freq = 0 
            
    # for each consumer
    for j in range(len(data_loan)):

        # define consumer subset
        dat, mbs = consumer_subset(j,theta,data_loan,market_df,df_mbs_com_sub_market)
        
        # get break-even interest rate            
        r_min = min_rate(dat, theta, mbs)
        
        # if negative or not profitable, break 
        if (r_min < 0).any() or (r_min[ dat.lender_obs ] > dat.r_obs) or ( np.isnan(r_min).any() ):
            invalid_freq = invalid_freq + 1

    print()
    print( invalid_freq/len(data_loan)  )
    print() 
    return invalid_freq/len(data_loan) 
   

for i in np.arange(-.01,.1,.01):
    init_guess[len(demand_spec) + len(bank_spec)] = i
    theta.set(init_guess)
    obj = parm_func(init_guess,theta,data_loan,market_df,df_mbs_com_sub_market)
    print(i,obj)
 
   
def parm_func_final(x):
    return parm_func(x,theta,data_loan,market_df,df_mbs_com_sub_market)
    
from scipy.optimize import differential_evolution

bounds = [  np.array((-1,1)) for i in range(len(demand_spec) + len(consumer_spec) + len(bank_spec)) ] 

result = differential_evolution( parm_func_final, bounds )

def constraint_check(p_guess,theta,data_loan,market_df,df_mbs_com_sub_market):
    
    # for each parameter guess
    for i in range(len(p_guess)):
        
        theta.set(p_guess[i]) 
        
        # initialize valid condition 
        valid = 1 
        print(p_guess[i])
                
        # for each consumer
        for j in range(len(data_loan)):

            # define consumer subset
            dat, mbs = consumer_subset(j,theta,data_loan,market_df,df_mbs_com_sub_market)
            
            # get break-even interest rate
            r_min = min_rate(dat, theta, mbs)
            
            # if negative or not profitable, break 
            if (r_min < 0).any() or (r_min[ dat.lender_obs ] > dat.r_obs) or ( np.isnan(r_min).any() ):
                valid = 0
                break 
            
        if valid == 1:
            print( p_guess[i] )
            #return p_guess[i]
            break  
            

                

init_guess = np.ones(len(demand_spec) + len(consumer_spec) + len(bank_spec))*0.000
theta.set(init_guess)


# theta.beta_d = theta.beta_d*0 + 0.02
theta.gamma_WS[:] = 0.0
theta.gamma_ZS[:] = 0.0
# theta.gamma_ZH[:] = - np.copy(theta.gamma_ZS)/2
# theta.gamma_WH[:] = -np.copy(theta.gamma_WS)/2
theta.gamma_ZH[0] = 0.0
#df_mbs_com_sub_market[ mbs_price_spec ] = 0.008889

#fval = evaluate_likelihood(init_guess,theta,data_loan,market_df,df_mbs_com_sub_market)

""" 
    Estimate Second Stage
"""


i = 0
dat, mbs = consumer_subset(i,theta,data_loan,market_df,df_mbs_com_sub_market)

for alpha in np.arange(-50,-1,2):
    r_eq, itr = solve_eq_robust(alpha,dat,theta,mbs)
    print(alpha,itr,r_eq)

fval = evaluate_likelihood(init_guess,theta,data_loan,market_df,df_mbs_com_sub_market)

init_guess[0:len(theta.beta_x)] = 10
val, res = estimate_NR(init_guess, theta,data_loan,market_df,df_mbs_com_sub_market)
val, res = estimate_GA(init_guess, theta,data_loan,market_df,df_mbs_com_sub_market)





