import numpy as np
import scipy as sp
##### OTD & HTM PROFIT FUNCTIONS #####
    
## Profit from holding mortgage on balance sheet
# r - interest rate
# theta - parameter object
# d - data object   
# def pi_hold(r,theta,d):
#     cost = np.dot(d.W,theta.gamma_W)
#     prof =  r - cost
#     return prof 

# ## Minimum Profitable Interest Rate from HTM lending
# ## (Useful for characterizing expected marginal cost)
# # theta - parameter object
# # d - data object   

# def r_hold_min(theta,d):
#     cost = np.dot(d.W,theta.gamma_W)
#     return cost

# ## Derivative of HTM profit w.r.t. Interest Rate
# # r - interest rate
# # theta - parameter object
# # d - data object   
# def dpidr_hold(r,theta,d):
#     return 1 

## Profit from selling mortgage
# r - interest rate
# theta - parameter object
# d - data object   
# m - MBS pricing function
# def pi_sell(r,theta,d,m):
#     bank_cost = np.dot(d.W,(theta.gamma_WH + theta.gamma_WS) ) # sum of HTM and OTD parameters
#     cost = np.dot(np.transpose(d.Z),(theta.gamma_ZH + theta.gamma_ZS) ) # sum of HTM and OTD parameters
#     # Service fee revenue currently excluded (absorbed?)
#     return m.P(r - 0.0025) - credit_cost - bank_cost


## Derivative of OTD profit w.r.t. Interest Rate
# r - interest rate
# m - MBS pricing function (may change this when using MBS data)
# def dpidr_sell(r,m):
#     return m.dPdr(r - 0.0025)


def pi_sell_diff(r,theta,d):
    diff_h = theta.psi*r + np.dot(d.W,theta.gamma_diff)
    adj = np.minimum(0,-diff_h/theta.sigma)
    diff_exp = np.exp(diff_h/theta.sigma + adj)
    prob_h = diff_exp/(np.exp(adj) + diff_exp)
    diff = theta.sigma*(np.log(np.exp(adj) + diff_exp) - np.log(2))
    return diff, prob_h

## Minimum Profitable Interest Rate from OTD lending
## (Useful for characterizing expected marginal cost)
# theta - parameter object
# d - data object   
# m - MBS pricing function (may change this when using MBS data)
# def r_sell_min(theta,d,m):
#     bank_cost = np.dot(d.W,(theta.gamma_WH + theta.gamma_WS) ) # sum of HTM and OTD parameters
#     credit_cost = np.dot(np.transpose(d.Z),(theta.gamma_ZH + theta.gamma_ZS) ) # sum of HTM and OTD parameters
#     r_min = m.P_inv(credit_cost + bank_cost) +0.0025
#     return r_min


###### DEMAND FUNCTIONS #####

## Market Share Function (Product-Consumer-specific demand)
# r - vector of interest rates in the market
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
def market_shares(r,alpha,d,theta,return_bound=False):
    # Bounds tolerance on shares so log() doesn't crash
    tol = 1e-12

    # Utility Specification
    util = np.dot(d.X,theta.beta_x) + alpha*r
    if all(util<-50):
        s = np.repeat(tol/len(util),len(util))
        bound_flag  = 1
    else:
        norm_util = -np.maximum(max(util),0.0) # Normalization so exp() doesn't crash
        if all(util<-10):
            norm_util = -np.ceil(np.max(util)/10)*10
        out = np.exp(0.0 + norm_util)
        util = util + norm_util

        # Logit Share Probabilities
        eu = np.exp(util)
        s = eu/(out + np.sum(eu))
        
        bound_flag = 0 
        if any(s<tol):
            s = s*(1-tol) + tol*(sum(s)/len(s))
            bound_flag  = 1
        if ((1-sum(s))<tol):
            s = (s/sum(s))*(1-tol) 
            bound_flag  = 1
        if sum(s)<tol:
            s = np.repeat(tol/len(s),len(s))
            bound_flag  = 1
     
    if return_bound:
        return s, bound_flag
    else:
        return s 

def conditional_shares(r,alpha,d,theta,return_bound=False):
    # Utility Specification
    util = np.dot(d.X,theta.beta_x) + alpha*r
    norm_util = -np.maximum(max(util),0.0) # Normalization so exp() doesn't crash
    if all(util<-10):
        norm_util = -np.ceil(np.max(util)/10)*10
    util = util + norm_util

    # Logit Share Probabilities
    eu = np.exp(util)
    s = eu/(np.sum(eu))
    
    return s 

def conditional_likelihood(r,alpha,d,theta,return_bound=False):
    # Utility Specification
    util = np.dot(d.X,theta.beta_x) + alpha*r
    norm_util = -np.maximum(max(util),0.0) # Normalization so exp() doesn't crash

    if all(util<-10):
        norm_util = -np.ceil(np.max(util)/10)*10

    util = util + norm_util


    # Logit Share Probabilities
    eu = np.exp(util)
    ll =util - np.log(np.sum(eu))
    
    return ll 


## Market Share Derivatives (Product-Consumer-specific demand)
# r - vector of interest rates in the market
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object

def share_derivative(r,alpha,d,theta):
    # Utility Specification
    util = np.dot(d.X,theta.beta_x) + alpha*r 
    # Logit Share Probabilities
    eu = np.exp(util)
    s = eu/(1+np.sum(eu))
    # Logit Derivatives (own-price only, all single product firms)
    dq = alpha*s*(1 -s)
    # Return both shares and derivatives
    return s, dq

##### PROFIT MAXIMIZATION FUNCTIONS ######

## Expected Profit Conditional on a Sale- each firm for a particular consumer 
# r - interest rate vector
# d - data object
# theta - parameter object
# m - MBS pricing function (may change this when using MBS data)

def expected_sale_profit(r,d,theta):
    min_rate = np.dot(d.W,theta.gamma_WH)
    diff_min, prob_h_min = pi_sell_diff(min_rate,theta,d)
    cost = min_rate + diff_min

    diff, prob_h = pi_sell_diff(r,theta,d)
    ExPi = r - cost + diff

    # # Profit from an origination 
    # pi_h = pi_hold(r,theta,d) # HTM
    # pi_s = pi_sell(r,theta,d,m) # OTD
    # pi_max = np.maximum(pi_h,pi_s)

 
    # # Exponential of Expected Origination Profit (pre-computation)
    # epi_h = np.exp((pi_h-pi_max)/theta.sigma)
    # epi_s = np.exp((pi_s-pi_max)/theta.sigma)

    # # Probability of Hold vs Sell
    # prob_h = epi_h/(epi_h+epi_s)
    # prob_s = 1 - prob_h
    # # Profit expectation over future balance sheet shock
    # ExPi = prob_h*pi_h + prob_s*pi_s

    # Currently, epsilon shock value isn't included in profitability
    # This matters for infra-marginal earnings, not exactly sure what is correct
    # Other possible formulation is below (but would need to update derivatives)
    # ExPi = theta.sigma*(np.log(np.exp(pi_h/theta.sigma) + np.exp(pi_s/theta.sigma)) - np.log(2))

    return ExPi


def dSaleProfit_dr(r,d,theta):

    cost = np.dot(d.W,theta.gamma_WH)
    # diff_min, prob_h_min = pi_sell_diff(min_rate,theta,d)
    # cost = min_rate + diff_min

    diff, prob_h = pi_sell_diff(r,theta,d)
    ExPi = r - cost + diff

    # Derivative of Hold Probability w.r.t. own interest rate
    dProb_h_dr = theta.psi*prob_h*(1-prob_h)/theta.sigma

    dEPidr = 1 + prob_h*theta.psi

    return ExPi,dEPidr

def d2SaleProfit_dr2(r,d,theta):

    cost = np.dot(d.W,theta.gamma_WH)
    # diff_min, prob_h_min = pi_sell_diff(min_rate,theta,d)
    # cost = min_rate + diff_min

    diff, prob_h = pi_sell_diff(r,theta,d)
    ExPi = r - cost + diff

    dProb_h_dr = theta.psi*prob_h*(1-prob_h)/theta.sigma
    d2Prob_h_dr2 = (theta.psi/theta.sigma)*dProb_h_dr*(1-2*prob_h)

    # Linearized to isolate alpha and q
    dEPidr = 1 + prob_h*theta.psi 
    d2EPidr2 = dProb_h_dr*theta.psi 

    return ExPi,dEPidr,d2EPidr2


def d3SaleProfit_dr3(r,d,theta):

    cost = np.dot(d.W,theta.gamma_WH)
    # diff_min, prob_h_min = pi_sell_diff(min_rate,theta,d)
    # cost = min_rate + diff_min

    diff, prob_h = pi_sell_diff(r,theta,d)
    ExPi = r - cost + diff

    # Derivative of Hold Probability w.r.t. own interest rate
    dProb_h_dr = theta.psi*prob_h*(1-prob_h)/theta.sigma
    d2Prob_h_dr2 = (theta.psi/theta.sigma)*dProb_h_dr*(1-2*prob_h)
    # d3Prob_h_dr3 = (theta.psi/theta.sigma)*(d2Prob_h_dr2*(1-2*prob_h) - 2*dProb_h_dr)

    # Linearized to isolate alpha and q
    dEPidr = 1 + prob_h*theta.psi 
    d2EPidr2 = dProb_h_dr*theta.psi 
    d2EPidr3 = d2Prob_h_dr2*theta.psi

    return ExPi,dEPidr,d2EPidr2,d2EPidr3

## First Order Condition - Maximization condition for expected profit
# r - interest rate vector
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
# m - MBS pricing function (may change this when using MBS data)
def expected_foc(r,alpha,d,theta,model="base"):

    # Profit from an origination 
    # if model=="base":
    pi,dpi_dr = dSaleProfit_dr(r,d,theta)
    # elif model=="hold":
    #     pi,dpi_dr = dHoldOnly_dr(r,d,theta)

    # Market Shares and Derivatives
    q = market_shares(r,alpha,d,theta)

    # Linearized to isolate alpha and q
    dEPidr = (dpi_dr)/(alpha*(1-q)) + pi

    return dEPidr

## First Order Condition - Not Linearized 
## This is useful for more robust equilibrium methods 
# r - interest rate vector
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
# m - MBS pricing function (may change this when using MBS data)
def expected_foc_nonlinear(r,alpha,d,theta,model="base"):

    # Profit from an origination 
    # if model=="base":
    pi,dpi_dr = dSaleProfit_dr(r,d,theta)
    # elif model=="hold":
    #     pi,dpi_dr = dHoldOnly_dr(r,d,theta)

    # Market Shares and Derivatives
    q = market_shares(r,alpha,d,theta)

    # Deriviative of profit w.r.t. own rates
    dEPidr = (dpi_dr)*q + (alpha*q*(1-q))*(pi)

 
    return dEPidr


## Approximate Marginal Cost - Useful for determining if parameters are feasible
## True, zero-profit interest rate doesn't have closed form solution
# d - data object
# theta - parameter object
# m - MBS pricing function 
def approx_mc(d,theta):
    # Minimum Interest Rates
    r_h_min = np.dot(d.W,theta.gamma_WH)
    # r_s_min = r_sell_min(theta,d,m)

    # mc = np.minimum(r_h_min,r_s_min)

    return r_h_min

## True, lowest profitable rate - Useful for determining if parameters are feasible
## No closed form solution, find zero using secant method
# d - data object
# theta - parameter object
# m - MBS pricing function 
def min_rate(d,theta,model="base"):
    # def obj_fun(x):
    #     return target_markup(x,0.0,d,theta,m,model=model)
    
    # res = sp.optimize.root(obj_fun,np.repeat(d.r_obs,d.X.shape[0]))
    return np.dot(d.W,theta.gamma_WH)


## Hold Only First Order Condition - Maximization with no OTD option
# r - interest rate vector
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object

# def hold_only_foc(r,alpha,d,theta,):

#     # Profit from an origination 
#     pi_h = pi_hold(r,theta,d) # HTM

#     # Derivative of origination profit
#     dpi_h = dpidr_hold(r,theta,d) #HTM

#     # Market Shares and Derivatives
#     q,dqdp = share_derivative(r,alpha,d,theta)

#     dPidr = dqdp*pi_h + q*dpi_h
 
#     return dPidr


# Balance Sheet Allocation - decision to sell or hold a particular mortgage
# r - interest rate vector
# d - data object
# theta - parameter object
# m - MBS pricing function (may change this when using MBS data)
def balance_sheet_alloc(r,d,theta):

    diff_h = theta.psi*r + np.dot(d.W,theta.gamma_diff)
    diff_exp = np.exp(diff_h/theta.sigma)
    prob_h = diff_exp/(1 + diff_exp)

    return 1-prob_h


# def target_markup(r,target,d,theta,m,model="base"):
#     if model=="base":
#         val = expected_sale_profit(r,d,theta,m) - target
#     elif model=="hold":
#         val = pi_hold(r,theta,d)- target
#     return val

# ######## Hold Only Profit Functions #######
# def dHoldOnly_dr(r,d,theta):

#     # Profit from an origination 
#     pi_h = pi_hold(r,theta,d) # HTM

#     # Derivative of origination profit
#     dpi_h = dpidr_hold(r,theta,d) #HTM

#     return pi_h,np.repeat(dpi_h,len(pi_h))

# def d2HoldOnly_dr2(r,d,theta):

#     # Profit from an origination 
#     pi_h = pi_hold(r,theta,d) # HTM

#     # Derivative of origination profit
#     dpi_h = dpidr_hold(r,theta,d) #HTM
    
#     return pi_h,np.repeat(dpi_h,len(pi_h)),np.repeat(0,len(pi_h))


# def d3HoldOnly_dr3(r,d,theta):

#         # Profit from an origination 
#     pi_h = pi_hold(r,theta,d) # HTM

#     # Derivative of origination profit
#     dpi_h = dpidr_hold(r,theta,d) #HTM
    
#     return pi_h,np.repeat(dpi_h,len(pi_h)),np.repeat(0,len(pi_h)),np.repeat(0,len(pi_h))