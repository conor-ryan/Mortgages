import numpy as np

##### OTD & HTM PROFIT FUNCTIONS #####
    
## Profit from holding mortgage on balance sheet
# r - interest rate
# theta - parameter object
# d - data object   
def pi_hold(r,theta,d):
    discount_rate = np.dot(np.transpose(d.D),theta.beta_d)
    bank_cost = np.dot(d.W,theta.gamma_WH)
    credit_cost = np.dot(np.transpose(d.Z),theta.gamma_ZH)
    prof =  r/(discount_rate) - credit_cost - bank_cost
    return prof 

## Minimum Profitable Interest Rate from HTM lending
## (Useful for characterizing expected marginal cost)
# theta - parameter object
# d - data object   

def r_hold_min(theta,d):
    discount_rate = np.dot(np.transpose(d.D),theta.beta_d)
    bank_cost = np.dot(d.W,theta.gamma_WH)
    credit_cost = np.dot(np.transpose(d.Z),theta.gamma_ZH)
    return (credit_cost + bank_cost)*discount_rate

## Derivative of HTM profit w.r.t. Interest Rate
# r - interest rate
# theta - parameter object
# d - data object   
def dpidr_hold(r,theta,d):
    discount_rate = np.dot(np.transpose(d.D),theta.beta_d)
    return 1 /(discount_rate)

## Profit from selling mortgage
# r - interest rate
# theta - parameter object
# d - data object   
# m - MBS pricing function
def pi_sell(r,theta,d,m):
    bank_cost = np.dot(d.W,(theta.gamma_WH + theta.gamma_WS) ) # sum of HTM and OTD parameters
    credit_cost = np.dot(np.transpose(d.Z),(theta.gamma_ZH + theta.gamma_ZS) ) # sum of HTM and OTD parameters
    # Service fee revenue currently excluded (absorbed?)
    return m.P(r - 0.0025) - credit_cost - bank_cost


## Derivative of OTD profit w.r.t. Interest Rate
# r - interest rate
# m - MBS pricing function (may change this when using MBS data)
def dpidr_sell(r,m):
    return m.dPdr(r - 0.0025)

## Minimum Profitable Interest Rate from OTD lending
## (Useful for characterizing expected marginal cost)
# theta - parameter object
# d - data object   
# m - MBS pricing function (may change this when using MBS data)
def r_sell_min(theta,d,m):
    bank_cost = np.dot(d.W,(theta.gamma_WH + theta.gamma_WS) ) # sum of HTM and OTD parameters
    credit_cost = np.dot(np.transpose(d.Z),(theta.gamma_ZH + theta.gamma_ZS) ) # sum of HTM and OTD parameters
    r_min = m.P_inv(credit_cost + bank_cost) +0.0025
    return r_min


###### DEMAND FUNCTIONS #####

## Market Share Function (Product-Consumer-specific demand)
# r - vector of interest rates in the market
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
def market_shares(r,alpha,d,theta):
    # Utility Specification
    util = np.dot(d.X,theta.beta_x) + alpha*r
    max_util = np.maximum(max(util),0.0) # Normalization so exp() doesn't crash
    out = np.exp(0.0 - max_util)
    util = util - max_util

    # Logit Share Probabilities
    eu = np.exp(util)
    s = eu/(out + np.sum(eu))
    
    # Bounds on shares so log() doesn't crash
    tol = 1e-15
    if any(s<tol):
        s = s*(1-tol) + tol*(sum(s)/len(s))
    if ((1-sum(s))<tol):
        s = (s/sum(s))*(1-tol) 
    if sum(s)<tol:
        s = s + tol  
     
    return s 

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

def expected_sale_profit(r,d,theta,m):

    # Profit from an origination 
    pi_h = pi_hold(r,theta,d) # HTM
    pi_s = pi_sell(r,theta,d,m) # OTD

 
    # Exponential of Expected Origination Profit (pre-computation)
    max_prof = np.maximum(pi_h,pi_s)
    epi_h = np.exp((pi_h-max_prof)/theta.sigma)
    epi_s = np.exp((pi_s-max_prof)/theta.sigma)

    # Probability of Hold vs Sell
    prob_h = epi_h/(epi_h+epi_s)
    prob_s = 1 - prob_h
    # Profit expectation over future balance sheet shock
    ExPi = prob_h*pi_h + prob_s*pi_s

    # Currently, epsilon shock value isn't included in profitability
    # This matters for infra-marginal earnings, not exactly sure what is correct
    # Other possible formulation is below (but would need to update derivatives)
    # ExPi = theta.sigma*(np.log(np.exp(pi_h/theta.sigma) + np.exp(pi_s/theta.sigma)) - np.log(2))

    return ExPi

## First Order Condition - Maximization condition for expected profit
# r - interest rate vector
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
# m - MBS pricing function (may change this when using MBS data)
def expected_foc(r,alpha,d,theta,m):

    # Profit from an origination 
    pi_h = pi_hold(r,theta,d) # HTM
    pi_s = pi_sell(r,theta,d,m) # OTD

    # Derivative of origination profit
    dpi_h = dpidr_hold(r,theta,d) #HTM
    dpi_s = dpidr_sell(r,m) #OTD

    # Market Shares and Derivatives
    q = market_shares(r,alpha,d,theta)

    # Exponential of Expected Origination Profit (pre-computation)
    epi_h = np.exp(pi_h/theta.sigma)
    epi_s = np.exp(pi_s/theta.sigma)

    # Probability of Hold vs Sell
    prob_h = epi_h/(epi_h+epi_s)
    prob_s = 1 - prob_h

    # Linearized to isolate alpha and q
    dEPidr = (prob_h*dpi_h+prob_s*dpi_s)/(alpha*(1-q)) + prob_h*pi_h + prob_s*pi_s

    return dEPidr

## First Order Condition - Not Linearized 
## This is useful for more robust equilibrium methods 
# r - interest rate vector
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
# m - MBS pricing function (may change this when using MBS data)
def expected_foc_nonlinear(r,alpha,d,theta,m):

    # Profit from an origination 
    pi_h = pi_hold(r,theta,d) # HTM
    pi_s = pi_sell(r,theta,d,m) # OTD

    # Derivative of origination profit
    dpi_h = dpidr_hold(r,theta,d) #HTM
    dpi_s = dpidr_sell(r,m) #OTD

    # Market Shares and Derivatives
    q = market_shares(r,alpha,d,theta)

    # Exponential of Expected Origination Profit (pre-computation)
    epi_h = np.exp(pi_h/theta.sigma)
    epi_s = np.exp(pi_s/theta.sigma)

    # Probability of Hold vs Sell
    prob_h = epi_h/(epi_h+epi_s)
    prob_s = 1 - prob_h

    # Deriviative of profit w.r.t. own rates
    dEPidr = (prob_h*dpi_h+prob_s*dpi_s)*q + (alpha*q*(1-q))*(prob_h*pi_h + prob_s*pi_s)

 
    return dEPidr

## Approximate Marginal Cost - Useful for determining if parameters are feasible
## True, zero-profit interest rate doesn't have closed form solution
# d - data object
# theta - parameter object
# m - MBS pricing function 
def approx_mc(d,theta,m):
    # Minimum Interest Rates
    r_h_min = r_hold_min(theta,d)
    # r_s_min = r_sell_min(theta,d,m)

    # mc = np.minimum(r_h_min,r_s_min)

    return r_h_min

## True, lowest profitable rate - Useful for determining if parameters are feasible
## No closed form solution, find zero using secant method
# d - data object
# theta - parameter object
# m - MBS pricing function 
def min_rate(d,theta,m):
    # Minimum Interest Rates, by lending type
    r_h_min = r_hold_min(theta,d) # HTM lowest rate
    # r_s_min = r_sell_min(theta,d,m) # OTD lowest rate
    
    # Minimum possible rate
    lowest_rate = r_h_min*0.5 #np.minimum(r_h_min,r_s_min)
    # Initialize a guess for a rate that is profitable
    highest_rate = (lowest_rate+.01)*3
    # Compute profit at each rate guess
    low_profit = expected_sale_profit(lowest_rate,d,theta,m)
    high_profit = expected_sale_profit(highest_rate,d,theta,m)

    # First step of the secant method
    mc_n_1 = lowest_rate - low_profit* (lowest_rate - highest_rate)/(low_profit - high_profit)
    prof_n_1 = expected_sale_profit(mc_n_1,d,theta,m)
    # Initialize error
    err = np.sum(prof_n_1**2)
    # Save terms for next step of secant method
    mc_n_2 = lowest_rate
    prof_n_2 = low_profit
    mc = mc_n_1
    # Iterate until error is less than tolerance 
    while err>1e-10:
        # Secant method step
        mc = mc_n_1 - prof_n_1*(mc_n_1-mc_n_2)/(prof_n_1 - prof_n_2)
        prof = expected_sale_profit(mc,d,theta,m)
        # Recompute error
        err = np.sum(prof**2)

        # Save for next step
        mc_n_2 = np.copy(mc_n_1)
        mc_n_1 = np.copy(mc)
        prof_n_2 = np.copy(prof_n_1)
        prof_n_1 = np.copy(prof)
        
    return mc


## Hold Only First Order Condition - Maximization with no OTD option
# r - interest rate vector
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object

def hold_only_foc(r,alpha,d,theta,):

    # Profit from an origination 
    pi_h = pi_hold(r,theta,d) # HTM

    # Derivative of origination profit
    dpi_h = dpidr_hold(r,theta,d) #HTM

    # Market Shares and Derivatives
    q,dqdp = share_derivative(r,alpha,d,theta)

    dPidr = dqdp*pi_h + q*dpi_h
 
    return dPidr


## Balance Sheet Allocation - decision to sell or hold a particular mortgage
# r - interest rate vector
# d - data object
# theta - parameter object
# m - MBS pricing function (may change this when using MBS data)
def balance_sheet_alloc(r,d,theta,m):
    # Profit from each origination method
    pi_h = pi_hold(r,theta,d) # HTM
    pi_s = pi_sell(r,theta,d,m) # OTD

    # Logit Binary Probability Function 
    sell = np.exp(pi_s/theta.sigma)/(np.exp(pi_h/theta.sigma) + np.exp(pi_s/theta.sigma))
    return sell





