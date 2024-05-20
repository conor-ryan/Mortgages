import numpy as np
import scipy as sp
##### USEFUL MODEL TYPES ####

## Data Class 
# Contains consumer specific data necessary for evaluating the model
# An object to help parse incoming data frames and pass to functions
# (This is about readability not speed, and could be slow)
class Data:
    def __init__(self,i,X,W,D,Z,lender,r,out):
        self.i = i # Consumer Identifier (Mostly for tracking, debugging)
        self.X = X # Lender Demand Covariates 
        self.W = W # Lender Cost Covariates
        self.D = D # Discount Rate Covariates
        self.Z = Z # Consumer/Loan Characteristics

        self.lender_obs = lender # Observed Lender Index
        self.r_obs = r # Observed Rate Index 
        self.out = out # Consumer-Specific Outside option index

## MBS Price Function Class
# An object that translates MBS coupon rates and prices into a smooth, pricing function
class MBS_Func:
    def __init__(self,coupons,prices):
        self.coupon = coupons # Tracked Coupon Rates
        self.price = prices # Corresponding Prices for Coupon Rates
        self.func = sp.interpolate.CubicSpline(coupons,prices) # Interpolated Function
        
        
        
        ### Construct an Inverse Price function 
        ## Prices must be monotonically increasing for this to work
        ## Only helpful for locating zero-profit rates
        # First, sample interpolated function on a finer grid
        # min_coupon = np.min(coupons)
        # max_coupon = np.max(coupons)
        # coupon_evals = np.arange(min_coupon,max_coupon,0.00005) # Grid
        # price_evals = self.func.__call__(coupon_evals) # Predicted Price values
        # # Interpolate the inverse function
        # self.func_inv = sp.interpolate.PchipInterpolator(price_evals,coupon_evals)

        ### Define derivative functions
        # First Derivative
        self.der1 = self.func.derivative(nu=1)
        # Second Derivative
        self.der2 = self.func.derivative(nu=2)
        # Third Derivative 
        self.der3 = self.func.derivative(nu=3)

    ### Method to return price for a coupon 
        #Input: c - coupon rate (can be vector)
        #Output: MBS price
    def P(self,c):
        price = self.func(c)
        price[c<np.min(self.coupon)] = self.price[0]
        price[c>np.max(self.coupon)] = self.price[-1]
        return price
    ### Method to return coupon rate for a given MBS price (inverse of price function)
        # Input: p - MBS price (can be vector)
        # Output: coupon rate
    # def P_inv(self,p):
    #     coupon = self.func_inv.__call__(p)
    #     return coupon
    
    ### Method - first derivative of MBS price w.r.t. coupon rate
        #Input: c - coupon rate (can be vector)
        #Output: first derivative of MBS price   
    def dPdr(self,c):
        return self.der1.__call__(c)
    
    ### Method - second derivative of MBS price w.r.t. coupon rate
        #Input: c - coupon rate (can be vector)
        #Output: second derivative of MBS price  
    def d2Pdr2(self,c):
        return self.der2.__call__(c)
    
    ### Method - third derivative of MBS price w.r.t. coupon rate
        #Input: c - coupon rate (can be vector)
        #Output: third derivative of MBS price  
    def d3Pdr3(self,c):
        return self.der3.__call__(c)



## Second Stage Parameter Class 
# Object to hold, parse, update the parameters of the model while estimating 
class Parameters:

    def __init__(self,demand_spec,cost_spec,cons_spec,discount_spec, # Model Specifications
                 mbs_spec,mbs_coupons, # Model Specifications
                 rate_spec,lender_spec,market_spec,time_spec,out_spec, # Model Specifications
                 par_first_stage,# Parameters from First Stage estimation 
                 outside_share_vector,N_vector): # Vector of outside option shares and population size
        
        self.demand_spec = demand_spec # Variables that affect Consumer Demand
        self.cost_spec = cost_spec # Variables that affect the total "hold" cost
        self.cons_spec = cons_spec # Variables that affect consumer-specific cost
        self.discount_spec = discount_spec #Variables that determine discount rate
                                            #(must be same as first stage)

        self.mbs_spec = mbs_spec #Variables in MBS data that hold prices of each coupon
        self.mbs_coupons = mbs_coupons #The corresponding coupon rates of the above prices


        self.rate_spec = rate_spec # Observed interest rate for each loan
        self.lender_spec = lender_spec # Observed chosen lender
        self.market_spec = market_spec # Market indicator
        self.time_spec = time_spec # Time indicator
        self.out_index = out_spec  # Index for aggregate Outside option share o
        self.out_share = outside_share_vector.to_numpy() # Aggregate outside option share
        self.N = N_vector.to_numpy() # Vector of population size  

        self.beta_x = np.zeros(len(demand_spec)) # Initialize Demand Parameters
        self.gamma_WH = np.zeros(len(cost_spec)) # Initialize HTM Cost parameters
        self.gamma_ZH = np.zeros(len(cons_spec)) # Initialize HTM Cons/Loan cost parameters

        ### Transfer First Stage Parameters ### 
        self.beta_d = np.copy(par_first_stage.beta_d)
        self.sigma = np.copy(par_first_stage.sigma)
        self.gamma_WS = np.copy(par_first_stage.gamma_WS)
        self.gamma_ZS = np.copy(par_first_stage.gamma_ZS)

        ### Construct indices for pulling second-stage parameters from a combined parameter vector
        self.beta_x_ind = range(0,len(demand_spec)) # First - demand parameters
        self.gamma_WH_ind = range(len(demand_spec),len(demand_spec)+len(cost_spec)) # Second - Bank cost parameters
        self.gamma_ZH_ind = range(len(demand_spec)+len(cost_spec),len(demand_spec)+len(cost_spec)+len(cons_spec)) # Last - Consumer-loan specific parameters

    ### Method - Set parameters from a numpy vector 
        # Input: parameter_vector, numpy vector with appropriate length
        # Output: none
    def set(self,parameter_vector):
        self.beta_x = parameter_vector[self.beta_x_ind]
        self.gamma_WH = parameter_vector[self.gamma_WH_ind]
        self.gamma_ZH = parameter_vector[self.gamma_ZH_ind]

    ### Method - print the parameters 
        # Input: none
        # Ouput: none
    def summary(self):
        print(f"Sigma: {self.sigma:.3}") 
        print("Discount Par:", np.round(self.beta_d,3)) 
        print("Cons/Loan HTM Cost:", np.round(self.gamma_ZH,3)) 
        print("Bank OTD HTM Cost:", np.round(self.gamma_WH,2)) 
        print("Cons/Loan OTD Diff:", np.round(self.gamma_ZS,3)) 
        print("Bank OTD Diff:", np.round(self.gamma_WS,2)) 
        print("Demand Parameters:", np.round(self.beta_x,3))

    ### Method - return the combined vector that was used for the parameters
        # Input: none
        # Output: parameter_vector (numpy vector of appropriate length)
    def all(self):
        return np.concatenate( (self.beta_x,self.gamma_WH, self.gamma_ZH ))


