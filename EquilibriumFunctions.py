import numpy as np
import scipy as sp
import ModelFunctions
import Derivatives
##### Solution Algorithms for Bertrand Nash Equilibrium ######

## Equilibrium Solution Given Consumer Price Elasticity Parameter
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
# m - MBS pricing function 
# kwargs (r_start) - allow for an optional starting vector
def solve_eq(alpha,d,theta,m,model="base",**kwargs):
    # Define zero profit interest rate which will help bound feasible equi. interest rates
    r_min = ModelFunctions.min_rate(d,theta,m,model=model) 

    # Check if marginal cost pricing is an equilibrium 
    # Occurs if the probability of purchase is very small
    err_check = ModelFunctions.expected_foc(r_min,alpha,d,theta,m,model=model)
    if np.sum(np.square(err_check))<1e-12:
        return r_min, 0
    
    # Check if a starting vector was supplied
    r_start = kwargs.get("r_start")
    if r_start is None:
        # If r_start is not supplied...
        # Define Initial Price Vector as average markup over zero-profit rate   
        r = r_min + 1 /(-alpha)
    else:
        # Otherwise, use starting guess 
        r = r_start

    ## Algorithm follows newton-method, fast but not robust
        
    # Initial gradient evaluation
    # foc_k - first order condition value
    # grad_k - gradient of first order condition w.r.t. rates
    dump, grad_k, foc_k = Derivatives.d_foc(r,alpha,d,theta,m,model=model)

    # Initialize error and iteration number
    err = 1 
    itr = 0
    # Iterate until a tolerance of 1e-12 or maximum of 100 iterations
    while err>1e-12 and itr<1000:
        itr = itr+1
        # Newtown step for new r vector
        r_new = r - np.dot(np.linalg.inv(grad_k),foc_k)
        # Bound the r vector below by lowest profitable rate
        r_new = np.maximum(r_new,(r-r_min)*0.1 + r_min)
        # Compute the (bounded) step size
        step = r_new - r
        # Update r vector
        r = r_new 

        # Evaluate Objective (foc_k) and Gradient (grad_k)
        dump, grad_k, foc_k =  Derivatives.d_foc(r,alpha,d,theta,m,model=model)

        # Compute sum squared error of first order conditions
        err = np.sum(np.square(foc_k)) 

    # Return converged rate vector and iteration number 
    return r, itr

## Robust Equilibrium Solution Given Consumer Price Elasticity Parameter
## Slower but more certain convergence, especially important for large magnitudes of alpha
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
# m - MBS pricing function 
# kwargs (r_start) - allow for an optional starting vector
def solve_eq_robust(alpha,d,theta,m,model="base",**kwargs):
    # Define zero profit interest rate which will help bound feasible equi. interest rates
    r_min = ModelFunctions.min_rate(d,theta,m,model=model) 
    # Check if marginal cost pricing is an equilibrium 
    # Occurs if the probability of purchase is very small
    err_check = ModelFunctions.expected_foc(r_min,alpha,d,theta,m,model=model)
    if np.sum(np.square(err_check))<1e-12:
        return r_min, 0

    # Check if a starting vector was supplied
    r_start = kwargs.get("r_start")
    if r_start is None:
        # If r_start is not supplied...
        # Define Initial Price Vector as average markup over zero-profit rate   
        r = r_min + 1 /(-alpha)
    else:
        # Otherwise, use starting guess 
        r = r_start

    ## Algorithm is similar to a gradient ascent method
        
    # Initial evaluation of first order condition 
    foc_k = ModelFunctions.expected_foc_nonlinear(r,alpha,d,theta,m,model=model)
    # Set initial step size based on FOC magnitudes
    B_k_inv = 0.001/np.max(np.abs(foc_k)) 

    # Initial error and iteration number
    err = np.sum(np.square(foc_k))
    itr = 0
    # Iterate until a tolerance of 1e-12 or iterations exceed 200
    while err>1e-12 and itr<1000:
        itr = itr+1
        # Step in the direction of the first order condition (gradient of profit function)
        step = B_k_inv*foc_k

        # Bound the step-size to be relatively small
        max_adj = 0.01
        if any(np.abs(foc_k)>100):
            max_adj=0.001 # Even smaller step size if FOC is way off
        # Normalize the step by the maximum allowed step size
        if np.max(np.abs(step))>max_adj:
            step = max_adj*(step/np.max(np.abs(step)))
        # Compute new r vector
        r_new = r + step
        # Bound the r vector below by minimum profitable rate
        r_new = np.maximum(r_new,r_min)
        # Compute true step after bounds are applied
        step = r_new - r

        # Evaluate First Order Condition
        foc_next = ModelFunctions.expected_foc_nonlinear(r_new,alpha,d,theta,m,model=model)
        # Copy new r value into r variable
        r = np.copy(r_new )

        # Compute change in FOC value
        y_k = foc_next - foc_k

        # Set a default step size
        B_k_inv = np.repeat(0.0001,len(r))
        # Update step size for FOC values that updated 
        B_k_inv[y_k!=0] = np.abs((step[y_k!=0]/y_k[y_k!=0]))
        # Reset default step size for prices where the step was zero (potentially due to bound)
        B_k_inv[B_k_inv==0] = 0.0001

        foc_k = foc_next # Update prior function value
        err = np.sum(np.square(foc_k)) # Compute sum squared error 


    return r, itr

## Very Robust Equilibrium Solution Given Consumer Price Elasticity Parameter
## Very similar algorithm to solve_eq_robust, but with smaller step sizes that shrink as FOC approaches zero
# alpha - consumer specific price elasticity parameter
# d - data object
# theta - parameter object
# m - MBS pricing function 
# kwargs (r_start) - allow for an optional starting vector
def solve_eq_very_robust(alpha,d,theta,m,model="base",**kwargs):
    # Define zero profit interest rate which will help bound feasible equi. interest rates
    r_min = ModelFunctions.min_rate(d,theta,m,model=model) 

    # Check if a starting vector was supplied
    r_start = kwargs.get("r_start")
    if r_start is None:
        # If r_start is not supplied...
        # Define Initial Price Vector as average markup over zero-profit HTM rate   
        r = r_min + 1 /(-alpha)
    else:
        # Otherwise, use starting guess 
        r = r_start

    ## Algorithm is similar to a gradient ascent method
        
    # Initial evaluation of first order condition 
    foc_k = ModelFunctions.expected_foc_nonlinear(r,alpha,d,theta,m,model=model)
    # Set small initial step size 
    B_k_inv = np.repeat(1e-6,len(r))

    # Initial error and iteration number
    err = np.sum(np.square(foc_k))
    itr = 0
    # Iterate until a tolerance of 1e-12 or maximum iterations of 500
    while err>1e-12 and itr<500:
        itr = itr+1
        # Step in the direction of the first order condition (gradient of profit function)
        step = B_k_inv*foc_k

        # Bound the step-size to be relatively small
        max_adj = 0.01
        if any(np.abs(foc_k)>100):
            max_adj=0.001 # Even smaller step size if FOC is way off
        # Normalize the step by the maximum allowed step size
        if np.max(np.abs(step))>max_adj:
            step = max_adj*(step/np.max(np.abs(step)))
        # Compute new r vector
        r_new = r + step
        # Bound the r vector below by minimum profitable rate
        r_new = np.maximum(r_new,r_min)
        # Compute true step after bounds are applied
        step = r_new - r

        # Evaluate First Order Condition
        foc_next = ModelFunctions.expected_foc_nonlinear(r_new,alpha,d,theta,m,model=model)
        # Update r vector
        r = np.copy(r_new )

        # Products with where the FOC changes sign (signal that price is close to optimal)
        sign_flip = (foc_k*foc_next)<0

        # Default is to increase step size by 10%
        B_k_inv = B_k_inv*1.1
        # If total error is greater than 1e-5
        if err>1e-5:
            B_k_inv[sign_flip] = 1e-6 # Set "close" products to a small step size
        else:
            B_k_inv[sign_flip] = 1e-8 # Set smaller step size when error is smaller than 1e-5
        
        foc_k = foc_next # Update prior function value
        err = np.sum(np.square(foc_k)) # Compute sum squared error 
    # Return converged rate vector and iteration number 
    return r, itr


## Equilibrium Solution Given Observed Transaction Interest Rate
# Solve for alpha and all other unknown interest rates
# r0 - Observed interest rate
# j - Observed lender selection
# d - data object
# theta - parameter object
# m - MBS pricing function
def solve_eq_r(r0,j,d,theta,m,itr_max=100,model="base"):
    # Define zero profit interest rate which will help bound feasible equi. interest rates
    r_min = ModelFunctions.min_rate(d,theta,m,model=model) 

    #### Return Without solving if observed price is below marginal cost
    if r0 < r_min[j]:
        return 0.0,r_min,-1
    
    # Starting Value Guess for consumer price elasticity parameter
    alpha = -(1/(r0 - r_min[j]))

    ### Return without solving if one firm has huge MC advantage
    # Less important in the presence of an outside option
    q_init = ModelFunctions.market_shares(r_min,alpha,d,theta)
    if any(q_init>1-1e-4):
        return alpha,r_min,-2
    
    # Initial equilibrium solution given starting guess
    r, itr = solve_eq(alpha,d,theta,m,model=model)

    ### Check advantage again after initial equilibrium
    q_init = ModelFunctions.market_shares(r,alpha,d,theta)
    if any(q_init>1-1e-4):
        return alpha,r_min,-2

    # Set initial "parameter" vector
    # Consists of the equlibrium interest rates
    # But, replace the observed interest rate with unknown alpha
    # Still J unknown parameters and J first order conditions
    x = np.copy(r)
    x[j] = alpha
    r[j] = r0
    ## Algorithm follows newton method
    # Initial function and gradient evaluation
    # foc_k - first order condition value (objective function)
    # grad_k - gradient of first order condition w.r.t. rates
    # grad_alpha - gradient of foc w.r.t. alpha
    grad_alpha, grad_k, foc_k =  Derivatives.d_foc(r,alpha,d,theta,m,model=model)
    # Replace the gradient w.r.t. observed rate with the gradient w.r.t. alpha
    grad_k[j,:] = grad_alpha

    # Initialize error value and iteration count
    err = 1
    itr = 0
    # Iterate until a tolerance of 1e-12 or maximum iterations of 100
    while err>1e-12 and itr<itr_max:
        itr +=1
        # Update parameters with a newton step
        x = x - np.dot(np.transpose(np.linalg.inv(grad_k)),foc_k)
        # Set interest rates from parameters, substituting in the observed rate.
        r = np.copy(x)
        r[j] = r0
        # Update alpha
        alpha_new = x[j]

        ### Bound Parameter Steps
        # alpha must be negative (otherwise no equilibrium, should not be binding)
        alpha_new = min(alpha_new,-0.1)
        # Limit size of alpha step to 100. 
        if (alpha - alpha_new)>100:
            alpha_new = alpha - 100
        # Update parameter vector and alpha after applying bounds
        x[j] = alpha_new
        alpha = np.copy(alpha_new)
        
        # # Bound the r vector below by lowest profitable rate
        r = np.maximum(r,r_min)

        # Evaluate foc and gradients
        grad_alpha, grad_k, foc_k =  Derivatives.d_foc(r,alpha,d,theta,m,model=model)
        grad_k[j,:] = grad_alpha # Substitute alpha gradient for observed rate gradient

        err = np.sum(np.square(foc_k)) # Compute sum squared error 

    # Return converged alpha parameter, equilibrium interest rate vector, and iterations
    return alpha ,r, itr

## Robust Equilibrium Solution Given Observed Transaction Interest Rate
## Slower algorithm with more robust convergence properties
# Solve for alpha and all other unknown interest rates
# r0 - Observed interest rate
# j - Observed lender selection
# d - data object
# theta - parameter object
# m - MBS pricing function
def solve_eq_r_robust(r0,j,d,theta,m,model="base"):
    # Compute zero profit interest rate which will help bound feasible equi. interest rates
    r_min = ModelFunctions.min_rate(d,theta,m,model=model) 

    # Return Without solving if observed price is below marginal cost
    if r0 < r_min[j]:
        return 0.0,r_min,-1
    

    # Starting Value Guess for consumer price elasticity parameter
    alpha = -(1/(r0 - r_min[j]))

    # Solve equilibrium at initial alpha guess
    # Apply increasingly robust solution methods for greater values of alpha
    if alpha>(-800):
        r, itr = solve_eq(alpha,d,theta,m,model=model)
    elif alpha>(-10000):
        r, itr = solve_eq_robust(alpha,d,theta,m,model=model)
    else: 
        r, itr = solve_eq_very_robust(alpha,d,theta,m,model=model)

    # Initialize step size for alpha
    B_k_inv = 10
    # Objective function: Difference between observed rate and predicted rate
    f_k = r[j] - r0
    # Initial error value
    err = f_k**2
    # Iterate until a tolerance of 1e-12 or maximum iterations of 4000
    while (err>1e-12) & (itr<12000):
        # Step in opposite direction of objective function
        # (If predicted r is too high, alpha should become more negative/more elastic)
        step = -B_k_inv*f_k   
        # Bound the step size by -500 (don't decrease too fast)
        step = np.maximum(step,-500)
        # Compute new alpha
        alpha_new = alpha + step
        # Bound alpha at zero (otherwise no equilibrium, should not bind)
        alpha_new = min(-0.1,alpha_new)
        #Compute step size after applying bounds
        step = alpha_new - alpha
        # Update alpha after applying bounds
        alpha = alpha_new

        # Solve equilibrium at new alpha guess
        # Apply increasingly robust solution methods for greater values of alpha
        if alpha>(-900):
            r, i_new = solve_eq(alpha,d,theta,m,model=model)
        elif alpha>(-10000):
            r, i_new = solve_eq_robust(alpha,d,theta,m,model=model)
        else: 
            r, i_new = solve_eq_very_robust(alpha,d,theta,m,model=model)

        # Update total number of iterations
        itr+= i_new
        # Compute objective function 
        f_next = r[j] - r0

        # Compute change in objective function to update step size
        y_k = f_next - f_k
        # Update step size 
        B_k_inv = np.abs((step/y_k))
        
        f_k = f_next # Update function value
        err = f_k**2 # Update error 

    # Error flags that indicate model is struggling
    # Most robust method not returning convergence
    if itr>=12000:
        if alpha>(-20000):
            print("No Convergence at low alpha",alpha,d.i)
    return alpha,r, itr


def solve_eq_optim(alpha,d,theta,m,model="base",**kwargs):
    # Define zero profit interest rate which will help bound feasible equi. interest rates
    r_min = ModelFunctions.min_rate(d,theta,m,model=model) 

    # Check if marginal cost pricing is an equilibrium 
    # Occurs if the probability of purchase is very small
    err_check = ModelFunctions.expected_foc(r_min,alpha,d,theta,m,model=model)
    if np.sum(np.square(err_check))<1e-12:
        return r_min, 0
    
    # Check if a starting vector was supplied
    r_start = kwargs.get("r_start")
    if r_start is None:
        # If r_start is not supplied...
        # Define Initial Price Vector as average markup over zero-profit rate   
        r = r_min + 1 /(-alpha)
    else:
        # Otherwise, use starting guess 
        r = r_start

    def obj_fun(x):
        dump, grad_k, foc_k = Derivatives.d_foc(x,alpha,d,theta,m,model=model)
        return foc_k, grad_k
    
    res = sp.optimize.root(obj_fun,r,jac=True)

    return res.x, res.nfev, res.success


def solve_eq_r_optim(r0,j,d,theta,m,model="base"):
    # Try Fast Method
    itr_max = 50
    alpha, r, itr = solve_eq_r(r0,j,d,theta,m,itr_max=itr_max,model=model)
    if itr<itr_max:
        if alpha < theta.alpha_min:
            alpha = theta.alpha_min
            r = ModelFunctions.min_rate(d,theta,m,model=model) - 1/theta.alpha_min
        return alpha, r, itr, True
    print("Using Optim",d.i)
    # Define zero profit interest rate which will help bound feasible equi. interest rates
    r_min = ModelFunctions.min_rate(d,theta,m,model=model)
    if model=="base": 
        prof, dprof = ModelFunctions.dSaleProfit_dr(np.repeat(r0,len(r_min)),d,theta,m)
    elif model=="hold":
        prof, dprof = ModelFunctions.dHoldOnly_dr(np.repeat(r0,len(r_min)),d,theta)
    

    alpha_max = -dprof[j]/prof[j]

    upper_bound = np.repeat(0.5,len(r_min))
    lower_bound = np.copy(r_min)
    upper_bound[j] = alpha_max
    lower_bound[j] = -3000
    bounds = [[lower_bound[i],upper_bound[i]] for i in range(len(upper_bound)) ]

    #### Return Without solving if observed price is below marginal cost
    if r0 < r_min[j]:
        return 0.0,r_min,-1
    
    # Starting Value Guess for consumer price elasticity parameter
    alpha_init = alpha_max - 10
    r_init = r_min + 1 /(-alpha_init)

    def obj_fun(x):
        r = np.copy(x)
        r[j] = r0
        alpha = x[j]
        f = ModelFunctions.expected_foc(r,alpha,d,theta,m,model=model)
        g_alpha, grad, f = Derivatives.d_foc(r,alpha,d,theta,m,model=model)
        grad[j,:] = g_alpha
    
        obj = np.inner(f,f)
        obj_grad = np.dot(f,np.transpose(grad)) +  np.dot(grad,np.transpose(f))
        return obj, obj_grad

    # def obj_fun_hess(x):
    #     r = np.copy(x)
    #     r[j] = r0
    #     alpha = x[j]
    #     f = ModelFunctions.expected_foc(r,alpha,d,theta,m)
    #     grad, d1, hess,d2,d3 = Derivatives.d2_foc_all_parameters(r,alpha,d,theta,m)
    
    #     obj = np.inner(f,f)
    #     obj_grad = np.dot(f,np.transpose(grad)) +  np.dot(grad,np.transpose(f))

    #     obj_hess = np.zeros((len(f),len(f)))
    #     for i in range(len(f)):
    #         obj_hess[i,:] = np.dot(grad[i,:],np.transpose(grad)) + np.dot(f,np.transpose(hess[:,i,:])) +  np.dot(hess[:,i,:],np.transpose(f))+  np.dot(grad,np.transpose(grad[i,:]))
        
    #     return  obj_hess
    
    x_start = np.copy(r_init)
    x_start[j] = alpha_init
    res = sp.optimize.minimize(obj_fun,x_start,jac=True,method="TNC",bounds = bounds,tol=1e-12)
    alpha = res.x[j]
    r = np.copy(res.x)
    r[j] = r0


    if alpha < theta.alpha_min:
        alpha = theta.alpha_min
        r = ModelFunctions.min_rate(d,theta,m,model=model) - 1/theta.alpha_min

    return alpha,r, res.nfev, res.success


