import scipy as sp
import numpy as np
## MBS Price Function Class
# Currenly, just another discount factor 
# slight quadratic curve to prevent colinearity 
class MBS_Func:
    def __init__(self,coupons,prices):
        self.coupon = coupons
        self.price = prices
        self.func = sp.interpolate.PchipInterpolator(coupons,prices)

        ### More detail for an inverse function
        min_coupon = np.min(coupons)
        max_coupon = np.max(coupons)
        coupon_evals = np.arange(min_coupon,max_coupon,0.00005)
        price_evals = self.func.__call__(coupon_evals)
        self.func_inv = sp.interpolate.PchipInterpolator(price_evals,coupon_evals)

        ### Derivative Splines
        self.der1 = self.func.derivative(nu=1)
        self.der2 = self.func.derivative(nu=2)
        self.der3 = self.func.derivative(nu=3)

    def P(self,c):
        price = self.func.__call__(c)
        return price
    def P_inv(self,p):
        coupon = self.func_inv.__call__(p)
        return coupon
    def dPdr(self,c):
        return self.der1.__call__(c)
    def d2Pdr2(self,c):
        return self.der2.__call__(c)
