import numpy as np

## Per Period Profit of a Bank
# Marginal cost of deposits wrapped into mc for loans
def per_period_revenue(
                    r_dep,r_cons,r_comm,p_inv,p_ins,
                    q_dep,q_cons,q_comm,q_inv,q_ins):
    revenue = -q_dep*r_dep + r_cons*q_cons + r_comm*q_comm + p_inv*q_inv + p_ins*q_ins
    return profit

## Equity Ratio Constraint
def equity_constraint(L,q_dep,q_cons,q_comm,e):
    val = (1-e)*(L + q_cons + q_comm) - q_dep
    return val

## Implied Marginal Costs
# First Order Conditions of a Bank
# Excludes the constraint, which contains only data and may not always hold

def implied_marginal_cost(
                            r_dep,r_cons,r_comm,p_inv,p_ins,
                            s_dep,s_cons,s_comm,s_inv,s_ins,
                            par):
    mc_assets = (r_return - r_dep)*(1-1/par.alpha_dep)
    mc_cons = r_cons - (1-par.e)*r_dep + ( 1/(par.alpha_cons*(1-s_cons)) - (1-par.e)/(par.alpha_dep*(1-s_dep)) )
    mc_comm = r_comm - (1-par.e)*r_dep + ( 1/(par.alpha_comm*(1-s_comm)) - (1-par.e)/(par.alpha_dep*(1-s_dep)) )
    mc_inv  = p_inv + 1/(par.alpha_inv*(1-s_inv))
    mc_ins  = p_ins + 1/(par.alpha_ins*(1-s_ins))

    # Zero out the MC estimates for products that aren't offered

    mc_cons = (s_cons>0)*mc_cons
    mc_comm = (s_comm>0)*mc_comm
    mc_inv = (s_inv>0)*mc_inv
    mc_ins = (s_ins>0)*mc_ins

    return mc_cons, mc_comm, mc_inv, mc_ins

def markups(
                r_dep,r_cons,r_comm,p_inv,p_ins,
                s_dep,s_cons,s_comm,s_inv,s_ins,
                par):
    mkup_dep = 1/(par.alpha_dep*(1-s_dep))
    mkup_cons = 1/(par.alpha_cons*(1-s_cons))
    mkup_comm = 1/(par.alpha_comm*(1-s_comm))
    mkup_inv  = 1/(par.alpha_inv*(1-s_inv))
    mkup_ins  =  1/(par.alpha_ins*(1-s_ins))

    # Zero out the MC estimates for products that aren't offered
    mkup_dep = (s_dep>0)*mkup_dep
    mkup_cons = (s_cons>0)*mkup_cons
    mkup_comm = (s_comm>0)*mkup_comm
    mkup_inv = (s_inv>0)*mkup_inv
    mkup_ins = (s_ins>0)*mkup_ins

    return mkup_dep, mkup_cons, mkup_comm, mkup_inv, mkup_ins



def gradient_marginal_cost(
                r_dep,r_cons,r_comm,p_inv,p_ins,
                s_dep,s_cons,s_comm,s_inv,s_ins,
                par):
    dmc_cons_d_cons = -1/(par.alpha_cons**2*(1-s_cons))
    dmc_cons_d_dep = (1-par.e)/(par.alpha_dep**2*(1-s_dep))

    dmc_comm_d_comm = -1/(par.alpha_comm**2*(1-s_comm))
    dmc_comm_d_dep = (1-par.e)/(par.alpha_dep**2*(1-s_dep))

    dmc_inv_d_inv = -1/(par.alpha_inv**2*(1-s_inv))
    dmc_ins_d_ins = -1/(par.alpha_ins**2*(1-s_ins))

    # Zero out the MC estimates for products that aren't offered
    dmc_cons_d_cons = (s_cons>0)*dmc_cons_d_cons
    dmc_cons_d_dep = (s_cons>0)*dmc_cons_d_dep

    dmc_comm_d_comm = (s_comm>0)*dmc_comm_d_comm
    dmc_comm_d_dep = (s_comm>0)*dmc_comm_d_dep

    dmc_inv_d_inv = (s_inv>0)*dmc_inv_d_inv
    dmc_ins_d_ins = (s_ins>0)*dmc_ins_d_ins

    return dmc_cons_d_cons, dmc_cons_d_dep, dmc_comm_d_comm, dmc_comm_d_dep, dmc_inv_d_inv, dmc_ins_d_ins


def hessian_marginal_cost(
                r_dep,r_cons,r_comm,p_inv,p_ins,
                s_dep,s_cons,s_comm,s_inv,s_ins,
                par):
    d2mc_cons_d_cons = 2/(par.alpha_cons**3*(1-s_cons))
    d2mc_cons_d_dep = - 2*(1-par.e)/(par.alpha_dep**3*(1-s_dep))

    d2mc_comm_d_comm = 2/(par.alpha_comm**3*(1-s_comm))
    d2mc_comm_d_dep = - 2*(1-par.e)/(par.alpha_dep**3*(1-s_dep))

    d2mc_inv_d_inv = 2/(par.alpha_inv**3*(1-s_inv))
    d2mc_ins_d_ins = 2/(par.alpha_ins**3*(1-s_ins))

    # Zero out the MC estimates for products that aren't offered
    d2mc_cons_d_cons = (s_cons>0)*d2mc_cons_d_cons
    d2mc_cons_d_dep = (s_cons>0)*d2mc_cons_d_dep

    d2mc_comm_d_comm = (s_comm>0)*d2mc_comm_d_comm
    d2mc_comm_d_dep = (s_comm>0)*d2mc_comm_d_dep

    d2mc_inv_d_inv = (s_inv>0)*d2mc_inv_d_inv
    d2mc_ins_d_ins = (s_ins>0)*d2mc_ins_d_ins

    return d2mc_cons_d_cons, d2mc_cons_d_dep, d2mc_comm_d_comm, d2mc_comm_d_dep, d2mc_inv_d_inv, d2mc_ins_d_ins

## Total Cost
## We assume that the cost function of each firm has the form:
# Cost = Aq^gamma
# In this format,
# mc = A*gamma*q^(gamma-1)
# avg cost = A*q^(gamma-1)
# total cost = (mc/gamma)*q
#  A = mc/(gamma*q^(gamma-1))
# total cost = Q^gamma*mc/(gamma*q^(gamma-1))
# total cost = q(Q/q)^gamma*mc/gamma
#
# and we only need to uncover a firm specific A.
# We can be more general than this,
# as long as each firms cost has only one product-specific parameter.
## We also assume that deposits have no marginal cost.
# This is important, since if it is false,
# Some component of each loan category's cost is shared.
# At this stage, easier to maybe to robustness checks after
# we get some sort of solution

def total_cost(L_cons,L_comm,
                r_dep,q_dep,
                q_cons,q_comm,q_inv,q_ins,
                mc_cons,mc_comm,mc_inv,mc_ins,
                par):

    cost = (L_cons)*mc_cons/par.gamma_cons + (L_comm)*mc_comm/par.gamma_comm + q_inv*mc_inv/par.gamma_inv + q_ins*mc_ins/par.gamma_ins + r_dep*q_dep
    return cost

def gradient_total_cost(L_cons,L_comm,
                r_dep,r_cons,r_comm,p_inv,p_ins,
                s_dep,s_cons,s_comm,s_inv,s_ins,
                q_cons,q_comm,q_inv,q_ins,
                mc_cons,mc_comm,mc_inv,mc_ins,
                par):
    dmc_cons_d_cons, dmc_cons_d_dep, dmc_comm_d_comm, dmc_comm_d_dep, dmc_inv_d_inv, dmc_ins_d_ins = gradient_marginal_cost(r_dep,r_cons,r_comm,p_inv,p_ins,s_dep,s_cons,s_comm,s_inv,s_ins,par)

    dcost_dalpha_dep = L_cons*dmc_cons_d_dep/par.gamma_cons + L_comm*dmc_comm_d_dep/par.gamma_comm
    dcost_dalpha_cons = L_cons*dmc_cons_d_cons/par.gamma_cons
    dcost_dalpha_comm = L_comm*dmc_comm_d_comm/par.gamma_comm
    dcost_dalpha_inv = q_inv*dmc_inv_d_inv/par.gamma_inv
    dcost_dalpha_ins = q_ins*dmc_ins_d_ins/par.gamma_ins

    dcost_dgamma_cons = -L_cons*mc_cons/par.gamma_cons**2
    dcost_dgamma_comm = -L_comm*mc_comm/par.gamma_comm**2
    dcost_dgamma_inv = -q_inv*mc_inv/par.gamma_inv**2
    dcost_dgamma_ins = -q_ins*mc_ins/par.gamma_ins**2

    # gradient = np.array([dcost_dalpha_dep,dcost_dalpha_cons,dcost_dalpha_comm,dcost_dalpha_inv,dcost_dalpha_ins,dcost_dgamma_cons,dcost_dgamma_comm,dcost_dgamma_inv,dcost_dgamma_ins])

    return dcost_dalpha_dep,dcost_dalpha_cons,dcost_dalpha_comm,dcost_dalpha_inv,dcost_dalpha_ins,dcost_dgamma_cons,dcost_dgamma_comm,dcost_dgamma_inv,dcost_dgamma_ins


def hessian_total_cost(L_cons,L_comm,
                r_dep,r_cons,r_comm,p_inv,p_ins,
                s_dep,s_cons,s_comm,s_inv,s_ins,
                q_cons,q_comm,q_inv,q_ins,
                mc_cons,mc_comm,mc_inv,mc_ins,
                par):
    dmc_cons_d_cons, dmc_cons_d_dep, dmc_comm_d_comm, dmc_comm_d_dep, dmc_inv_d_inv, dmc_ins_d_ins = gradient_marginal_cost(r_dep,r_cons,r_comm,p_inv,p_ins,s_dep,s_cons,s_comm,s_inv,s_ins,par)
    d2mc_cons_d_cons, d2mc_cons_d_dep, d2mc_comm_d_comm, d2mc_comm_d_dep, d2mc_inv_d_inv, d2mc_ins_d_ins = hessian_marginal_cost(r_dep,r_cons,r_comm,p_inv,p_ins,s_dep,s_cons,s_comm,s_inv,s_ins,par)


    d2cost_d2alpha_dep = L_cons*d2mc_cons_d_dep/par.gamma_cons + L_comm*d2mc_comm_d_dep/par.gamma_comm
    d2cost_dalpha_dep_dgamma_cons = -L_cons*dmc_cons_d_dep/par.gamma_cons**2
    d2cost_dalpha_dep_dgamma_comm = -L_comm*dmc_comm_d_dep/par.gamma_comm**2

    d2cost_d2alpha_cons = L_cons*d2mc_cons_d_cons/par.gamma_cons
    d2cost_dalpha_cons_dgamma_cons = -L_cons*dmc_cons_d_cons/par.gamma_cons**2

    d2cost_d2alpha_comm = L_comm*d2mc_comm_d_comm/par.gamma_comm
    d2cost_dalpha_comm_dgamma_comm = -L_comm*dmc_comm_d_comm/par.gamma_comm**2

    d2cost_d2alpha_inv = q_inv*d2mc_inv_d_inv/par.gamma_inv
    d2cost_dalpha_inv_dgamma_inv = -q_inv*dmc_inv_d_inv/par.gamma_inv**2

    d2cost_d2alpha_ins = q_ins*d2mc_ins_d_ins/par.gamma_ins
    d2cost_dalpha_ins_dgamma_ins = -q_ins*dmc_ins_d_ins/par.gamma_ins**2



    d2cost_d2gamma_cons = 2*L_cons*mc_cons/par.gamma_cons**3
    d2cost_d2gamma_comm = 2*L_comm*mc_comm/par.gamma_comm**3
    d2cost_d2gamma_inv = 2*q_inv*mc_inv/par.gamma_inv**3
    d2cost_d2gamma_ins = 2*q_ins*mc_ins/par.gamma_ins**3

    hessian = np.array([
    [d2cost_d2alpha_dep             ,0                  ,0                  ,0                  ,0                  ,d2cost_dalpha_dep_dgamma_cons  ,d2cost_dalpha_dep_dgamma_comm,0        ,0],
    [0                              ,d2cost_d2alpha_cons,0                  ,0                  ,0                  ,d2cost_dalpha_cons_dgamma_cons ,0                  ,0                  ,0],
    [0                              ,0                  ,d2cost_d2alpha_comm,0                  ,0                  ,0                              ,d2cost_dalpha_comm_dgamma_comm,0       ,0],
    [0                              ,0                  ,0                  ,d2cost_d2alpha_inv ,0                  ,0                              ,0                  ,d2cost_dalpha_inv_dgamma_inv,0],
    [0                              ,0                  ,0                  ,0                  ,d2cost_d2alpha_ins ,0                              ,0                  ,0                  ,d2cost_dalpha_ins_dgamma_ins],
    [d2cost_dalpha_dep_dgamma_cons  ,d2cost_dalpha_cons_dgamma_cons,0       ,0                  ,0                  ,d2cost_d2gamma_cons            ,0                  ,0                  ,0],
    [d2cost_dalpha_dep_dgamma_comm  ,0                  ,d2cost_dalpha_comm_dgamma_comm,0       ,0                  ,0                              ,d2cost_d2gamma_comm,0                  ,0],
    [0                              ,0                  ,0                  ,d2cost_dalpha_inv_dgamma_inv,0         ,0                              ,0                  ,d2cost_d2gamma_inv ,0],
    [0                              ,0                  ,0                  ,0                  ,d2cost_dalpha_ins_dgamma_ins,0                     ,0                  ,0                  ,d2cost_d2gamma_ins],

                    ])

    return hessian
