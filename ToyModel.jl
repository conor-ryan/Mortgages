using Plots

#### Functions ####
mbs_curve = 10
function P(r,ffr)
    par_rate = (ffr+m_r)/(1-m_r)
    return (1/par_rate)*(r - mbs_curve*(r-par_rate)^2)
end

function P_inv(P,ffr) 
    par_rate = (ffr+m_r)/(1-m_r)
    c = -mbs_curve*par_rate^2/par_rate - P
    b = (1+ mbs_curve*2*par_rate)/par_rate
    a = -mbs_curve/par_rate
    r = (- b + sqrt(b^2 - 4*a*c))/(2*a)
    return r
    # return c + b*r + a*r^2
end

function π_hold(r,λ,ffr)
    # return balance_sheet_penalty*r*(1-λ-risk_factor)/(ffr + risk_factor +λ) - (μ - ffr_funding_slope*ffr) + γ*λ/(ffr + risk_factor + λ)
    return balance_sheet_penalty*r*(1-m_r)/(ffr+m_r) - λ - (1 + μ - ffr_funding_slope*ffr)
end


function π_sell(r,λ,ffr)
    return P(r - 0.0025,ffr) + 0.0025*(1-m_s)/(ffr+m_s)- λ - (1 + μ + ϵ)
end



function r_hold_min(λ,ffr)
    return ((1 + μ - ffr_funding_slope*ffr) + λ)*(ffr+m_r)/(1-m_r)/balance_sheet_penalty
end


function r_sell_min(λ,ffr)
    return P_inv((1 + μ + ϵ)+λ-0.0025*(1-m_s)/(ffr+m_s),ffr) + 0.0025 
end


#### Constants 

# γ = 0.1
# α = 1
# ffr_funding_slope = 8.0

# risk_factor = 0.02
balance_sheet_penalty = 0.73
# μ_s = 0.00
# μ_h = 0.9
# i =  0.0075

μ = -0.25
ϵ = 0.0
m_r = 0.02
m_s = 0.02
ffr_funding_slope = 6

#### Experiments

λ = 0.01
ffr = Float64.(range(0.005,0.06,length=20))

plot(ffr,[r_hold_min.(λ,ffr),r_sell_min.(λ,ffr)],label=["Hold" "Sell"])



# λ = Float64.(range(0.000,0.08,length=20))
# ffr = 0.02
# plot(λ,[r_hold_min.(λ,ffr),r_sell_min.(λ,ffr)],label=["Hold" "Sell"])


ffr = 0.04
λ = 0.01
r = Float64.(range(0.00,0.15,length=20))
plot(r,[ π_hold.(r,λ,ffr),π_sell.(r,λ,ffr)],label=["Hold" "Sell"])




(1/β)*(r_l*p - r_d)
(p/β)*r_l - (1/β)*r_d


r = (i/ffr)*β

plot(ffr,((i.+0.15.*ffr)./(ffr.+i)).*(0.02 .+ffr.*1.2) )

plot(ffr,0.02 .+ 1.2 .*ffr)








r = range(0.0,0.07,length=20)

price = P.(r,ffr)
price_2 = P_inv.(r,ffr)

plot(r,price)
plot(r,price_2)
plot(r,P_inv.(price,ffr) )





u = -.05

r_star = -u


π_h = r_star - (γ*Z + β_h*X + ϵ + μ_h)
π_s = P(r_star - 0.0025 - g) + 0.25 - (γ_s*Z + β_s*X + ϵ + μ_s)


par_rate = 0.05
r_star = 0.055
λ = 0
π_h = r_star/(par_rate+λ) - i/ffr - (μ) + γ*λ/(ffr + λ)


π_s = P(r_star - 0.0025 - g) + 0.25 - (1 + μ_s)




