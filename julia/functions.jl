
function P(r::Float64,p::Primitives,θ::Parameters)
    mbs_curve = 0
    # par_rate = (p.ffr+p.m_mbs)/(1-p.m_mbs)
    par_rate = θ.m0_mbs + θ.m1_mbs*p.ffr
    if mbs_curve>0
        price =  (1/par_rate)*(r - mbs_curve*(r-par_rate)^2)
    else
        if r>=par_rate
            price = r/par_rate
        else
            price = r/par_rate
        end
    end
end

function dPdr(r::Float64,p::Primitives,θ::Parameters)
    mbs_curve = 0
    # par_rate = (p.ffr+p.m_mbs)/(1-p.m_mbs)
    par_rate = θ.m0_mbs + θ.m1_mbs*p.ffr
    return (1/par_rate)*(1 - 2*mbs_curve*(r-par_rate))
end   

function P_inv(Price::Float64,p::Primitives,θ::Parameters) 
    mbs_curve = 0
    # par_rate = (p.ffr+p.m_mbs)/(1-p.m_mbs)
    par_rate = θ.m0_mbs + θ.m1_mbs*p.ffr
    if mbs_curve>0
        c = -mbs_curve*par_rate^2/par_rate - Price
        b = (1+ mbs_curve*2*par_rate)/par_rate
        a = -mbs_curve/par_rate

        if (b^2 - 4*a*c)<0
            return 0.55
        else 
            r = (- b + sqrt(b^2 - 4*a*c))/(2*a)
            return r
        end
        # return c + b*r + a*r^2
    else
        if Price>=1
            r = Price*par_rate
        else
            r = Price*par_rate
        end
        return r
    end
end

function π_hold(r::Float64,ν::Float64,ϵ::Float64,θ::Parameters,p::Primitives)
    # return balance_sheet_penalty*r*(1-λ-risk_factor)/(ffr + risk_factor +λ) - (μ - ffr_funding_slope*ffr) + γ*λ/(ffr + risk_factor + λ)
    return r/(θ.m0_r + θ.m1_r*p.ffr) - p.λ - (1  + ν + ϵ - θ.ffr_funding_slope*p.ffr)
end

function dπdr_hold(r::Float64,p::Primitives)
    # return balance_sheet_penalty*r*(1-λ-risk_factor)/(ffr + risk_factor +λ) - (μ - ffr_funding_slope*ffr) + γ*λ/(ffr + risk_factor + λ)
    return 1/(θ.m0_r + θ.m1_r*p.ffr)
end

function π_sell(r::Float64,ν::Float64,ϵ::Float64,θ::Parameters,p::Primitives)
    return P(r - 0.0025,p,θ) + 0.0025/(θ.m0_s + θ.m1_s*p.ffr)- p.λ - (1+ ν)
end

function dπdr_sell(r::Float64,p::Primitives)
    return dPdr(r - 0.0025,p,θ)
end

function r_hold_min(ν::Float64,ϵ::Float64,θ::Parameters,p::Primitives)
    return ((1 + ν + ϵ - θ.ffr_funding_slope*p.ffr) + p.λ)*(θ.m0_r + θ.m1_r*p.ffr)
end

function r_sell_min(ν::Float64,ϵ::Float64,θ::Parameters,p::Primitives)
    return P_inv((1 + ν)+p.λ-0.0025/(θ.m0_s + θ.m1_s*p.ffr),p,θ) + 0.0025 
end


function monopoly_expected_profit_hold(r::Float64,ν::Float64,ϵ::Float64,
                                        probs::Vector{Float64},profit::Vector{Float64},V2::Float64,
                                        θ::Parameters,p::Primitives)
    r = first(r)
    prob_mon, exp_prof = search_after_price(r,probs,profit,V2,θ)
    prof = π_hold(r,ν,ϵ,θ,p)*(prob_mon) + (1-prob_mon)*exp_prof
    return prof
end


function monopoly_expected_profit_sell(r::Float64,ν::Float64,ϵ::Float64,
                                        probs::Vector{Float64},profit::Vector{Float64},V2::Float64,
                                        θ::Parameters,p::Primitives)
    r = first(r)

    prob_mon, exp_prof = search_after_price(r,probs,profit,V2,θ)
    prof = π_sell(r,ν,ϵ,θ,p)*(prob_mon) + (1-prob_mon)*exp_prof
    return prof
end

function monopoly_expected_profit(r::Float64,ν::Float64,ϵ::Float64,
    probs::Vector{Float64},profit::Vector{Float64},V2::Float64,
    θ::Parameters,p::Primitives)

    prof_hold = monopoly_expected_profit_hold(r_hold_min,ν,ϵ,probs,profit,V2,θ,p)
    prof_sell = monopoly_expected_profit_hold(r,ν,ϵ,probs,profit,V2,θ,p)

    EΠ = θ.σ_ϵ*log(exp(prof_hold/θ.σ_ϵ) + exp(prof_sell/θ.σ_ϵ)) - log(2)
 
    return EΠ
end

function model_outcomes(θ::Parameters,p::Primitives;allow_otd=true)

    starting_point = [p.ffr + p.λ + 0.04]
    I = length(θ.mon_ϵ)
    price = Matrix{Float64}(undef,I,5)
    price_h = Matrix{Float64}(undef,I,5)
    price_s = Matrix{Float64}(undef,I,5)
    sold = Matrix{Float64}(undef,I,5)
    search = Matrix{Float64}(undef,I,5)

    
    for i in 1:I
        ret, p_h, h = monopoly_price(starting_point,monopoly_expected_profit_hold,θ.mon_ν[i],θ.mon_ϵ[i],θ,p,allow_otd=allow_otd)
        ret, p_s, s = monopoly_price(starting_point,monopoly_expected_profit_sell,θ.mon_ν[i],θ.mon_ϵ[i],θ,p,allow_otd=allow_otd)
        if allow_otd
            # ret, pi, r = monopoly_price(starting_point,monopoly_expected_profit,θ.mon_ν[i],θ.mon_ϵ[i],θ,p,allow_otd=allow_otd)
            # pi_h = π_hold(r,θ.mon_ν[i],θ.mon_ϵ[i],θ,p)
            # pi_s = π_sell(r,θ.mon_ν[i],θ.mon_ϵ[i],θ,p)
            # mon_sell = exp(pi_s/θ.σ_ϵ)/(exp(pi_s/θ.σ_ϵ) + exp(pi_h/θ.σ_ϵ))

            mon_sell = p_s>(p_h + 1e-12)         
            r = h + (mon_sell)*(s-h)
        else
            # ret, pi, r = monopoly_price(starting_point,monopoly_expected_profit_hold,θ.mon_ν[i],θ.mon_ϵ[i],θ,p,allow_otd=allow_otd)
            mon_sell = 0.0
            r = h
        end

        probs, profit, util, frac_sold, r_h, r_s = search_expectations(r,θ.mon_ν[i],θ.mon_ϵ[i],θ,p,allow_otd=allow_otd)

        price[i,1] = r
        price[i,2:5] = -util

        price_h[i,1] = h
        price_h[i,2:5] = r_h

        price_s[i,1] = s
        price_s[i,2:5] = r_s
        
        sold[i,1] = mon_sell
        sold[i,2:5] = frac_sold

        search[i,:] = probs

        price_h[isnan.(price_h)].=price_s[isnan.(price_h)]
        price_s[isnan.(price_s)].=price_h[isnan.(price_s)]

    end
    
    return (price=price, price_h = price_h, price_s = price_s, sold=sold, search=search)
end




function monopoly_price(x::Vector{Float64},func::Function,ν::Float64,ϵ::Float64,θ::Parameters,p::Primitives;
                        method=:LN_NELDERMEAD,allow_otd=true)
    # Set up the optimization
    # opt = Opt(method,length(x))
    # ftol_rel!(opt,1e-8)
    
    probs, profit, util, frac_sold, r_h, r_s = search_expectations(Inf,ν,ϵ,θ,p,allow_otd=allow_otd)
    eval(x) = -func(x,ν,ϵ,probs,profit,util[2],θ,p)

    # lb =  repeat([-0.5],inner=length(x))
    # ub = repeat([0.5],inner=length(x))

    # lower_bounds!(opt, lb)
    # upper_bounds!(opt, ub)
    function eval(x,grad)
        obj = eval(x)
        return obj
    end
    
    # max_objective!(opt, eval)
    # Run Optimization
    # minf, minx, ret = optimize(opt, x)
    # results = optimize(eval,x)
    # results = optimize(eval,x,BFGS())
    results = optimize(eval,0.0,0.3,Brent())
    profit = - results.minimum
    return Optim.converged(results), profit, first(results.minimizer)
end



function search_after_price(r::Float64,probs::Vector{Float64},profit::Vector{Float64},V2::Float64,
                                θ::Parameters)
    prob_mon = 1-cdf(θ.η_dist,V2+r)
    new_probs = similar(probs)
    new_probs[:] = probs[:]
    c1 =cumsum(probs)
    c1[5] = 1.0
    crit = findall((c1.<prob_mon))
    if length(crit)>0
        crit = maximum(crit)+1
    else
        crit = 2
    end
    new_probs[2:crit].=0.0
    new_probs[crit] = 1-prob_mon
    new_probs[1] = prob_mon
    exp_prof = sum(new_probs[2:5].*profit)
    return prob_mon, exp_prof
end


function search_expectations(r0::Float64,ν::Float64,ϵ::Float64,
                            θ::Parameters,p::Primitives;allow_otd=true)
    # Expected Value from Search
    prof_2, V_2, sold_2, r2_h, r2_s =  auction_average(1,ν,ϵ,θ,p,allow_otd = allow_otd)
    prof_3, V_3, sold_3, r3_h, r3_s =  auction_average(2,ν,ϵ,θ,p,allow_otd = allow_otd)
    prof_4, V_4, sold_4, r4_h, r4_s =  auction_average(3,ν,ϵ,θ,p,allow_otd = allow_otd)
    prof_5, V_5, sold_5, r5_h, r5_s =  auction_average(4,ν,ϵ,θ,p,allow_otd = allow_otd)

    p_1 = 1-cdf(θ.η_dist,V_2+r0)
    p_2 = max((1-p_1) - cdf(θ.η_dist,V_3-V_2),0)
    p_3 = max((1-(p_1+p_2)) - cdf(θ.η_dist,V_4-V_3),0)
    p_4 = max((1-(p_1+p_2+p_3)) - cdf(θ.η_dist,V_5-V_4),0)
    p_5 = (1-(p_1+p_2+p_3+p_4)) 

    if p_1<1
        expected_profit = (prof_2*p_2 + prof_3*p_3 + prof_4*p_4 + prof_5*p_5)/(1-p_1)
    else
        expected_profit = 0
    end
    return [p_1,p_2,p_3,p_4,p_5], [prof_2,prof_3,prof_4,prof_5], [V_2, V_3, V_4, V_5], 
                [sold_2,sold_3,sold_4,sold_5], 
                [r2_h,r3_h,r4_h,r5_h],
                [r2_s,r3_s,r4_s,r5_s]
end



function auction(mon_bid::Float64,bids_N::Matrix{Float64},N::Int)
    K = size(bids_N,1)
    mon_win = Vector{Float64}(undef,K)
    auction_price = Vector{Float64}(undef,K)
    for i in 1:K
        bids = sort(vcat(mon_bid,bids_N[i,1:N]))
        lowest_competitor = bids[1]
        auction_price[i] = bids[2]
        mon_win[i] = lowest_competitor==mon_bid
    end 
    return mon_win,auction_price
end

function competitive_bids(ν_draws::Matrix{Float64},ϵ_draws::Matrix{Float64},θ::Parameters,p::Primitives)
    hold_bids = r_hold_min.(ν_draws,ϵ_draws,Ref(θ),Ref(p))
    sell_bids = r_sell_min.(ν_draws,ϵ_draws,Ref(θ),Ref(p))
    bids_N = best_bid.(hold_bids,sell_bids)
    return bids_N
end

function auction_average(N::Int,ν::Float64,ϵ::Float64,
                            θ::Parameters,p::Primitives;allow_otd=true)
    #Monopoly Bid
    mon_hold_bid = r_hold_min(ν,ϵ,θ,p)
    mon_sell_bid = r_sell_min(ν,ϵ,θ,p)
    if allow_otd
        bid = min(mon_hold_bid,mon_sell_bid)
    else 
        bid = mon_hold_bid
    end
    
    #Auction
    if allow_otd
        win, price = auction(bid,θ.bids_N,N)
    else
        win, price = auction(bid,θ.bids_N_noOTD,N)
    end
    pi_h = π_hold.(price,ν,ϵ,Ref(θ),Ref(p))
    pi_s = π_sell.(price,ν,ϵ,Ref(θ),Ref(p))
    if allow_otd
        sold = pi_s.>(pi_h .+ 1e-12)
        # sold = exp.(pi_s/θ.σ_ϵ)./(exp.(pi_s/θ.σ_ϵ) .+ exp.(pi_h/θ.σ_ϵ))
    else
        sold = 0.0
    end
    prof = pi_h.*(1 .-sold) .+ sold.*pi_s #max.(prof_hold,prof_sell)
    avg_profit = mean(win.*prof)
    utility = -mean(price)
    frac_sold = mean(sold)
    r_h = sum(price.*(1 .-sold))/(sum( 1 .-sold))
    r_s = sum(price.*(sold))/(sum(sold))
    return avg_profit,utility,frac_sold, r_h, r_s
end


function best_bid(hold_bid::Float64,sell_bid::Float64)
    return min(hold_bid,sell_bid)
end

function best_profit(hold_prof::Float64,sell_prof::Float64)
    return max(hold_prof,sell_prof)
end

