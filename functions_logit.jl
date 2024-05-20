
function P(r::Float64,m::MBS)
    mbs_curve = 2
    par_rate = m.par_rate
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

function dPdr(r::Float64,m::MBS)
    mbs_curve = 2
    par_rate = m.par_rate
    return (1/par_rate)*(1 - 2*mbs_curve*(r-par_rate))
end   

function P_inv(Price::Float64,m::MBS)
    mbs_curve = 2
    par_rate = m.par_rate
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

function π_hold(r::Vector{Float64},θ::Parameters,d::Data)
    discount_rate = d.D'*θ.β_d
    bank_cost = d.W*θ.γ_WH
    credit_cost = d.Z'*θ.γ_ZH
    return r./(discount_rate) .- credit_cost .- bank_cost
end

function π_hold_idtest(r::Vector{Float64},θ::Parameters,d::Data)
    discount_rate = d.D'*θ.β_d
    bank_cost = d.W*θ.γ_WH
    credit_cost = d.Z'*θ.γ_ZH
    return r./(discount_rate)  .- credit_cost .- bank_cost
    # return r./(discount_rate) .-bank_cost
end


function r_hold_min(θ::Parameters,d::Data)
    discount_rate = d.D'*θ.β_d
    bank_cost = d.W*θ.γ_WH
    credit_cost = d.Z'*θ.γ_ZH
    return (credit_cost .+ bank_cost).*discount_rate
end

function dπdr_hold(r::Vector{Float64},θ::Parameters,d::Data)
    discount_rate = d.D'*θ.β_d
    return 1 ./(discount_rate)
end

function π_sell(r::Vector{Float64},θ::Parameters,d::Data,m::MBS)
    discount_rate = d.D'*θ.β_d
    bank_cost = d.W*(θ.γ_WS.+θ.γ_WH)
    credit_cost = d.Z'*(θ.γ_ZS.+θ.γ_ZH)
    return P.(r .- 0.0025,Ref(m)) .- credit_cost .-bank_cost
end

function r_sell_min(θ::Parameters,d::Data,m::MBS)
    bank_cost = d.W*(θ.γ_WS.+θ.γ_WH)
    credit_cost = d.Z'*(θ.γ_ZS.+θ.γ_ZH)
    return P_inv.(credit_cost .+ bank_cost,Ref(m)) .+ 0.0025
end

function π_sell_idtest(r::Vector{Float64},θ::Parameters,d::Data,m::MBS)
    discount_rate = d.D'*θ.β_d
    bank_cost = d.W*(θ.γ_WS.+θ.γ_WH)
    credit_cost = d.Z'*(θ.γ_ZS.+θ.γ_ZH)
    return P.(r .- 0.0025,Ref(m)) .- credit_cost .-bank_cost
    # return P.(r .- 0.0025,Ref(m)) -bank_cost
end

function dπdr_sell(r::Vector{Float64},m::MBS)
    return dPdr.(r .- 0.0025,Ref(m))
end


function market_shares(r::Vector{Float64},α::Float64,d::Data,θ::Parameters)
    util = d.X*θ.β_x .+ α.*r
    eu = exp.(util)
    s = eu./(1 + sum(eu))
    return s 
end

function share_derivative(r::Vector{Float64},α::Float64,d::Data,θ::Parameters)
    util = d.X*θ.β_x .+ α.*r
    eu = exp.(util)
    s = eu./(1 + sum(eu))
    dq = α.*s.*(1 .-s)
    return s, dq
end

function expected_profit(r::Vector{Float64},α::Float64,d::Data,θ::Parameters,m::MBS)

    prof_hold = π_hold(r,θ,d)
    prof_sell = π_sell(r,θ,d,m)
    q = market_shares(r,α,d,θ)
    π_h = prof_hold.*q
    π_s = prof_sell.*q

    EΠ = θ.σ*(log.(exp.(π_h./θ.σ) + exp.(π_s./θ.σ)) .- log(2))
 
    return EΠ
end

function expected_sale_profit(r::Vector{Float64},d::Data,θ::Parameters,m::MBS)

    prof_hold = π_hold_idtest(r,θ,d)
    prof_sell = π_sell_idtest(r,θ,d,m)
    π_h = prof_hold
    π_s = prof_sell

    sell = exp.(π_s./θ.σ)./(exp.(π_h./θ.σ) + exp.(π_s./θ.σ))


    EΠ = π_h.*(1 .-sell) .+ π_s.*sell
 
    return EΠ
end

function balance_sheet_alloc(r::Vector{Float64},α::Float64,d::Data,θ::Parameters,m::MBS)
    # sell = Vector{Vector{Float64}}(undef,length(p.λ))  
    prof_hold = π_hold_idtest(r,θ,d)
    prof_sell = π_sell_idtest(r,θ,d,m)
    q = market_shares(r,α,d,θ)
    π_h = prof_hold
    π_s = prof_sell

    sell = exp.(π_s./θ.σ)./(exp.(π_h./θ.σ) + exp.(π_s./θ.σ))
 
    return sell
end

function expected_foc(r::Vector{Float64},α::Float64,d::Data,θ::Parameters,m::MBS)

    π_h = π_hold(r,θ,d)
    π_s = π_sell(r,θ,d,m)

    dπ_h = dπdr_hold(r,θ,d)
    dπ_s = dπdr_sell(r,m)

    q,dqdp = share_derivative(r,α,d,θ)

    eπ_h = exp.(π_h/θ.σ)
    eπ_s = exp.(π_s/θ.σ)

    dEΠdr = (dqdp.*(π_h.*eπ_h .+ π_s.*eπ_s) .+ q.*(dπ_h.*eπ_h .+ dπ_s.*eπ_s))./(eπ_h .+ eπ_s)
 
    return dEΠdr
end

function hold_only_foc(r::Vector{Float64},α::Float64,d::Data,θ::Parameters)

    π_h = π_hold(r,θ,d)
    # π_s = π_sell(r,θ,d,m)

    dπ_h = dπdr_hold(r,θ,d)
    # dπ_s = dπdr_sell(r,m)

    q,dqdp = share_derivative(r,α,d,θ)

    dΠdr = dqdp.*π_h .+ q.*dπ_h
 
    return dΠdr
end

function min_rate(d::Data,θ::Parameters,m::MBS)
    # Minimum Interest Rates, by lending type
    r_h_min = r_hold_min(θ,d) # HTM lowest rate
    r_s_min = r_sell_min(θ,d,m) # OTD lowest rate
    
    # Minimum possible rate
    lowest_rate = min.(r_h_min,r_s_min)
    # Initialize a guess for a rate that is profitable
    highest_rate = lowest_rate*3
    # Compute profit at each rate guess
    low_profit = expected_sale_profit(lowest_rate,d,θ,m)
    high_profit = expected_sale_profit(highest_rate,d,θ,m)

    # First step of the secant method
    mc_n_1 = lowest_rate .- low_profit.* (lowest_rate .- highest_rate)./(low_profit .- high_profit)
    prof_n_1 = expected_sale_profit(mc_n_1,d,θ,m)
    # Initialize error
    err = sum(prof_n_1.^2)
    # Save terms for next step of secant method
    mc = mc_n_1
    mc_n_2 = lowest_rate
    prof_n_2 = low_profit
    # Iterate until error is less than tolerance 
    while err>1e-12
        # Secant method step
        mc = mc_n_1 .- prof_n_1.*(mc_n_1.-mc_n_2)./(prof_n_1 .- prof_n_2)
        prof = expected_sale_profit(mc,d,θ,m)
        # Recompute error
        err = sum(prof.^2)

        # Save for next step
        mc_n_2 = copy(mc_n_1)
        mc_n_1 = copy(mc)
        prof_n_2 = copy(prof_n_1)
        prof_n_1 = copy(prof)
    end
    return mc
end