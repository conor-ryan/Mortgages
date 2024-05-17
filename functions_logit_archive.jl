
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

function π_hold(i::Int,r::Float64,ν::Float64,ϵ::Float64,θ::Parameters,p::Primitives)
    # return balance_sheet_penalty*r*(1-λ-risk_factor)/(ffr + risk_factor +λ) - (μ - ffr_funding_slope*ffr) + γ*λ/(ffr + risk_factor + λ)
    return r/(θ.m0_r + θ.m1_r*p.ffr) - p.λ[i] - (1  + ν + ϵ)
end

function dπdr_hold(r::Float64,θ::Parameters,p::Primitives)
    # return balance_sheet_penalty*r*(1-λ-risk_factor)/(ffr + risk_factor +λ) - (μ - ffr_funding_slope*ffr) + γ*λ/(ffr + risk_factor + λ)
    return 1/(θ.m0_r + θ.m1_r*p.ffr)
end

function π_sell(i::Int,r::Float64,ν::Float64,ϵ::Float64,θ::Parameters,p::Primitives)
    return P(r - 0.0025,p,θ) + 0.0025/(θ.m0_s + θ.m1_s*p.ffr)- p.λ[i] - (1+ ν)
end

function dπdr_sell(r::Float64,θ::Parameters,p::Primitives)
    return dPdr(r - 0.0025,p,θ)
end


function market_shares(i::Int,r::Vector{Float64},θ::Parameters)
    util = θ.δ .+ θ.α[i].*r
    eu = exp.(util)
    s = eu./(sum(eu))
    return s 
end

function market_shares(r::Vector{Vector{Float64}},θ::Parameters)
    shares = Vector{Vector{Float64}}(undef,length(p.λ))
    for i in eachindex(p.λ)         
        shares[i] = market_shares(i,r[i],θ)
    end
    return shares
end

function share_derivative(i::Int,r::Vector{Float64},θ::Parameters)
    util = θ.δ .+ θ.α[i].*r
    eu = exp.(util)
    s = eu./(sum(eu))
    dq = θ.α[i].*s.*(1 .-s)
    return dq
end

function elasticities(r::Vector{Float64},θ::Parameters)
    elas = Vector{Float64}(undef,length(p.λ))
    for i in eachindex(p.λ)
        q = market_shares(i,r,θ)
        dqdr =share_derivative(i,r,θ)
        elas[i] = sum(dqdr.*r)/sum(q)
       end
    return elas
end

function expected_profit(i::Int,r::Vector{Float64},
    θ::Parameters,p::Primitives)

    prof_hold = π_hold.(Ref(i),r,θ.ν,θ.ϵ,Ref(θ),Ref(p))
    prof_sell = π_sell.(Ref(i),r,θ.ν,θ.ϵ,Ref(θ),Ref(p))
    q = market_shares(i,r,θ)
    π_h = prof_hold.*q
    π_s = prof_sell.*q

    EΠ = θ.σ_bs*(log.(exp.(π_h./θ.σ_bs) + exp.(π_s./θ.σ_bs)) .- log(2))
 
    return EΠ
end

function balance_sheet_alloc(r::Vector{Vector{Float64}},
    θ::Parameters,p::Primitives)
    sell = Vector{Vector{Float64}}(undef,length(p.λ))
    for i in eachindex(p.λ)         
        prof_hold = π_hold.(Ref(i),r[i],θ.ν,θ.ϵ,Ref(θ),Ref(p))
        prof_sell = π_sell.(Ref(i),r[i],θ.ν,θ.ϵ,Ref(θ),Ref(p))
        q = market_shares(i,r[i],θ)
        π_h = prof_hold.*q
        π_s = prof_sell.*q

        hold = exp.(π_h./θ.σ_bs)./(exp.(π_h./θ.σ_bs) + exp.(π_s./θ.σ_bs))
        sell[i] = 1 .- hold
    end
    return sell
end

function expected_foc(i::Int,r::Vector{Float64},
    θ::Parameters,p::Primitives)

    π_h = π_hold.(Ref(i),r,θ.ν,θ.ϵ,Ref(θ),Ref(p))
    π_s = π_sell.(Ref(i),r,θ.ν,θ.ϵ,Ref(θ),Ref(p))

    dπ_h = dπdr_hold.(r,Ref(θ),Ref(p))
    dπ_s = dπdr_sell.(r,Ref(θ),Ref(p))


    q = market_shares(i,r,θ)
    dqdp = share_derivative(i,r,θ)

    eπ_h = exp.(π_h.*q/θ.σ_bs)
    eπ_s = exp.(π_s.*q/θ.σ_bs)

    dEΠdr = (dqdp.*(π_h.*eπ_h .+ π_s.*eπ_s) .+ q.*(dπ_h.*eπ_h .+ dπ_s.*eπ_s))./(eπ_h .+ eπ_s)
 
    return dEΠdr
end
