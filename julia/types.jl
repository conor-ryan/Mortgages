struct Primitives
    #Temporary Data
    λ::Float64 
    ffr::Float64

    # Competitor Draws
    comp_ϵ_draw::Matrix{Float64}
    comp_ν_draw::Matrix{Float64}

    # Monopolist Draws
    mon_ν_draw::Vector{Float64}
    mon_ϵ_draw::Vector{Float64}
end

function Primitives()
    O_mon = 1000
    O_comp = 2000

    #Temporary Data
    λ = 0.01
    ffr = 0.04


    comp_ϵ= randn((O_comp,4))
    comp_ν  = randn((O_comp,4))


    mon_ν = randn(O_mon)
    mon_ϵ = randn(O_mon)

    
    return Primitives(λ,ffr,comp_ϵ,comp_ν,mon_ν,mon_ϵ)

end

function Primitives(M::Int64,C::Int64)
    O_mon = M
    O_comp = C

    #Temporary Data
    λ = 0.01
    ffr = 0.04


    comp_ϵ= randn((O_comp,4))
    comp_ν  = randn((O_comp,4))


    mon_ν = randn(O_mon)
    mon_ϵ = randn(O_mon)

    
    return Primitives(λ,ffr,comp_ϵ,comp_ν,mon_ν,mon_ϵ)

end


struct Draws
    comp_ϵ::Matrix{Float64}
    comp_ν::Matrix{Float64}


    mon_ν::Vector{Float64}
    mon_ϵ::Vector{Float64}
end


function Primitives(f::Float64,draws::Draws)
    #Temporary Data
    λ = 0.01
    ffr = f
    
    return Primitives(λ,ffr,draws.comp_ϵ,draws.comp_ν,draws.mon_ν,draws.mon_ϵ)

end


function Draws(M::Int64,C::Int64)
    O_mon = M
    O_comp = C
    X = MVHaltonNormal(O_comp,8)
    comp_ϵ= X[:,1:4]
    comp_ν  = X[:,5:8]

    X = MVHaltonNormal(O_mon,2)
    mon_ν = X[:,1]
    mon_ϵ = X[:,2]

    
    return Draws(comp_ϵ,comp_ν,mon_ν,mon_ϵ)
end

struct Parameters
    # FFR Passthrough to funding costs
    ffr_funding_slope::Float64
    # Balance sheet penalty from HTM
    balance_sheet_penalty::Float64
    # HTM Discount factor
    m0_r::Float64
    m1_r::Float64
    # Servicing rights Discount factor
    m0_s::Float64
    m1_s::Float64
    # MBS Discount factor
    m0_mbs::Float64
    m1_mbs::Float64

    # Effect of ffr on funding cost
    # ffr_funding_slope::Float64

    # Distribution of OTD costs
    μ_ϵ::Float64
    σ_ϵ::Float64
    #Distribution of Lending costs
    μ_ν::Float64
    σ_ν::Float64
    #Distribution of Search costs
    μ_η::Float64
    σ_η::Float64

    # Computed Distributions
    η_dist::LogNormal{Float64}
    ϵ_dist::Normal{Float64}

    # Competitor Draws
    comp_ϵ::Matrix{Float64}
    comp_ν::Matrix{Float64}
    bids_N::Matrix{Float64}
    bids_N_noOTD::Matrix{Float64}

    # Monopolist Draws
    mon_ν::Vector{Float64}
    mon_ϵ::Vector{Float64}

end

function Parameters(x::Vector{Float64},p::Primitives)
        # FFR Passthrough to funding costs
        ffr_funding_slope= x[7]
        # Balance sheet penalty from HTM
        balance_sheet_penalty = x[8]
        # HTM Discount factor
        m0_r = x[9]
        m1_r = x[10]
        # Servicing rights Discount factor
        m0_s= m0_r
        m1_s= m1_r
        # MBS Discount factor
        m0_mbs= x[11]
        m1_mbs= x[12]
        # Effect of ffr on funding cost
        # ffr_funding_slope::Float64
    
        # Distribution of OTD costs
        μ_ϵ = x[1]
        σ_ϵ = x[2]
        #Distribution of Lending costs
        μ_ν = x[3]
        σ_ν = x[4]
        #Distribution of Search costs
        μ_η = x[5]
        σ_η = x[6]

        
    # Distributions
    η_dist = LogNormal(μ_η,σ_η)
    ϵ_dist = Normal(μ_ϵ,σ_ϵ)

    comp_ϵ = p.comp_ϵ_draw.*σ_ϵ .+ μ_ϵ
    comp_ν = p.comp_ν_draw.*σ_ν .+ μ_ν
    bids_N = Matrix{Float64}(undef,0,0)


    mon_ν = p.mon_ν_draw.*σ_ν .+ μ_ν
    mon_ϵ = p.mon_ϵ_draw.*σ_ϵ .+ μ_ϵ
    θ = Parameters(ffr_funding_slope,balance_sheet_penalty,
                    m0_r,m1_r,m0_s,m1_s,m0_mbs,m1_mbs,
                    μ_ϵ,σ_ϵ,μ_ν,σ_ν,μ_η,σ_η,
                    η_dist,ϵ_dist,
                    comp_ϵ,comp_ν,bids_N,bids_N,mon_ν,mon_ϵ)

    # Compute the competitive bids
    bids_N = competitive_bids(comp_ν,comp_ϵ,θ,p)
    bids_N_noOTD = r_hold_min.(comp_ν,comp_ϵ,Ref(θ),Ref(p))
    return Parameters(ffr_funding_slope,balance_sheet_penalty,
                        m0_r,m1_r,m0_s,m1_s,m0_mbs,m1_mbs,
                        μ_ϵ,σ_ϵ,μ_ν,σ_ν,μ_η,σ_η,
                        η_dist,ϵ_dist,
                        comp_ϵ,comp_ν,bids_N,bids_N_noOTD,mon_ν,mon_ϵ)
end


