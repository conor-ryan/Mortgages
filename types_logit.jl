struct Data
    X::Matrix{Float64} # Demand Data 
    W::Matrix{Float64} # Cost Data
    D::Vector{Float64} # Discount Rate 
    Z::Vector{Float64} # Consumer Credit
end

struct MBS
    par_rate::Float64
end


struct Parameters
    # HTM Discount factor
    β_d::Vector{Float64}

    ## Logit Demand Parameters
    β_x::Vector{Float64}

    #Balance-sheet cost
    γ_WH::Vector{Float64}

    #Origination cost difference
    γ_WS::Vector{Float64}

    #Balance-sheet consumer costs
    γ_ZH::Vector{Float64}

    #Origination consumer cost difference
    γ_ZS::Vector{Float64}

    ## Balance Sheet Shock
    σ::Float64

end


# function Parameters(x::Vector{Float64},J)

#     δ = x[1:J]
    
#     ω = x[(J+1):(J+J)]
#     ϵ0 = x[2*J+1]
#     ϵ1 = x[2*J+2]
#     λ = x[2*J+3]
#     σ = x[2*J+4]

#     # balance sheet Cost Parameters
#     γ_bs = vcat(ϵ0,ϵ1,ω)

#     # OTD Cost Parameters
#     γ_otd = vcat(0.0,0.0,ω)

#     # HTM Discount factor
#     m0_r = x[2*J+5]
#     m1_r = x[2*J+6]
#     β = [m0_r,m1_r]

#     θ = Parameters(β,δ,γ_bs,γ_otd,λ,σ)
# end
    
    

    
# function Parameters(x::Vector{Float64},J)

#     # HTM Discount factor
#     m0_r = x[8]
#     m1_r = x[9]
#     # Servicing rights Discount factor
#     m0_s= m0_r
#     m1_s= m1_r
#     # MBS Discount factor
#     m0_mbs= x[10]
#     m1_mbs= x[11]
#     # Effect of ffr on funding cost
#     # ffr_funding_slope::Float64
#     δ = x[1:4]
#     α_i = x[5] .+ rand(length(p.λ)).*x[6] .+ p.λ.*x[7]

#     # OTD costs
#     ϵ = x[16:19]
#     # Lending costs
#     ν = x[12:15]
#     #Distribution of Search costs
#     μ_η = -4
#     σ_η = 1

#     σ_bs = x[20]
        
#     # Distributions
#     η_dist = LogNormal(μ_η,σ_η)
#     ϵ_dist = Normal(0.0,1.0)

#     comp_ϵ = Matrix{Float64}(undef,0,0)
#     comp_ν =  Matrix{Float64}(undef,0,0)
#     bids_N = Matrix{Float64}(undef,0,0)


#     mon_ν =  Vector{Float64}(undef,0)
#     mon_ϵ =  Vector{Float64}(undef,0)
#     θ = Parameters(m0_r,m1_r,m0_s,m1_s,m0_mbs,m1_mbs,
#                     δ,α_i,
#                     ϵ,ν,σ_bs,
#                     μ_η,σ_η,
#                     η_dist,ϵ_dist,
#                     comp_ϵ,comp_ν,bids_N,bids_N,mon_ν,mon_ϵ)

# end


