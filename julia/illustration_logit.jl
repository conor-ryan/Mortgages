# function plot_profit(r_vec::Vector{Float64},r_eq::Vector{Float64},test_firm::Int,
#     θ::Parameters,p::Primitives)

#     prof_vec = Vector{Float64}(undef,length(r_vec))
#     for i in eachindex(r_vec)
#         r_eq[test_firm] = r_vec[i]
#         Π = expected_profit(r_eq,θ,p)
#         prof_vec[i] = Π[test_firm]
#     end
#     return prof_vec
# end

# function orig_spread(r_eq::Vector{Vector{Float64}},θ::Parameters,p::Primitives)
#     r = [tup[k] for tup in r_eq,k in 1:length(r_eq[1])][:]
#     sell = balance_sheet_alloc(r_eq,θ,p)
#     sell = [tup[k] for tup in sell,k in 1:length(sell[1])][:]
#     q = market_shares(r_eq,θ)
#     q = [tup[k] for tup in q,k in 1:length(q[1])][:]

#     r_h = sum(r.*(1 .- sell).*q)/sum((1 .-sell).*q)
#     r_s = sum(r.*sell.*q)/sum(sell.*q)
#     sold = sum(sell.*q)/sum(q)
#     spread = (r_h - r_s)/0.0001
#     return spread, sold
    
# end

# function outcomes_plot(r_eq::Vector{Vector{Float64}},θ::Parameters,p::Primitives)
#     # r_eq = solve_eq(θ,p)
#     r_list = [tup[k] for tup in r_eq,k in 1:length(r_eq[1])][:]
#     sell = balance_sheet_alloc(r_eq,θ,p)
#     sell_list = [tup[k] for tup in sell,k in 1:length(sell[1])][:]
#     q = market_shares(r_eq,θ)
#     q_list = [tup[k] for tup in q,k in 1:length(q[1])][:]

#     risk = [x for x in p.λ,k in 1:length(q[1])][:]
#     dem = [x for x in θ.α,k in 1:length(q[1])][:]


    
#     return r_list, sell_list, q_list, risk, dem
    
# end


function simulateMoments(x::Vector{Float64},x2::Vector{Float64},
    M::Int,T::Int,J::Int,N::Int)

        θ = Parameters(x,J)
        α_mean = x2[1] # Mean price sensitivity
        α_var = x2[2] # Price sensitivity dispersion

        z_mean = x2[3] # Standard Normal Consumer Characteristics
        z_std = x2[4] 

        m0_mbs = x2[5] # MBS discount Rate
        m1_mbs = x2[6] # MBS discount slope

        ## Market Data 
        # Market Index
        # Lender Index
        # Market Characteristics
        # Lender Characteristics

        market_data = Matrix{Float64}(undef,M*T*J,J+2+1+1)
        market_data[:].=0.0
        # Initial federal funds rate
        ffr = range(0,0.05,length=T)

        index = 0
        m_ind = 0
        for m in 1:M
            for t in 1:T
            m_ind +=1
                for j in 1:J
                    index +=1
                    market_data[index,1] = m_ind
                    market_data[index,2] = j
                    market_data[index,3] = 1.0
                    market_data[index,4] = ffr[t]
                    market_data[index,4+j] = 1.0
                end
            end
        end

        ##### Data Structure
        ## Consumer Choice Data
        # Market Index
        # Consumer credit risk
        # Price Sensitivity
        # Chosen Lender Index
        # Paid interest Rate
        # Loan Sold

        consumer_data = Matrix{Float64}(undef,M*T*N,7)

        index = 0 
        m_ind = 0
        for m in 1:M 
            for t in 1:T
            m_ind +=1
                for n in 1:N
                    index +=1 
                    consumer_data[index,1] = m_ind
                    consumer_data[index,2] = z_mean + randn()*z_std
                    consumer_data[index,3] = α_mean - α_var/2 + rand()*α_var
                end
            end
        end




        demand_ind = Int.(5:size(market_data,2))
        cost_ind = Int.(3:size(market_data,2))
        discount_ind = Int.(3:4)

        spread_trend = Vector{Float64}(undef,M*T)
        sell_trend = Vector{Float64}(undef,M*T)
        ffr_trend = Vector{Float64}(undef,M*T)
        
        rate_avg = Vector{Float64}(undef,M*T)
        rate_no_otd = Vector{Float64}(undef,M*T)

        for m in 1:(M*T)
            market_consumers = findall(consumer_data[:,1].==m)
            r_h = Vector{Float64}(undef,length(market_consumers))
            s_h = Vector{Float64}(undef,length(market_consumers))
            r_s = Vector{Float64}(undef,length(market_consumers))
            r_no_otd = Vector{Float64}(undef,length(market_consumers))
            sell_vec = Vector{Float64}(undef,length(market_consumers))
            for (ind,i) in enumerate(market_consumers)
                market = consumer_data[i,1]
                market_index = market_data[:,1].==market
                X = market_data[market_index,demand_ind]
                W = market_data[market_index,cost_ind]
                D = market_data[market_index,discount_ind]
                z = consumer_data[i,2]
                dat = Data(X,W,D,z)

                MBS_par = market_data[market_index,discount_ind[1]][1]*m0_mbs + market_data[market_index,discount_ind[2]][1]*m1_mbs
                MBS_par = MBS(MBS_par)

                α = consumer_data[i,3]

                r_eq, itr1 = solve_eq(α,dat,θ,MBS_par)
                shares = market_shares(r_eq,α,dat,θ)
                sell = balance_sheet_alloc(r_eq,α,dat,θ,MBS_par)

                r_h[ind] = sum(r_eq.*shares.*(1 .-sell))/sum(shares.*(1 .-sell))
                s_h[ind] = sum(shares.*(1 .-sell))
                r_s[ind] = sum(r_eq.*sell.*shares)/sum(sell.*shares)
                sell_vec[ind] = sum(sell.*shares)/sum(shares)
                ffr_trend[m] = market_data[market_index,discount_ind[2]][1]

                ## Counterfactual
                r_h_only, itr1 = solve_eq_hold_only(α,dat,θ)
                shares_hold = market_shares(r_h_only,α,dat,θ)
                r_no_otd[ind] = sum(r_h_only.*shares_hold)
                
            end
            spread_trend[m] = (sum(r_h.*s_h)/sum(s_h) - sum(r_s.*(1 .-s_h))/sum(1 .- s_h))/.0001
            sell_trend[m] = 1 .- mean(s_h)
            rate_avg[m] = mean(r_h.*s_h .+ r_s.*(1 .-s_h))
            rate_no_otd[m] = mean(r_no_otd)

        end

        return spread_trend,sell_trend,ffr_trend, rate_avg,rate_no_otd
end
