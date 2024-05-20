function rate_spread(x::Vector{Float64},f::Float64,draws::Draws)

    p = Primitives(f,draws)
    θ = Parameters(x,p)

    res = model_outcomes(θ,p)

    p_s = sum(res.price_h.*res.sold.*res.search)/sum(res.search.*res.sold)
    p_h = sum(res.price_s.*(1 .- res.sold).*res.search)/sum(res.search.*(1 .- res.sold))
    p_avg = sum(res.price.*res.search)/sum(res.search)
    spread = (p_h-p_s).*10000

    frac_mon = mean(res.search[:,1])


    frac_sold = sum(res.sold.*res.search)./sum(res.search)
    frac_sold_mon = mean(res.sold[:,1])

    return spread, frac_sold, p_avg, p_s, p_h, frac_mon, frac_sold_mon
end

function mechanism_bid(x::Vector{Float64},f::Float64,draws::Draws)

    p = Primitives(f,draws)
    θ = Parameters(x,p)

    r_h = mean(r_hold_min.(θ.comp_ν,θ.comp_ϵ,Ref(θ),Ref(p)))
    r_s = mean(r_sell_min.(θ.comp_ν,θ.comp_ϵ,Ref(θ),Ref(p)))
    

    return r_h, r_s
end

function mechanism_profit(x::Vector{Float64},f::Float64,r::Float64,draws::Draws)

    p = Primitives(f,draws)
    θ = Parameters(x,p)

    pi_h = mean(π_hold.(Ref(r),θ.comp_ν,θ.comp_ϵ,Ref(θ),Ref(p)))
    pi_s = mean(π_sell.(Ref(r),θ.comp_ν,θ.comp_ϵ,Ref(θ),Ref(p)))
    

    return pi_h, pi_s
end

function mechanism_mon_profit(x::Vector{Float64},f::Float64,r::Float64,draws::Draws)

    p = Primitives(f,draws)
    θ = Parameters(x,p)


    s_exp = search_expectations.(Ref(Inf),θ.mon_ν,θ.mon_ϵ,Ref(θ),Ref(p))
    probs = [tup[1] for tup in s_exp]
    profit = [tup[2] for tup in s_exp]
    V2 = [tup[3][2] for tup in s_exp]

    pi_h = mean(monopoly_expected_profit_hold.(Ref(r),θ.mon_ν,θ.mon_ϵ,probs,profit,V2,Ref(θ),Ref(p)))
    pi_s = mean(monopoly_expected_profit_sell.(Ref(r),θ.mon_ν,θ.mon_ϵ,probs,profit,V2,Ref(θ),Ref(p)))
    

    return pi_h, pi_s
end

function counterfactual(x::Vector{Float64},f::Float64,draws::Draws)

    p = Primitives(f,draws)
    θ = Parameters(x,p)
    
    res = model_outcomes(θ,p)
    p_avg = sum(res.price.*res.search)/sum(res.search)
    p_mon_avg = mean(res.price[:,1])
    # p_avg = mean(res.price[:,1])
    s_avg = mean(res.search,dims=1)
    
    
    res_no_otd = model_outcomes(θ,p,allow_otd=false)
    p_avg_no_otd = sum(res_no_otd.price.*res_no_otd.search)/sum(res_no_otd.search)
    p_mon_avg_no_otd  = mean(res_no_otd.price[:,1])
    # p_avg_no_otd = mean(res_no_otd.price[:,1])
    s_avg_no_otd = mean(res_no_otd.search,dims=1)
    

    return p_avg, p_avg_no_otd,p_mon_avg,p_mon_avg_no_otd, s_avg, s_avg_no_otd
end



function toy_est_obj(x::Vector{Float64},d::Draws)
    # ffr_moment_vector =       [0.0025, 0.02, 0.04,    0.06]
    # ffr_rate_spread_moments = [0.0020, 0.00, -0.0040, -0.0070].*1000
    # ffr_sec_moments          = [0.72,  0.68, 0.60,    0.5]
    ffr_moment_vector =       [2.036013005,
    2.058688684,
    2.341535253,
    2.504161013,
    2.656009747,
    2.788193255,
    2.931513406,
    2.972486109,
    3.074867391,
    3.122463435,
    3.201818417,
    3.187857012,
    2.979812745,
    2.882521343,
    2.833175659,
    2.685020937,
    2.629698486,
    2.24949525,
    2.299916435,
    2.39870715,
    2.484054301,
    2.386607839,
    2.328249455,
    2.28357292,
    2.215029671,
    2.180538061,
    1.613080353,
    1.526006764,
    1.320253785,
    1.198833933,
    1.203065421,
    1.048888456,
    0.927914418,
    0.739718437,
    0.617374065,
    0.404397366,
    0.176453422,
    -0.084540606,
    -0.366125207,
    -0.38276945,
    -0.441031603,
    -0.170300513,
    0.104192525,
    0.139556624,
    0.09155041,
    0.158592963,
    0.43547454,
    0.895625444,
    1.225973693,
    1.969154101,
    2.807265355,
    3.541883098,
    3.601142285,
    4.129581788,
    4.473832625,
    4.66902676,
    5.429726741,
    6.238280457,
    6.515894141,
    6.230044197]./100
    ffr_rate_spread_moments = [-0.198247324,
    -0.22209102,
    -0.306752483,
    -0.279492003,
    -0.284289454,
    -0.309434766,
    -0.259255192,
    -0.242577564,
    -0.230034661,
    -0.237107248,
    -0.255159944,
    -0.221561486,
    -0.223684529,
    -0.567709127,
    -0.133411253,
    -0.070784255,
    -0.064274897,
    -0.043421982,
    -0.125,
    -0.011010393,
    0.036352712,
    0.214943404,
    -0.041104569,
    -0.070130426,
    -0.026049044,
    0.439385191,
    0.029658834,
    0.037198295,
    -0.000979139,
    0.080636246,
    0.421518205,
    0.234822413,
    0.250557198,
    0.209520146,
    0.167655997,
    0.056593885,
    0.251177082,
    0.215569523,
    0.127781065,
    0.041624806,
    0.044214067,
    0.072770235,
    0.079813348,
    0.10176876,
    0.088047555,
    0.048815033,
    -0.016072332,
    -0.062373234,
    -0.058316113,
    -0.23096834,
    -0.350217238,
    -0.532774102,
    -0.787340108,
    -0.82050072,
    -0.808383096,
    -0.620093229,
    -0.499479506,
    -0.706042519,
    -0.785228957,
    -0.570284265]
    ffr_sec_moments          = [0.70506344,
    0.700693023,
    0.691611986,
    0.689412743,
    0.696716422,
    0.699800777,
    0.692013334,
    0.677727013,
    0.672011471,
    0.631496185,
    0.535767551,
    0.318443922,
    0.66328705,
    0.673085792,
    0.680143284,
    0.693142791,
    0.69311741,
    0.695719817,
    0.718670126,
    0.717005971,
    0.720789881,
    0.674333545,
    0.570542203,
    0.315412915,
    0.68244897,
    0.712437961,
    0.759405856,
    0.791819392,
    0.787419228,
    0.80638472,
    0.826396329,
    0.828752085,
    0.831865889,
    0.813036353,
    0.714116038,
    0.371555674,
    0.829572102,
    0.838568519,
    0.812071571,
    0.780137568,
    0.760751088,
    0.757737777,
    0.755074073,
    0.755023232,
    0.73760855,
    0.698405249,
    0.59863758,
    0.333778977,
    0.662870206,
    0.648652203,
    0.616820508,
    0.601845141,
    0.561275935,
    0.550878263,
    0.534847498,
    0.540364717,
    0.541074147,
    0.477116297,
    0.404694367,
    0.263554929]
    search_moments          = [0.502309996, 0.360460984, 0.116616744, 0.013606133, 0.007006143]
    search_discount_moments = [-0.00042,     -0.00095,  -0.0013, -0.0016]

    ffr_rate_spread_est = Vector{Float64}(undef,length(ffr_moment_vector))
    ffr_sec_est         = Vector{Float64}(undef,length(ffr_moment_vector))
    search_est          = Vector{Float64}(undef,5)
    search_discount_est = Vector{Float64}(undef,4)

    search_est[:].=0.0
    search_discount_est[:].=0.0


    for i in eachindex(ffr_moment_vector)
        f = ffr_moment_vector[i]
        p = Primitives(f,d)
        θ = Parameters(x,p)
        res = model_outcomes(θ,p)

        p_s = sum(res.price.*res.sold.*res.search)/sum(res.search.*res.sold)
        p_h = sum(res.price.*(1 .- res.sold).*res.search)/sum(res.search.*(1 .- res.sold))

        ffr_rate_spread_est[i] = (p_h - p_s).*100
        ffr_sec_est[i] = sum(res.sold.*res.search)./sum(res.search)

        s_prob = mean(res.search,dims=1)[:]
        s_rate = mean(res.price,dims=1)
        s_disc = s_rate[2:5] .- s_rate[1]

        search_est = search_est.+ s_prob./length(ffr_moment_vector)
        # search_discount_est = search_discount_est .+ s_disc/length(ffr_moment_vector)
    end

    obj_spread = sum( (ffr_rate_spread_moments.*10 .-ffr_rate_spread_est.*10).^2 )
    obj_sold   = sum( (ffr_sec_moments.-ffr_sec_est).^2)
    obj_search = sum( (search_moments.*10 .- search_est.*10).^2)
    # obj_discount = sum( (search_discount_moments.-search_discount_est).^2)/0.01

    # println("At Parameters $x")
    # println( "spread: $(ffr_rate_spread_est./10)")
    # println( "sold: $ffr_sec_est")
    # println( "search: $search_est")
    # println("discount: $search_discount_est")
    total = obj_spread+obj_sold+obj_search #+obj_discount
    return total
end


function toy_estimate(x::Vector{Float64},d::Draws)

    eval(x) = toy_est_obj(x,d)

    function eval(x,grad)
        obj = eval(x)
        return obj
    end
    options = Optim.Options(show_every=10,
                            show_trace = true,
                            g_tol = 1e-5,
                            iterations = 2000
                            )

    results = optimize(eval,x, options)
    println(results)
    return results
end