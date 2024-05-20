function objective(par_vec::Vector{Float64},
                targets::NamedTuple{(:price, :sold, :search), Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}},
                p::Primitives)
    θ = Parameters(par_vec,p)
    # println("Parameter Vector: $par_vec")
    price, sold, search = model_outcomes(θ,p)
    obj_price = sum((price .- targets.price).^2)*10
    obj_sold = sum( (sold .- targets.sold).^2 )/1000
    obj_search = sum( (search .- targets.search).^2 )/10
    total = obj_price+ obj_sold+ obj_search
    # println("price: $obj_price")
    # println("sold: $obj_sold")
    # println("search: $obj_search")
    # println("Total Objective: $total")
    return total 
end


function estimate(x::Vector{Float64},targets::NamedTuple,p::Primitives)

    eval(x) = objective(x,targets,p)

    function eval(x,grad)
        obj = eval(x)
        return obj
    end
    options = Optim.Options(show_every=25,
                            show_trace = true,
                            g_tol = 1e-10
                            )

    results = optimize(eval,x, options)
    println(results)
    return results
end


