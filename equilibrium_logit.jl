# function equilibrium_obj(r::Vector{Float64},θ::Parameters,p::Primitives)
#     foc = expected_foc(r,θ,p).*100
#     return sum(foc.^2)
# end


# function solve_eq(θ::Parameters,p::Primitives)

#     r = Vector{Vector{Float64}}(undef,length(p.λ))
#     for i in eachindex(r)
#         r[i] = repeat([0.06],4)
#     end
#     step = 1e-3
#     for i in eachindex(r)
#         err = 1
#         while err>1e-12
#             foc = expected_foc(i,r[i],θ,p)
#             r[i] = r[i] .+ step.*foc
#             err = sum(foc.^2)
#             # println(r[i])
#         end
#     end
#     return r
# end



function solve_eq(α::Float64,d::Data,θ::Parameters,m::MBS;r_start::Vector{Float64}=Vector{Float64}(undef,0))
    
    r_min = min_rate(d,θ,m) 
    if length(r_start)==0
        r = r_min .+ 1 ./(-α)
    else
        r = copy(r_start)
    end
    B_k_inv = repeat([1e-4],length(r))
    foc_k = expected_foc(r,α,d,θ,m)
    err = 1 
    itr = 0
    while (err>1e-12) & (itr<250)
        itr = itr+1
        # step = -B_k_inv.*foc_k      
        r_new = r .+ B_k_inv.*foc_k
        r_new = max.(r_new,r_min.-0.01)
        r_new = min.(r_new,r_min.*15)
        step = r_new - r
        r = r_new

        foc_next = expected_foc(r,α,d,θ,m)
        y_k = foc_next - foc_k
        # B_k_inv = B_k_inv .+ (((step .- B_k_inv.*y_k).*step)./(step.*B_k_inv.*y_k)).*B_k_inv
        B_k_inv = abs.((step./y_k))
        B_k_inv[step.==0] .= 1e-4
        B_k_inv[y_k.==0] .= 1e-4

        foc_k = foc_next
        err = sum(foc_k.^2)
    end

    return r, itr
end

function solve_eq_robust(α::Float64,d::Data,θ::Parameters,m::MBS;r_start::Vector{Float64}=Vector{Float64}(undef,0))
    
    r_min = min_rate(d,θ,m)
    if length(r_start)==0
        r = r_min .+ 1 ./(-α)
    else
        r = copy(r_start)
    end
    B_k_inv = repeat([1e-6],length(r))
    foc_k = expected_foc(r,α,d,θ,m)
    err = 1 
    itr = 0
    while err>1e-12
        itr = itr+1
        step = B_k_inv.*foc_k   
        max_adj = 0.01
        if any(abs.(foc_k).>100)
            max_adj=0.001
        end
        if maximum(abs.(step))>max_adj
            step = max_adj.*(step./maximum(abs(step)))  
        end 
        r_new = r .+ step
        r_new = max.(r_new,r_min.-0.01)
        r_new = min.(r_new,r_min.*15)
        step = r_new - r
        r = r_new

        foc_next = expected_foc(r,α,d,θ,m)
        y_k = foc_next - foc_k
        sign_flip = (foc_next.*foc_k).<0
        B_k_inv = B_k_inv.*1.1
        if err>1e-5
            B_k_inv[sign_flip] .= 1e-6
        else
            B_k_inv[sign_flip] .= 1e-8
        end
        foc_k = foc_next
        err = sum(foc_k.^2)
    end

    return r, itr
end


function solve_eq_hold_only(α::Float64,d::Data,θ::Parameters)
    r = repeat([0.06],size(d.X,1))
    B_k_inv = diagm(repeat([1e-6],length(r)))
    foc_k = hold_only_foc(r,α,d,θ)
    err = 1 
    itr = 0
    while err>1e-12
        itr = itr+1
        step = -B_k_inv*foc_k      
        r = r .+ B_k_inv*foc_k

        foc_next = hold_only_foc(r,α,d,θ)
        y_k = foc_next - foc_k
        B_k_inv = B_k_inv .+ (((step .- B_k_inv*y_k)*step')./(step'*B_k_inv*y_k))*B_k_inv

        foc_k = foc_next
        err = sum(foc_k.^2)
        if itr%100==0
            println("Interest $r at iteration $itr with error $err")
        end
    end

    return r, itr
end


function solve_eq_r(r0::Float64,j::Int64,d::Data,θ::Parameters,m::MBS)
    α = -100.0
    r, itr = solve_eq(α,d,θ,m)

    B_k_inv = 10

    f_k = r[j] - r0
    err = 1 
    while err>1e-12
        step = -B_k_inv*f_k    
        α_new = α .+ step
        α_new = min(-10,α_new)
        α_new = max(-200,α_new)
        step = α_new - α
        α = α_new
        r, i_new = solve_eq(α,d,θ,m,r_start=r)
        itr+= i_new
        f_next = r[j] - r0
        y_k = f_next - f_k
        # B_k_inv = B_k_inv + (((step - B_k_inv*y_k)*step)/(step*B_k_inv*y_k))*B_k_inv
        B_k_inv = abs.((step./y_k))

        f_k = f_next
        err = f_k.^2
        # println("Parameter $α at iteration $itr with error $err")
    end

    return α,r, itr
end



# function solve_eq_r(r0::Float64,j::Int64,d::Data,θ::Parameters,m::MBS)
#     α_init = -100.0
#     r = repeat([0.06],size(d.X,1))
#     x = vcat(r,α_init)

#     init_step = repeat([1e-6],length(r)+1)
#     B_k_inv = diagm(init_step)

#     f(x) = vcat(expected_foc(x[1:length(r)],x[length(x)],d,θ,m),x[j]-r0)

#     foc_k = f(x)
#     err = 1 
#     itr = 0
#     while err>1e-12
#         itr = itr+1
#         step = -B_k_inv*foc_k      
#         x = x .+ B_k_inv*foc_k
#         foc_next = f(x)
#         y_k = foc_next - foc_k
#         B_k_inv = B_k_inv .+ (((step .- B_k_inv*y_k)*step')./(step'*B_k_inv*y_k))*B_k_inv

#         foc_k = foc_next
#         err = sum(foc_k.^2)
#     end

#     return x[j],r, itr
# end

# function solve_eq_r(r0::Float64,j::Int64,d::Data,θ::Parameters,m::MBS)
#     α_init = - 100
#     r = repeat([0.06],size(d.X,1))
#     r[j] = r0
#     update_ind = Int.(1:length(r))
#     update_ind = update_ind[update_ind.!=j]

#     init_step = repeat([1e-6],length(r))
#     # init_step[j] = 0.001
#     B_k_inv = diagm(init_step)

#     x = Vector{Float64}(undef,length(r))
#     x[update_ind] = r[update_ind]
#     x[j] = α_init
#     foc_k = expected_foc(r,x[j],d,θ,m)
#     err = 1 
#     itr = 0
#     while err>1e-12
#         itr = itr+1
#         step = -B_k_inv*foc_k      
#         x_new = x .+ B_k_inv*foc_k
#         x_new[j] = max(min(-10,x_new[j]),-500)
#         step = x .- x_new
#         x = x_new
#         r[update_ind] = x[update_ind]
#         foc_next = expected_foc(r,x[j],d,θ,m)
#         y_k = foc_next - foc_k
#         B_k_inv = B_k_inv .+ (((step .- B_k_inv*y_k)*step')./(step'*B_k_inv*y_k))*B_k_inv

#         foc_k = foc_next
#         err = sum(foc_k.^2)
#     end

#     return x[j],r, itr
# end

# function solve_eq_r_init(r0::Float64,j::Int64,d::Data,θ::Parameters,m::MBS)
#     α = -25.0
#     r = solve_eq(α,d,θ,m)
#     r[j] = r0
    
#     B_k_inv = 0.0001

#     foc_k = expected_foc(r,α,d,θ,m)[j]
#     err = 1 
#     itr = 0
#     while err>1e-12
#         itr = itr+1
#         step = -B_k_inv*foc_k     
#         α = α .- step

#         foc_next = expected_foc(r,α,d,θ,m)[j]
#         y_k = foc_next - foc_k
#         B_k_inv = B_k_inv + (((step - B_k_inv*y_k)*step)/(step*B_k_inv*y_k))*B_k_inv

#         foc_k = foc_next
#         err = sum(foc_k.^2)
#         println(α)
#     end
#     r = solve_eq(α,d,θ,m)
#     return α,r
# end
