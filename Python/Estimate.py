import numpy as np
import GMM as gmm

def newton_raphson(data,par,W,ftol=1e-3,valtol=1e-4,itr_max=10000):
    p0 = par.param_vec
    p_idx = list(range(len(p0))) #In case we want to fix some parameters, currently estimating all

    ### Print Format
    np.set_printoptions(precision=4)

    ### Initial Evaluation
    fval, G, H = gmm.compute_gmm_hessian(data,par,W)

    G = G[p_idx]
    H = H[np.ix_(p_idx,p_idx)]

    print('Function value at starting parameter:',"{:.8g}".format(fval))
    # print('Gradient value at starting parameter:',G)
    # print('hessian value at starting parameter:',H)

    # Norm for convergence tolerance
    grad_size = np.sqrt(np.dot(G,G))

    itr=0
    shock_attempt=0
    min_eval = fval
    slow_count = 0
    deviation_flag=0

    while (grad_size>ftol) & (fval>valtol) & (itr<itr_max):
        itr+=1

        # w, v = np.linalg.eig(H)
        # print('Hessian Eigen Minimum Absolute Value',min(np.absolute(w)))

        # if itr%200==0:
        #     deviation = np.random.rand(len(par.param_vec))*0.01 - 0.005
        #     par.update(deviation)
        #     print("Random Deviation Step")
        #     # Evaluation for next iteration
        #     fval, G, H = gmm.compute_gmm_hessian(data,par,W)
        #
        #     ## New derivatives
        #     G = G[p_idx]
        #     H = H[np.ix_(p_idx,p_idx)]
        #     grad_size = np.sqrt(np.dot(G,G))
        #     # Print Status Report
        #     print('Function value is',"{:.8g}".format(fval),'and gradient size is',"{:.3g}".format(grad_size),'on iteration number',itr)
        #
        #
        #     continue

        ## Current Parameter Vector
        param_cur = par.param_vec.copy()
        ## Candidate New Parameter Vector (Updated)
        param_new = par.param_vec.copy()
        ## Find NR step and new parameter vector
        if len(p_idx)>1:
            step = -np.matmul(G,np.linalg.inv(H))
        else:
            step = -G/H

        # # Cap step size
        cap_size = 1000
        check_cap = [abs(step[x])>cap_size for x in p_idx]
        if True in check_cap:
        #     cap = max(abs(step[p_idx]))
        #     step = step/cap*cap_size
            print(np.array(p_idx)[check_cap])
            print(param_cur[check_cap])
            print(step[check_cap])
            print('Hit step cap of ', cap_size)


        ## Candidate Update Vector
        param_new[p_idx] = param_cur[p_idx] + step
        par.set(param_new)
        # print('Now trying parameter vector', par.param_vec)
        # print('Step is ', step)

        ## Evaluate Function at new parameter vector
        new_fval = gmm.compute_gmm(data,par,W)

        ## If the function is not minimizing, update the parameter with a gradient step
        ## Do this anyway if the gradient is really large

        step_tol = 1.10
        if (grad_size>5e5) or (slow_count>10) or ((True in check_cap) and (grad_size>0.2)): # or (itr<10):
            print("Follow Gradient")
            new_fval = fval*100
            alpha = abs(1/np.diag(H))
            step = -G*alpha

            step_tol = 1.00

        reduction_itr= 0
        while (new_fval>fval*step_tol):
            reduction_itr +=1


            if reduction_itr==4:
                alpha = abs(1/np.diag(H))
                step = -G*alpha
                print("Switch to Gradient")
                step_tol = 1.00
            step = step/10
            # cap = max(abs(step[capped_params_idx])/param_cur[capped_params_idx])
            # if cap>0.5:
            #     step = step/cap*0.5
            #     print('Hit step cap of 50% parameter value on non_linear_parameters')

            # print("New value","{:.3g}".format(new_fval),"exceeds old value","{:.3g}".format(fval),"by too much")
            # print("Step along the gradient:",step)
            print("Reduced step")
            ## Candidate Update Vector
            param_new[p_idx] = param_cur[p_idx] + step
            par.set(param_new)
            # print('Now trying parameter vector', par.param_vec)
            # print('Step is ', step)
            ## Evaluate Function at new parameter vector
            new_fval = gmm.compute_gmm(data,par,W)

            ## If still not moving in right direction, smaller gradient step
            #alpha = alpha/10

        # Final Parameter Update
        par.set(param_cur)
        par.update(step)

        # Evaluation for next iteration
        fval, G, H = gmm.compute_gmm_hessian(data,par,W)

        if fval<min_eval*0.98:
            min_eval = fval
            slow_count = 0
        else:
            if slow_count<50:
                slow_count+=1
                print("Slow Count", slow_count)
            else:
                slow_count = 0
                deviation_flag=1


        ## Allow for estiamtion to finish even if it's not well identified (Currently only checking)
        check_unidentified = [x for x in p_idx if (param_cur[x]>1e7) or (param_cur[x]<-1e7) ]
        if len(check_unidentified)>0:
            print('Parameters running off to infinity: ', check_unidentified)
        # G[check_unidentified] = 0

        ## New derivatives
        G = G[p_idx]
        H = H[np.ix_(p_idx,p_idx)]
        grad_size = np.sqrt(np.dot(G,G))

        if (deviation_flag==1):
            shock_attempt+=1
            deviation = np.random.rand(len(par.param_vec))*par.param_vec*0.2 - par.param_vec*0.1
            par.update(deviation)
            print("Random Deviation Step")
            # Evaluation for next iteration
            fval, G, H = gmm.compute_gmm_hessian(data,par,W)
            deviation_flag=0
            min_eval = fval

            ## New derivatives
            G = G[p_idx]
            H = H[np.ix_(p_idx,p_idx)]
            grad_size = np.sqrt(np.dot(G,G))

        # Print Status Report
        print('Function value is',"{:.8g}".format(fval),'and gradient size is',"{:.3g}".format(grad_size),'on iteration number',itr, 'Parameter Track: ', par.param_vec[0])

    print('Solution!', par.param_vec)
    print('Function value is ',"{:.8g}".format(fval),'and gradient is',"{:.3g}".format(grad_size),'after',itr,'iterations')
    return fval, itr



def estimate_parallel(data,par,W,N=10):
    P_values = np.zeros((len(par.param_vec),N))
    for n in range(N):
        P_values[:,n]  = par.param_vec.copy() + np.random.rand(len(par.param_vec))*0.5 - 0.25
    fvals = np.zeros(N)
    lowest_fval = 1e8
    best_index = 0
    func_evals = 0
    while lowest_fval>1e-4:
        for n in range(N):
            print("Particle", n)
            par.set(P_values[:,n])
            fvals[n],itr = newton_raphson(data,par,W,ftol=1e-3,valtol=5e-4,itr_max=199)
            func_evals+=itr
            P_values[:,n] = par.param_vec.copy()

            if fvals[n]<lowest_fval:
                best_index = n
                lowest_fval = fvals[n]

        for n in range(N):
            print("Randomize around best iteration", best_index)
            P_best = P_values[:,best_index]
            if n!=best_index:
                P_values[:,n]  = P_best + np.random.rand(len(par.param_vec))*0.1 - 0.05


    par.set(P_values[:,best_index])
    fval = gmm_objective(data,par,W)
    G = compute_gmm_gradient(data,p,W)
    grad_size = np.sqrt(np.dot(G,G))
    print('Function value is ',"{:.8g}".format(fval),'and gradient is',"{:.3g}".format(grad_size),'after',func_evals,'evaluations')

    return
