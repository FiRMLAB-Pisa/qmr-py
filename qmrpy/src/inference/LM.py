"""
Levenberg-Marquardt related routines.
"""
import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def jacobian(fun, pars, args, delta=1e-12):
    """
    https://rh8liuqy.github.io/Finite_Difference.html#jacobian-matrix
    """
    nout = len(fun(pars, args))
    npars = pars.shape[0]
         
    # initialize output and versor in parameter space
    output = np.zeros((npars, nout), dtype=args[0].dtype)
    ei = np.zeros(npars, dtype=args[0].dtype)
    
    for i in range(npars):
        ei[i] = 1
        dij = (fun(pars + delta * ei, args) - fun(pars- delta * ei, args)) / (2 * delta)
        ei[i] = 0
        output[i] = dij
        
    return output


@nb.njit(cache=True, fastmath=True)
def lmdif(fun, initial_guess, args, tau=1e-2, eps1=1e-6, eps2=1e-6, maxiter=20):
    """Implementation of the Levenberg-Marquardt algorithm in pure Python. Solves the normal equations.
    https://gist.github.com/geggo/92c77159a9b8db5aae73
    
    Args:
        fun: function computing residuals of the fitting model: 
                fun(pars, (predictors, observations) = observations - model(pars, predictors)
        pars: fitting parameters
        args: tuple with (predictors, observations)
    """
    p = initial_guess
    f, J = fun(p, args)

    A = np.dot(J, J.T)
    g = np.dot(J, f.T)
    mu = tau * max(np.diag(A))

    I = np.eye(len(p), dtype=f.dtype)
    
    # initialize parameters
    niter = 0
    nu = 2
    stop = np.linalg.norm(g, np.Inf) < eps1
    
    while not stop and niter < maxiter:
        niter += 1

        try:
            d = np.linalg.solve(A + mu * I, -g)
        except: # singular matrix encountered
            return 0 * p

        if np.linalg.norm(d) < eps2 * (np.linalg.norm(p) + eps2): # small step
            return p
        
        # update
        pnew = p + d
        fnew, Jnew = fun(pnew, args)
        rho = (np.linalg.norm(f)**2 - np.linalg.norm(fnew)**2) / np.dot(d, (mu * d - g).T)
    
        if rho > 0:
            p = pnew
            A = np.dot(Jnew, Jnew.T)
            g = np.dot(Jnew, fnew.T)
            f = fnew
            J = Jnew
            if (np.linalg.norm(g, np.Inf) < eps1): # small gradient
                return p      
            mu = mu * max([1.0 / 3, 1.0 - (2 * rho - 1)**3])
            nu = 2.0
        else:
            mu = mu * nu
            nu = 2 * nu
            
    else: # max iter reached
        return p
    
    return p

