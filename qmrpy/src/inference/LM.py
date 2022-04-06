"""
Levenberg-Marquardt related routines.
"""
import numpy as np
import numba as nb


@nb.njit(cache=True, fastmath=True)
def jacobian(func, initial, delta=1e-3):
    """
    https://rh8liuqy.github.io/Finite_Difference.html#jacobian-matrix
    """
    f = func
    nrow = len(f(initial))
    ncol = len(initial)
    output = np.zeros(nrow*ncol)
    output = output.reshape(nrow,ncol)
    for i in range(nrow):
      for j in range(ncol):
        ej = np.zeros(ncol)
        ej[j] = 1
        dij = (f(initial+ delta * ej)[i] - f(initial- delta * ej)[i])/(2*delta)
        output[i,j] = dij
        
    return output


@nb.njit(cache=True, fastmath=True)
def lmdif(fun, pars, args, tau = 1e-2, eps1 = 1e-6, eps2 = 1e-6, kmax = 20):
    """Implementation of the Levenberg-Marquardt algorithm in pure Python. Solves the normal equations.
    https://gist.github.com/geggo/92c77159a9b8db5aae73
    """
    p = pars
    f, J = fun(p, *args), jacobian(fun, p)

    A = np.inner(J,J)
    g = np.inner(J,f)

    I = np.eye(len(p))

    k = 0; nu = 2
    mu = tau * max(np.diag(A))
    stop = np.norm(g, np.Inf) < eps1
    while not stop and k < kmax:
        k += 1
        try:
            d = np.solve( A + mu * I, -g)
        except np.linalg.LinAlgError:
            stop = True
            break

        if np.norm(d) < eps2*(np.norm(p) + eps2):
            stop = True
            break

        pnew = p + d

        fnew, Jnew = fun(pnew, *args), jacobian(fun, pnew)
        rho = (np.norm(f)**2 - np.norm(fnew)**2)/np.inner(d, mu*d - g)
        
        if rho > 0:
            p = pnew
            A = np.inner(Jnew, Jnew)
            g = np.inner(Jnew, fnew)
            f = fnew
            J = Jnew
            if (np.norm(g, np.Inf) < eps1): # or norm(fnew) < eps3):
                stop = True
                break
            mu = mu * max([1.0/3, 1.0 - (2*rho - 1)**3])
            nu = 2.0
        else:
            mu = mu * nu
            nu = 2*nu
    else:
        pass
    
    return p

