# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 14:57:00 2022

@author: Matteo Cencini
"""
import numpy as np
import numba as nb


from NumbaMinpack import minpack_sig, lmdif

np.random.seed(0)

x = np.linspace(0,2*np.pi,100)
y = np.sin(x)+ np.random.rand(100)


#%% Signal models
class InversionRecoveryT1Mapping:
    """
    Inversion-Recovery T1 Mapping related routines.
    """
    @staticmethod
    @nb.jit
    def signal_model(ti, A, B, T1):
        return A - B * np.exp(-ti / T1)
    
    @staticmethod
    def get_function_pointer():
        
        # get function
        func = InversionRecoveryT1Mapping.signal_model 
        
        @nb.cfunc(minpack_sig)
        def _optimize(res, params_, args_):
            # get parameters
            params = nb.carray(params_, (len(params_),))
            A, B, T1  = params
            
            # get variables
            nti = len(args) // 2
            args = nb.carray(args_, (2 * nti,))
            x = args[:nti]
            y = args[nti:]
            
            # compute residual
            for i in range(nti):
                res[i] = func(x[i], A, B, T1) - y[i] 
                
        return _optimize.address
    

class MultiechoTransverseRelaxationMapping:
    """
    Multi-echo Spin- (Gradient-) Echo T2 (T2*) Mapping related routines.
    """
    @staticmethod
    @nb.jit
    def signal_model(te, A, B, T2):
        return A*np.exp(- te / T2) + B 
    
    @staticmethod
    def get_function_pointer():
        
        # get function
        func = InversionRecoveryT1Mapping.signal_model 
        
        @nb.cfunc(minpack_sig)
        def _optimize(res, params_, args_):
            # get parameters
            params = nb.carray(params_, (len(params_),))
            A, B, T2  = params
            
            # get variables
            nte = len(args) // 2
            args = nb.carray(args_, (2 * nte,))
            x = args[:nti]
            y = args[nti:]
            
            # compute residual
            for i in range(nte):
                res[i] = func(x[i], A, B, T2) - y[i] 
                
        return _optimize.address


@nb.njit
def fast_function():  
    try:
        neqs = 100
        u_init = np.array([2.0,.8],np.float64)
        args = np.append(x,y)
        fitparam, fvec, success, info = lmdif(optimize_ptr , u_init, neqs, args)
        if not success:
            raise Exception

    except:
        print('doing something else')


    