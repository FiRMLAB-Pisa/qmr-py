# -*- coding: utf-8 -*-
"""
Fitting routines and signal model for Inversion-Recovery Spin-Echo data.

Created on Thu Feb 10 16:41:52 2022

@author: Matteo Cencini
"""
import numpy as np
import numba as nb


from NumbaMinpack import minpack_sig, lmdif


__all__ = ['ir_se_t1_fitting']



def ir_se_t1_fitting(input: np.ndarray, ti: float, mask: np.ndarray = None) -> np.ndarray:
    """
    Calculate t1 maps from inversion recovery spin echo data.
    
    Args:
        input (ndarray): inversion recovery magnitude data of size (nti, nz, ny, nx)
        ti (ndarray): array of inversion times [ms].
        mask (ndarray): binary mask to accelerate fitting (optional)
        
    Returns:
        output (ndarray): T1 map of size (nz, ny, nx) in [ms].
    """ 
    # preserve input
    input = np.abs(input.copy())
    
    # process input
    ti = ti.astype(np.float64)   
    ishape = input.shape
    
    if mask is not None:
        input = input[:, mask]
    else:
        input = input.reshape(ishape[0], np.prod(ishape[1:]))
        
    input = np.ascontiguousarray(input.transpose(), dtype=np.float64)
        
    # normalize
    s0 = input.max(axis=1)
    s0[s0 == 0] = 1    
    input /= s0[:, None]
    
    # preallocate output
    tmp = np.zeros(input.shape[0], input.dtype)
    output = np.zeros(ishape[1:], input.dtype)
            
    # do actual fit
    pointer = InversionRecoveryT1Mapping.get_function_pointer(len(ti))
    InversionRecoveryT1Mapping.fitting(tmp, input, ti, pointer)
    
    # reshape
    if mask is not None:
        output[mask] = tmp
    else:
        output = tmp.reshape(ishape[1:])
        
    # remove unphysical values
    output[output < 0] = 0
    
    return np.ascontiguousarray(output, dtype=np.float32)


class InversionRecoveryT1Mapping:
    """
    Inversion-Recovery T1 Mapping related routines.
    """        
    @staticmethod
    @nb.njit(cache=True, fastmath=True)
    def signal_model(ti, A, B, T1):
        return np.abs(B + A * np.exp(-ti / T1))
    
    @staticmethod
    def get_function_pointer(nti):
        
        # get function
        func = InversionRecoveryT1Mapping.signal_model 
        
        @nb.cfunc(minpack_sig)
        def _optimize(params_, res, args_):
            
            # get parameters
            params = nb.carray(params_, (3,))
            A, B, T1  = params
            
            # get variables
            args = nb.carray(args_, (2 * nti,))
            x = args[:nti]
            y = args[nti:]
            
            # compute residual
            for i in range(nti):
                res[i] = func(x[i], A, B, T1) - y[i] 
                
        return _optimize.address
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def fitting(output, input, ti, optimize_ptr):
        
        # general fitting options
        nvoxels, neqs = input.shape
        initial_guess = np.array([-2.0, 1.0, 1000.0], input.dtype) # A, B, T1
        
        # loop over voxels
        for n in nb.prange(nvoxels):            
            args = np.append(ti, input[n])
            fitparam, fvec, success, info = lmdif(optimize_ptr , initial_guess, neqs, args)
            if success:
                output[n] = fitparam[-1]