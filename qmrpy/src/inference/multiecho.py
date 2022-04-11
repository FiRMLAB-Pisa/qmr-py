# -*- coding: utf-8 -*-
"""
Fitting routines and signal model for Multiecho Spin-Echo and Gradient-Echo data.

Created on Thu Feb 10 16:41:52 2022

@author: Matteo Cencini
"""
import numpy as np
import numba as nb


# from NumbaMinpack import minpack_sig, lmdif
from qmrpy.src.inference.LM import lmdif


__all__ = ['me_transverse_relaxation_fitting']


def me_transverse_relaxation_fitting(input: np.ndarray, te: float, mask: np.ndarray = None) -> np.ndarray:
    """
    Calculate t2/t2* maps from multiecho spin echo / gradient echo data.
    
    Args:
        input (ndarray): magnitude data of size (nte, nz, ny, nx)
        te (ndarray): array of echo times [ms].
        mask (ndarray): binary mask to accelerate fitting (optional)
        
    Returns:
        output (ndarray): T2 / T2* map of size (nz, ny, nx) in [ms].
    """
    # preserve input
    input = np.abs(input.copy())
    
    # process input
    te = te.astype(np.float64)
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
    # pointer = MultiechoTransverseRelaxationMapping.get_function_pointer(len(te))
    pointer = MultiechoTransverseRelaxationMapping.get_function_pointer()
    MultiechoTransverseRelaxationMapping.fitting(tmp, input, te, pointer)
        
    # reshape
    if mask is not None:
        output[mask] = tmp
    else:
        output = tmp.reshape(ishape[1:])
        
    # remove unphysical values
    output[output < 0] = 0
            
    return np.ascontiguousarray(output, dtype=np.float32)


class MultiechoTransverseRelaxationMapping:
    """
    Multi-echo Spin- (Gradient-) Echo T2 (T2*) Mapping related routines.
    """
    # @staticmethod
    # @nb.njit(cache=True, fastmath=True)
    # def signal_model(te, A, T2):
    #     return A * np.exp(- te / T2)
    
    # @staticmethod
    # def get_function_pointer(nte):
        
    #     # get function
    #     func = MultiechoTransverseRelaxationMapping.signal_model 
        
    #     @nb.cfunc(minpack_sig)
    #     def _optimize(params_, res, args_):
            
    #         # get parameters
    #         params = nb.carray(params_, (2,))
    #         A, T2  = params
            
    #         # get variables
    #         args = nb.carray(args_, (2 * nte,))
    #         x = args[:nte]
    #         y = args[nte:]
            
    #         # compute residual
    #         for i in range(nte):
    #             res[i] = func(x[i], A, T2) - y[i] 
                
    #     return _optimize.address
    
    @staticmethod
    def get_function_pointer():
        
        @nb.njit(cache=True, fastmath=True)
        def signal_model(params, args):
        
            # calculate elements
            f = (params[0] * np.exp(-params[1] * args[0])) - args[1]
            
            # analytical jacobian
            Jf0 = np.exp(-params[1] * args[0])
            Jf1 = -params[0] * args[0] * Jf0

            return f, np.stack((Jf0, Jf1), axis=0)
        
        return signal_model
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def fitting(output, input, te, optimize_ptr):
        
        # general fitting options
        nvoxels, neqs = input.shape
        initial_guess = np.array([1.0, 1 / 50.0], input.dtype) # M0, T2
                
        # loop over voxels
        for n in nb.prange(nvoxels):
            args = (te, input[n])  
            try:
                fitparam = lmdif(optimize_ptr , initial_guess, args)
                output[n] = 1 / fitparam[-1]
            except:
                pass
                
            # args = np.append(te, input[n])         
            # fitparam, fvec, success, info = lmdif(optimize_ptr , initial_guess, neqs, args)
            # if success:
            #     output[n] = fitparam[-1]
