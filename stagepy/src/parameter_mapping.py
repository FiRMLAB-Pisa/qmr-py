# -*- coding: utf-8 -*-
"""
Fitting routines for parameter mapping.

Created on Mon Feb  7 14:57:00 2022

@author: Matteo Cencini
"""
import numpy as np
import numba as nb


from scipy import ndimage


from NumbaMinpack import minpack_sig, lmdif


def ir_se_t1_fitting(input, ti, mask=None):
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
    
    return np.ascontiguousarray(output, dtype=np.float32)

    
def me_transverse_relaxation_fitting(input, te, mask=None):
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
    pointer = MultiechoTransverseRelaxationMapping.get_function_pointer(len(te))
    MultiechoTransverseRelaxationMapping.fitting(tmp, input, te, pointer)
        
    # reshape
    if mask is not None:
        output[mask] = tmp
    else:
        output = tmp.reshape(ishape[1:])
    
    return np.ascontiguousarray(output, dtype=np.float32)


def b1_dam_fitting(input, fa, mask=None):
    """
    Calculate b1+ maps from dual flip angle data.
    
    Args:
        input (ndarray): magnitude data of size (2, nz, ny, nx)
        fa (ndarray): array of flip angles [deg]
        mask (ndarray): binary mask for clean-up (optional)
        
    Returns:
        output (ndarray): B1+ scaling factor map of size (nz, ny, nx).
    """
    # check format
    assert len(fa) == 2, "DAM requires two flip angle only"
    
    # preserve input
    input = np.abs(input.copy())
    
    # get min and max flip angle
    min_flip, min_flip_ind = fa.min(), fa.argmin()
    max_flip, max_flip_ind = fa.max(), fa.argmax()
    
    # check we actually have a double angle series
    assert max_flip / min_flip == 2, "Acquired angles must be x and 2x"
    
    # calculate cos x
    max_flip_img = input[max_flip_ind]
    min_flip_img = input[min_flip_ind]
    min_flip_img[min_flip_img == 0] = 1
    cos_flip = 0.5 * max_flip_img / min_flip_img
    
    # clean up unmeaningful values
    cos_flip[cos_flip < 0] = 0
    cos_flip[cos_flip > 1] = 1
    
    # actual calculation
    b1map = np.rad2deg(np.arccos(cos_flip)) / min_flip
    
    # final cleanup
    b1map = np.nan_to_num(b1map)
    
    # mask
    if mask is not None:
        b1map = mask * b1map
        
    return b1map

#%% Utils
def mask(input, threshold=0.05):
    """
    Generate binary mask from input data.
    
    Args:
        input (ndarray): input image space data.
        threshold (float): keeps data below 0.05 * input.max()
    
    Returns:
        mask (ndarray): output binary mask of tissues.
    """
    # preserve input
    input = np.abs(input.copy())
    
    # select input volume
    if len(input.shape) == 4: # image series (contrast, nz, ny, nx)
        input = input[0]
        
    # normalize
    input = input / input.max()
    
    # get mask
    mask = input > threshold
    
    # clean mask
    mask = ndimage.binary_opening(mask, structure=np.ones((1, 2, 2)))
    mask = ndimage.binary_closing(mask, structure=np.ones((1, 2, 2)))

    return mask

#%% Signal models
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
                

class MultiechoTransverseRelaxationMapping:
    """
    Multi-echo Spin- (Gradient-) Echo T2 (T2*) Mapping related routines.
    """
    @staticmethod
    @nb.njit(cache=True, fastmath=True)
    def signal_model(te, A, T2):
        return A * np.exp(- te / T2)
    
    @staticmethod
    def get_function_pointer(nte):
        
        # get function
        func = MultiechoTransverseRelaxationMapping.signal_model 
        
        @nb.cfunc(minpack_sig)
        def _optimize(params_, res, args_):
            
            # get parameters
            params = nb.carray(params_, (2,))
            A, T2  = params
            
            # get variables
            args = nb.carray(args_, (2 * nte,))
            x = args[:nte]
            y = args[nte:]
            
            # compute residual
            for i in range(nte):
                res[i] = func(x[i], A, T2) - y[i] 
                
        return _optimize.address
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def fitting(output, input, te, optimize_ptr):
        
        # general fitting options
        nvoxels, neqs = input.shape
        initial_guess = np.array([1.0, 50.0], input.dtype) # M0, T2
                
        # loop over voxels
        for n in nb.prange(nvoxels):
            args = np.append(te, input[n])         
            fitparam, fvec, success, info = lmdif(optimize_ptr , initial_guess, neqs, args)
            if success:
                output[n] = fitparam[-1]
            




    