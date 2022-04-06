# -*- coding: utf-8 -*-
"""
Fitting routines and signal model for Helmoltz-based Electric Properties Tomograph (hEPT).

Created on Thu Feb 10 16:41:52 2022

@author: Matteo Cencini
"""
import warnings


import numpy as np
import numba as nb
from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from scipy.ndimage import gaussian_filter


from skimage.restoration import unwrap_phase as unwrap
from NumbaMinpack import minpack_sig, lmdif


from qmrpy.src.inference import utils


# vacuum permeability
mu0 = 4.0e-7 * np.pi # [H/m]


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


__all__ = ['helmholtz_conductivity_fitting']


def helmholtz_conductivity_fitting(input: np.ndarray, resolution: np.ndarray, omega0: float,
                                   gaussian_kernel_sigma: float = None,
                                   laplacian_kernel_width: int = 20, fitting_threshold: float = 10.0, 
                                   median_filter_width: int = 20, mask: np.ndarray = None) -> np.ndarray:
    """
    Calculate conductivity map from bSSFP data.
    
    Args:
        input (ndarray): complex image data of size (nz, ny, nx)
        resolution (ndarray): array of image sizes along each axis (nz, ny, nx) in [m].
        omega0 (float): Larmor frequency (units: [rad / s]).
        gaussian_kernel_sigma (int): sigma of Gaussian kernel for image pre-processing.
        laplacian_kernel_width (int): full width of local parabolic fit region (half width per side).
        fitting_threshold (float): use only voxels with magnitude = +-10% of target voxel magnitude during fitting.
        median_filter_width (int): full width of adaptive median filter kernel (half width per side).
        mask (ndarray): binary mask to accelerate fitting (optional)
        
    Returns:
        output (ndarray): Conductivity map of size (nz, ny, nx) in [S/m].
    """
    # preserve input
    input = input.copy()
        
    # get mask
    if mask is None:
        mask = utils.mask(input)
    
    # preallocate output
    output = np.zeros(input.shape, dtype=np.float64)
        
    # preprocess input        
    input = np.ascontiguousarray(input, dtype=np.complex128)
    if gaussian_kernel_sigma is not None and gaussian_kernel_sigma > 0:
        input = gaussian_filter(input.real, gaussian_kernel_sigma) + 1j * gaussian_filter(input.imag, gaussian_kernel_sigma)
    
    # get magnitude and phase
    input_mag = mask * np.abs(input)
    input_phase = mask * np.angle(input)
    
    # uwrap
    true_phase = input_phase[input_phase.shape[0] // 2, input_phase.shape[1] // 2, input_phase.shape[2] // 2]
    input_phase = unwrap(input_phase)
    input_phase = input_phase - input_phase[input_phase.shape[0] // 2, input_phase.shape[1] // 2, input_phase.shape[2] // 2] + true_phase
        
    # preallocate output
    out_tmp, tmp, todo, mask_out, grid = HelmholtzConductivity.prepare_data(input_mag.copy(), input_phase.copy(), resolution, laplacian_kernel_width, fitting_threshold, mask)
    
    # do actual fit
    pointer = HelmholtzConductivity.get_function_pointer(laplacian_kernel_width)
    HelmholtzConductivity.fitting(out_tmp, tmp, grid, todo.astype(np.float64), pointer)
        
    # reshape
    output[mask_out] = out_tmp.sum(axis=-1)
    
    # finish computation
    output = -0.5 * output / omega0 / mu0
    
    # post-process
    if median_filter_width is not None and median_filter_width > 0:
        _, tmp, todo, mask_out, _ = HelmholtzConductivity.prepare_data(input_mag.copy(), output.copy(), resolution, median_filter_width, fitting_threshold, mask)
        
        tmp = [np.concatenate([tmp[n, ax, todo[n, ax]] for ax in range(3)]) for n in range(todo.shape[0])]
        out_tmp = np.zeros(todo.shape[0], dtype=np.float64)
        
        # do median filter
        HelmholtzConductivity.median_filter(out_tmp, tmp)
        
        # preallocate output
        output = np.zeros(input.shape, dtype=np.float32)
        output[mask_out] = out_tmp
               
    return np.ascontiguousarray(mask * output.astype(np.float32))


class HelmholtzConductivity:
    """
    Helmholtz-based Conductivity Mapping related routines.
    """
    @staticmethod
    @nb.njit(cache=True, fastmath=True)
    def signal_model(x, a, b, c):
        return a * x**2 + b * x + c
    
    @staticmethod
    def get_function_pointer(width):
        
        # get function
        func = HelmholtzConductivity.signal_model 
        
        @nb.cfunc(minpack_sig)
        def _optimize(params_, res, args_):
            
            # get parameters
            params = nb.carray(params_, (3,))
            a, b, c  = params
            
            # get variables
            args = nb.carray(args_, (3 * width,))
            x = args[:width]
            y = args[width:2*width]
            w = args[2*width:]
            
            # compute residual
            for i in range(width):
                res[i] = w[i] * (func(x[i], a, b, c) - y[i]) 
                
        return _optimize.address
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def fitting(output, input, grid, todo, optimize_ptr):
        
        # general fitting options
        nvoxels, _, neqs = input.shape
        initial_guess = np.array([10.0, 10.0, 10.0], input.dtype) # a, b, c
                                
        # loop over voxels
        for n in nb.prange(nvoxels):
            for ax in range(3):
                args = np.concatenate((grid[ax], input[n][ax], todo[n][ax]))         
                fitparam, fvec, success, info = lmdif(optimize_ptr , initial_guess, neqs, args)
                if success:
                    output[n, ax] = fitparam[0]
                
    @staticmethod
    def prepare_data(input_mag, value, resolution, kernel_width, fitting_threshold, mask):
        
        # determine number of non-zero element
        nvoxels = mask.sum()
        
        # determine half width
        width = kernel_width // 2
        
        # initialize output
        tmp = np.zeros((nvoxels, 3, kernel_width), dtype=value.dtype)
        
        # determine fitting points
        weight = tmp.copy()
        
        # get indexes of fitting voxels
        i, j, k = np.argwhere(mask).transpose()
        
        # pad
        input_mag = np.pad(input_mag, ((width, width), (width, width), (width, width)))
        value = np.pad(value, ((width, width), (width, width), (width, width)))
        
        i += width
        j += width
        k += width
        
        # fill temporary matrix
        for n in range(nvoxels):
            tmp[n, 0] = value[i[n]-width:i[n]+width, j[n], k[n]]
            tmp[n, 1] = value[i[n], j[n]-width:j[n]+width, k[n]]
            tmp[n, 2] = value[i[n], j[n], k[n]-width:k[n]+width]
            
            weight[n, 0] = input_mag[i[n]-width:i[n]+width, j[n], k[n]]
            weight[n, 1] = input_mag[i[n], j[n]-width:j[n]+width, k[n]]
            weight[n, 2] = input_mag[i[n], j[n], k[n]-width:k[n]+width]
            
        # determine voxels to be kept for fitting
        todo = 100 * np.abs(weight - weight[..., [width]]) / weight[..., [width]]
        todo = todo < fitting_threshold
         
        # be sure there are points on both sides
        ltodo = (todo[:, :, :width].sum(axis=-1) > 0).prod(axis=-1)
        rtodo = (todo[:, :, width+1:].sum(axis=-1) > 0).prod(axis=-1)
        keep = (ltodo * rtodo).astype(bool)
        
        # select voxels
        todo = todo[keep]
        tmp = tmp[keep]
        
        mask_out = np.zeros(mask.shape, mask.dtype)
        mask_out[np.argwhere(mask)[keep][:, 0], np.argwhere(mask)[keep][:, 1], np.argwhere(mask)[keep][:, 2]] = True
        
        # initialize grid
        grid = [resolution[ax] * np.arange(-width, width) for ax in range(3)]
        grid = np.stack(grid, axis=0)
                
        # preallocate output
        out = np.zeros((todo.shape[0], 3), grid.dtype)
        
        return out, tmp.astype(np.float64), todo, mask_out, grid
    
    @staticmethod
    @nb.njit(parallel=True, fastmath=True)
    def median_filter(output, input):
        
        # get number of voxels
        nvoxels = len(input)
        
        # loop over voxels
        for n in nb.prange(nvoxels):
            output[n] = np.median(input[n])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
