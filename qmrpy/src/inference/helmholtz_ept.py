# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:54:03 2022

@author: mcencini
"""
import warnings


import numpy as np
import numba as nb
from scipy import ndimage


from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from scipy.ndimage import gaussian_filter


from skimage.restoration import unwrap_phase as unwrap


from qmrpy.src.inference import utils


# vacuum permeability
mu0 = 4.0e-7 * np.pi # [H/m]


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


__all__ = ['PhaseBasedLaplacianEPT', 'PhaseBasedSurfaceIntegralEPT']


# vacuum permeability
mu0 = 4.0e-7 * np.pi # [H/m]


def PhaseBasedLaplacianEPT(input: np.ndarray, resolution: np.ndarray, omega0: float,
                           gaussian_preprocessing_sigma: float = 0.0, gaussian_weight_sigma: float = 0.45,
                           kernel_size: int = 16, kernel_shape='cross', 
                           median_filter_width: int = 0, segmentation_mask: np.ndarray = None) -> np.ndarray:
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
    input = input.copy().astype(np.complex64)
        
    # get segmentation mask
    if segmentation_mask is None:
        segmentation_mask = utils.mask(input)
        
    # reformat segmentation mask
    if len(segmentation_mask.shape) == 3:
        segmentation_mask = segmentation_mask[None, ...]
   
    # get mask
    mask = segmentation_mask.sum(axis=0)
              
    # preprocess input        
    if gaussian_preprocessing_sigma is not None and gaussian_preprocessing_sigma > 0:
        input = gaussian_filter(input.real, gaussian_preprocessing_sigma) + 1j * gaussian_filter(input.imag, gaussian_preprocessing_sigma)
    
    # get magnitude and phase
    magnitude = mask * np.abs(input)
    mask = mask.astype(bool)
    
    # Normalise magnitude
    magnitude -= magnitude[mask].mean()
    if magnitude[mask].std() != 0: 
        magnitude /=  magnitude[mask].std()
    
    magnitude[np.invert(mask)] = 100

    # prepare phase
    phase = np.angle(input)
    
    # uwrap
    true_phase = phase[phase.shape[0] // 2, phase.shape[1] // 2, phase.shape[2] // 2]
    phase = mask * unwrap(phase)
    phase = phase - phase[phase.shape[0] // 2, phase.shape[1] // 2, phase.shape[2] // 2] + true_phase
    
    # transceive phase approximation
    phase *= -0.5
    
    # clean phase
    phase = np.nan_to_num(phase).astype(np.float32)
    
    # actual computation
    conductivity = _PhaseBasedLaplacianEPT(phase, magnitude, segmentation_mask.copy(), omega0, resolution, kernel_size, kernel_shape, gaussian_weight_sigma)
    
    # post process
    if median_filter_width is not None and median_filter_width > 0:
        conductivity = _adaptive_median_filter(conductivity, segmentation_mask.copy(), median_filter_width, kernel_shape)
        
    return conductivity


def PhaseBasedSurfaceIntegralEPT(input: np.ndarray, resolution: np.ndarray, omega0: float,
                                 gaussian_preprocessing_sigma: float = 0.0, gaussian_weight_sigma: float = 0.45,
                                 kernel_diff_size: int = 16, kernel_int_size: int = 32, kernel_shape='cross', 
                                 median_filter_width: int = 0, segmentation_mask: np.ndarray = None) -> np.ndarray:
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
    input = input.copy().astype(np.complex64)
        
    # get segmentation mask
    if segmentation_mask is None:
        segmentation_mask = utils.mask(input)
        
    # reformat segmentation mask
    if len(segmentation_mask.shape) == 3:
        segmentation_mask = segmentation_mask[None, ...]
        
    # get mask
    mask = segmentation_mask.sum(axis=0)
              
    # preprocess input        
    if gaussian_preprocessing_sigma is not None and gaussian_preprocessing_sigma > 0:
        input = gaussian_filter(input.real, gaussian_preprocessing_sigma) + 1j * gaussian_filter(input.imag, gaussian_preprocessing_sigma)
    
    # get magnitude and phase
    magnitude = mask * np.abs(input)
    mask = mask.astype(bool)
    
    # Normalise magnitude
    magnitude -= magnitude[mask].mean()
    if magnitude[mask].std() != 0: 
        magnitude /=  magnitude[mask].std()
    
    magnitude[np.invert(mask)] = 100

    # prepare phase
    phase = np.angle(input)
    
    # uwrap
    true_phase = phase[phase.shape[0] // 2, phase.shape[1] // 2, phase.shape[2] // 2]
    phase = mask * unwrap(phase)
    phase = phase - phase[phase.shape[0] // 2, phase.shape[1] // 2, phase.shape[2] // 2] + true_phase
    
    # transceive phase approximation
    phase *= -0.5
    
    # clean phase
    phase = np.nan_to_num(phase)
    
    # actual computation
    conductivity = _PhaseBasedSurfaceIntegralEPT(phase, magnitude, segmentation_mask.copy(), omega0, resolution, kernel_diff_size, kernel_int_size, kernel_shape, gaussian_weight_sigma)
    
    # post process
    if median_filter_width is not None and median_filter_width > 0:
        conductivity = _adaptive_median_filter(conductivity, segmentation_mask.copy(), median_filter_width, kernel_shape)
        
    return conductivity


def _PhaseBasedLaplacianEPT(phase, magnitude, segmentation, omega0, resolution=1, kernel_size=26, kernel_shape='cross', gauss_sigma=0.45):
    
    # check if  resolution is scalar
    if isinstance(resolution, (np.ndarray, list, tuple)) is False:
        resolution = np.array(3 * [resolution], dtype=np.float32)
    else:
        resolution = np.asarray(resolution, dtype=np.float32)
        
    # prepare data
    data_prep = DataReformat(kernel_size)
    phase, magnitude, segmentation, ind = data_prep.prepare_data(phase, magnitude, segmentation)
    
    # do laplacian
    laplacian_operator = LocalDerivative(kernel_size, gauss_sigma, shape=kernel_shape, dx=resolution, order=2)
    laplacian = laplacian_operator(ind, phase, magnitude, segmentation)
    
    # calculate conductivity
    conductivity = laplacian.sum(axis=0) / omega0 / mu0
        
    # crop
    return data_prep.reformat_data(conductivity)
            
    
def _PhaseBasedSurfaceIntegralEPT(phase, magnitude, segmentation, omega0, resolution=1, kernel_diff_size=20, kernel_int_size=40, kernel_shape='ellipsoid', gauss_sigma=0.45):
    
    # check if  resolution is scalar
    if isinstance(resolution, (np.ndarray, list, tuple)) is False:
        resolution = np.array(3 * [resolution], dtype=np.float32)
    else:
        resolution = np.asarray(resolution, dtype=np.float32)
        
    # prepare data
    grad_prep = DataReformat(kernel_diff_size)
    phase, mag_diff, seg_diff, ind = grad_prep.prepare_data(phase, magnitude, segmentation)
    
    # do laplacian
    gradient_operator = LocalDerivative(kernel_diff_size, gauss_sigma, shape=kernel_shape, dx=resolution, order=1)
    phase_gradient = gradient_operator(ind, phase, mag_diff, seg_diff)
    
    # crop
    phase_gradient = [grad_prep.reformat_data(phase_ax) for phase_ax in phase_gradient]
    phase_gradient = np.stack(phase_gradient, axis=0)
    
    # prepare data
    data_prep = DataReformat(kernel_int_size)
    phase_gradient, mag_int, seg_int, ind = data_prep.prepare_data(phase_gradient, magnitude, segmentation)
    
    # do laplacian
    integral_operator = SurfaceIntegral(kernel_int_size, gauss_sigma, shape=kernel_shape, dx=resolution)
    integral = integral_operator(ind, phase_gradient, mag_int.astype(np.float32), seg_int)
    
    # calculate conductivity
    conductivity = integral / omega0 / mu0
    
    # crop
    return data_prep.reformat_data(conductivity)


#%% padding / cropping utils
class DataReformat:
    
    def __init__(self, kernel_size):       
        # check if  kernel size is scalar
        if isinstance(kernel_size, (np.ndarray, list, tuple)) is False:
            kernel_size = np.array(3 * [kernel_size], dtype=np.int16)
        else:
            kernel_size = np.asarray(kernel_size, dtype=np.int16)
            
        self._kernel_size = kernel_size
        
    def prepare_data(self, phase, magnitude, segmentation):
        """
        Prepare data for local differentiation by padding.
        
        Args:
            kernel_size (int or tuple-like): size of differentiation window.
            input (list-of-ndarray): input data (phase, magnitude, segmentation).
            
        Returns
            *output (list-of-ndarray): output padded data.
            idx (ndarray): indexes of original (non-padded) voxels.
            mask (ndarray): binary mask for output selection.
        """        
        # get kernel
        kernel_size = self._kernel_size
            
        # get input and output shape
        if len(phase.shape) == 3:
            ishape = phase.shape
        else:
            ishape = phase.shape[1:]
        oshape = [ishape[n] + kernel_size[n] for n in range(3)]
        self.output = np.zeros(ishape, phase.dtype)
        
        # pad phase
        if len(phase.shape) == 3:
            phase = DataReformat._resize(phase, oshape)
        else:
            phase = DataReformat._resize(phase, [phase.shape[0]] + oshape)
            
        magnitude = DataReformat._resize(magnitude, oshape)
        segmentation = DataReformat._resize(segmentation, [segmentation.shape[0]] + oshape)
        
        # prepare mask
        imask = np.ones(ishape, dtype=bool)
        omask = DataReformat._resize(imask, oshape).astype(bool)
        self.imask = imask
        self.omask = omask
        
        # prepare indexes
        ind = []
        
        for n in range(segmentation.shape[0]):
            i, j, k = np.argwhere(segmentation[n]).transpose()
            ind.append(np.stack([i, j, k], axis=-1))
            
        ind = np.stack(ind, axis=0)
        
        return phase, magnitude, segmentation, ind
    
    def reformat_data(self, input):
        """
        Reformat data for output.
        
        Args:
            input (list-of-ndarray): input data (phase, magnitude, segmentation).
    
        Returns
            *output (list-of-ndarray): output cropped data.
        """
        output = self.output.copy()
        
        # assign
        output[self.imask] = input[self.omask]
        
        return output
        
    @staticmethod
    def _resize(input, oshape):
        """
        Resize with zero-padding or cropping.
    
        Args:
            input (array): Input array.
            oshape (tuple of ints): Output shape.
    
        Returns:
            array: Zero-padded or cropped result.
        """
        ishape1, oshape1 = DataReformat._expand_shapes(input.shape, oshape)
    
        if ishape1 == oshape1:
            return input.reshape(oshape)
    
        # get ishift and oshuft
        ishift = [max(i // 2 - o // 2, 0) for i, o in zip(ishape1, oshape1)]
        oshift = [max(o // 2 - i // 2, 0) for i, o in zip(ishape1, oshape1)]
    
        copy_shape = [min(i - si, o - so) for i, si, o, so in zip(ishape1, ishift, oshape1, oshift)]
        
        islice = tuple([slice(si, si + c) for si, c in zip(ishift, copy_shape)])
        oslice = tuple([slice(so, so + c) for so, c in zip(oshift, copy_shape)])
    
        output = np.zeros(oshape1, dtype=input.dtype)
        input = input.reshape(ishape1)
        output[oslice] = input[islice]
    
        return output.reshape(oshape)

    @staticmethod
    def _expand_shapes(*shapes):
    
        shapes = [list(shape) for shape in shapes]
        max_ndim = max(len(shape) for shape in shapes)
        shapes_exp = [[1] * (max_ndim - len(shape)) + shape for shape in shapes]
    
        return tuple(shapes_exp)


#%% surface integral
class SurfaceIntegral:
    
    def __init__(self, size, sigma, shape="cuboid", dx=None):
        
        # set up dx
        if dx is None:
            dx = np.ones(3, dtype=np.float32)
        
        # calculate resolution
        self.prod_dx = np.prod(dx)
        
        # calculate kernels
        ds = np.array([dx[1]*dx[2], dx[0]*dx[2], dx[0]*dx[1]], dtype=np.float32)
        
        self.volume_kernel = self._volume_kernel(ds)

        self.surface_kernel = self._surface_kernel(ds)
        
        # calculate patch grid
        _, idx_offset, patch_mask, patch_edges = _get_local_mesh_grid(size, shape, order=1, dx=dx)
        
        self.idx_offset = idx_offset
        self.patch_mask = patch_mask
        self.patch_edges = patch_edges
        
        # set gaussian smoothing width
        self.sigma = sigma
        
    def _surface_kernel(self, ds):
        # z axis
        zkernel = np.zeros((3, 1, 1), dtype=np.float32)
        zkernel[0, 0, 0] = -ds[0]
        zkernel[2, 0, 0] = ds[0]
        
        # y axis
        ykernel = np.zeros((1, 3, 1), dtype=np.float32)
        ykernel[0, 0, 0] = -ds[1]
        ykernel[0, 2, 0] = ds[1]
        
        # x axis
        xkernel = np.zeros((1, 1, 3), dtype=np.float32)
        xkernel[0, 0, 0] = -ds[2]
        xkernel[0, 0, 2] = ds[2]
        
        return zkernel, ykernel, xkernel
        
    def _volume_kernel(self, ds):
        kernel = np.zeros((3, 3, 3), dtype=np.float32)
        kernel[[0, 2]] = ds[0] 
        kernel[:, [0, 2]] = ds[1] 
        kernel[:, :, [0, 2]] = ds[2]
        kernel = kernel / kernel.sum()
        return kernel
        
    def __call__(self, ind, phase_gradient, magnitude, seg_mask):
        
        # prepare output
        output = np.zeros(magnitude.shape, magnitude.dtype)
        
        # unpack
        idx_offset = self.idx_offset
        sigma = self.sigma
        patch_mask = self.patch_mask
        patch_edges = self.patch_edges
        prod_dx = self.prod_dx
        volume_kernel = self.volume_kernel
        surface_kernel = self.surface_kernel
        
        # actual integration
        for n in range(seg_mask.shape[0]):
            SurfaceIntegral._parallel_convolution(output, phase_gradient, magnitude, seg_mask[n], ind[n], idx_offset, sigma, patch_mask, patch_edges, prod_dx, volume_kernel, surface_kernel)
        
        return output
        
    @staticmethod
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _parallel_convolution(output, phase_gradient, magnitude, seg_mask, ind, idx_offset, sigma, patch_mask, patch_edges, prod_dx, volume_kernel, surface_kernel):
        
        # general fitting options
        nvoxels, _ = ind.shape
                                
        # loop over voxels
        for n in nb.prange(nvoxels):
            output[ind[n, 0], ind[n, 1], ind[n, 2]] = _integrate_within_patch(ind[n], idx_offset, phase_gradient, 
                                                                              magnitude, sigma, seg_mask, 
                                                                              patch_mask, patch_edges, 
                                                                              prod_dx, 
                                                                              volume_kernel, surface_kernel)
           
    
@nb.njit(cache=True, fastmath=True)
def _integrate_within_patch(idx, idx_offset, phase_gradient, magnitude, sigma, seg_mask, patch_mask, patch_edges, prod_dx, volume_kernel, surface_kernel):
    
    # get local phase
    local_phase_diff = [_get_local_patch(phase_diff, idx, idx_offset) for phase_diff in phase_gradient]
    
    # get local magnitude and voxel magnitude
    local_magnitude = _get_local_patch(magnitude, idx, idx_offset)
    
    # get local segmentation mask
    local_mask = _get_local_patch(seg_mask, idx, idx_offset)
    
    # get local magnitude        
    voxel_magnitude = magnitude[idx[0], idx[1], idx[2]]
    local_magnitude_diff = np.abs(local_magnitude - voxel_magnitude)
    
    # get local patch mask
    local_patch_mask = _get_local_patch_mask(local_mask, local_magnitude_diff, sigma, patch_mask, patch_edges)
    
    # get patch volume
    local_volume = _get_local_patch_volume(local_patch_mask, volume_kernel, prod_dx)
    
    # compute patch surface integral
    local_surface_integral = _get_local_patch_surface_integral(local_phase_diff, local_patch_mask, patch_mask, surface_kernel)
    
    return -local_surface_integral / local_volume


@nb.njit(cache=True, fastmath=True)
def _get_local_patch_mask(local_mask, local_magnitude_diff, sigma, patch_mask, patch_edges):
    
    # preserve input
    local_patch_mask = patch_mask.copy()
    idx = local_patch_mask > 0
    
    # form matrix
    local_magnitude_diff_tmp = np.zeros(local_patch_mask.shape, np.float32)
    local_mask_tmp = local_magnitude_diff_tmp.copy()
        
    _fill_matrix(local_magnitude_diff_tmp, local_magnitude_diff, idx)
    _fill_matrix(local_mask_tmp, local_mask, idx)
    
    # get local indexes    
    tmp = (patch_edges * (local_magnitude_diff_tmp > sigma).astype(np.float32)) + (patch_edges * (1 - local_mask_tmp))
    idx = tmp > 0
    
    # correct local mask
    _mask_matrix(local_patch_mask, idx)
    
    return local_patch_mask


@nb.njit(cache=True, fastmath=True)
def _fill_matrix(output, val, idx):
    nz, ny, nx = output.shape
    n = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if idx[z, y, x]:
                    output[z, y, x] = val[n]
                    n += 1
    
                    
@nb.njit(cache=True, fastmath=True)
def _mask_matrix(output, idx):
    nz, ny, nx = output.shape
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if idx[z, y, x]:
                    output[z, y, x] = 0.0
         
                
@nb.njit(cache=True, fastmath=True)
def _get_local_patch_volume(local_patch_mask, volume_kernel, prod_dx):
    
    # get volume
    volume = _conv3(local_patch_mask, volume_kernel) * local_patch_mask
    
    return volume.sum() * prod_dx


@nb.njit(cache=True, fastmath=True)
def _get_local_patch_surface_integral(local_phase_diff, local_patch_mask, global_patch_mask, surface_kernel):
    
    # get boolean indexes
    idx = global_patch_mask > 0
    
    # get z surface
    zsurface = _vectorize_matrix(_conv3(local_patch_mask, surface_kernel[0]) * local_patch_mask, idx)
    
    # get y surface
    ysurface = _vectorize_matrix(_conv3(local_patch_mask, surface_kernel[1]) * local_patch_mask, idx)
    
    # get x surface
    xsurface = _vectorize_matrix(_conv3(local_patch_mask, surface_kernel[2]) * local_patch_mask, idx)
    
    # get surface integral
    surface_integral = local_phase_diff[0] * zsurface + local_phase_diff[1] * ysurface + local_phase_diff[2] * xsurface
    
    return surface_integral.sum()


@nb.njit(cache=True, fastmath=True)
def _vectorize_matrix(input, idx):
    output = np.zeros(idx.sum(), input.dtype)
    nz, ny, nx = input.shape
    n = 0
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                if idx[z, y, x]:
                    output[n] = input[z, y, x]
                    n += 1
    
    return output


@nb.njit(cache=True, fastmath=True)
def _conv3(input, kernel):
    """
    Numba-friendly 3d-convolution.        
    """
    # flip kernel
    kernel = np.flip(kernel)
    
    # get info
    nz, ny, nx = input.shape
    depth, width, height = kernel.shape
    
    # prepare output
    output = np.zeros(input.shape, dtype=input.dtype)
            
    # pad
    padded_input = np.zeros((nz + depth, ny + width, nx + height), dtype=input.dtype)
    padded_input[depth//2:-depth//2, width//2:-width//2, height//2:-height//2 ] = input
            
    # actual convolution
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                for dd in range(depth):
                    for ww in range(width):
                        for hh in range(height):
                            inputval = padded_input[z + dd, y + ww, x + hh]
                            kernelval = kernel[dd, ww, hh]
                            output[z, y, x] += inputval * kernelval
                                    
    return np.flip(output)
       
#%% derivative kernel generation
class LocalDerivative:
    
    def __init__(self, size, sigma, shape="cuboid", dx=None, order=1):
        
        # set up dx
        if dx is None:
            dx = np.ones(3, dtype=np.float32)
        
        # calculate patch grid
        grid, idx_offset, _, _ = _get_local_mesh_grid(size, shape, order=order, dx=dx)
        
        self.grid = grid
        self.idx_offset = idx_offset
        
        # set gaussian smoothing width
        self.sigma = sigma
        
        # choose differentiation
        if shape == 'cross':
            self._differentiate = LocalDerivative._differentiate_within_cross
        else:
            self._differentiate = LocalDerivative._differentiate_within_patch
            
    def __call__(self, ind, phase, magnitude, seg_mask):
        
        # prepare output
        output = np.zeros(list(magnitude.shape) + [3], magnitude.dtype)
        
        # unpack
        grid = self.grid
        idx_offset = self.idx_offset
        sigma = self.sigma
        _differentiate = self._differentiate
        
        # actual integration
        for n in range(seg_mask.shape[0]):
            LocalDerivative._parallel_convolution(output, phase, magnitude, seg_mask[n], ind[n], idx_offset, sigma, grid, _differentiate)
        
        return np.ascontiguousarray(output.transpose(-1, 0, 1, 2))
      
    @staticmethod
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _parallel_convolution(output, phase, magnitude, seg_mask, ind, idx_offset, sigma, grid, _differentiate):
        
        # general fitting options
        nvoxels, _ = ind.shape
                                      
        # loop over voxels
        for n in nb.prange(nvoxels):
            tmp = _differentiate(ind[n], idx_offset, phase, magnitude, sigma, seg_mask, grid)
            output[ind[n, 0], ind[n, 1], ind[n, 2], :] = tmp
      
    @staticmethod
    @nb.njit(cache=True, fastmath=True)
    def _differentiate_within_cross(idx, idx_offset, phase, magnitude, sigma, seg_mask, grid):

        # get local phase
        local_phase = _get_local_cross_patch(phase, idx, idx_offset)
        
        # get local magnitude and voxel magnitude
        local_magnitude = _get_local_cross_patch(magnitude, idx, idx_offset)
                
        # get local segmentation mask
        local_mask = _get_local_cross_patch(seg_mask, idx, idx_offset)
        
        # get local magnitude        
        voxel_magnitude = magnitude[idx[0], idx[1], idx[2]]
        
        # prepare out
        coeffs = []
                
        # loop over axis
        for ax in range(3):     
            # get weight
            weights = local_magnitude[ax] - voxel_magnitude
            weights = np.exp(- weights**2 / (2 * sigma**2))
            weights *= local_mask[ax]
            
            # do fit
            a = grid[ax] * np.expand_dims(weights, -1)
            b = local_phase[ax] * weights
            try:
                coeffs.append(_lstsq(a, b)[0])          
            except:
                coeffs.append(0.0)
        
        return np.array(coeffs, dtype=phase.dtype)
    
    @staticmethod
    @nb.njit(cache=True, fastmath=True)
    def _differentiate_within_patch(idx, idx_offset, phase, magnitude, sigma, seg_mask, grid):
        
        # get local phase
        local_phase = _get_local_patch(phase, idx, idx_offset)
        
        # get local magnitude and voxel magnitude
        local_magnitude = _get_local_patch(magnitude, idx, idx_offset)
        
        # get local segmentation mask
        local_mask = _get_local_patch(seg_mask, idx, idx_offset)
        
        # get weight        
        voxel_magnitude = magnitude[idx[0], idx[1], idx[2]]
        weights = local_magnitude - voxel_magnitude
        weights = np.exp(- weights**2 / (2 * sigma**2))
        weights *= local_mask
                
        # do fit
        a = grid * np.expand_dims(weights, -1)
        b = local_phase * weights
        try:
            coeffs = _lstsq(a, b)[:3]
        except:
            coeffs = np.zeros(3, b.dtype)
        
        return coeffs

#%% common utils for sliding integration and differentiation
@nb.njit(cache=True, fastmath=True)
def _get_local_cross_patch(img, idx, idx_offset):
    
    # prepare output value
    value = []
    
    # get patch center
    z, y, x = idx
    
    # z-axis
    patch_size = len(idx_offset[0])
    val = np.zeros(patch_size, dtype=img.dtype)
    
    # populate patch
    for n in range(patch_size):
        dz = idx_offset[0][n]
        val[n] = img[z + dz, y, x]
    
    # append
    value.append(val)
    
    # y-axis
    patch_size = len(idx_offset[1])
    val = np.zeros(patch_size, dtype=img.dtype)
    
    # populate patch
    for n in range(patch_size):
        dy = idx_offset[1][n]
        val[n] = img[z, y + dy, x]
    
    # append
    value.append(val)
    
    # x-axis
    patch_size = len(idx_offset[2])
    val = np.zeros(patch_size, dtype=img.dtype)
    
    # populate patch
    for n in range(patch_size):
        dx = idx_offset[2][n]
        val[n] = img[z, y, x + dx]
    
    # append
    value.append(val)
    
    return value


@nb.njit(cache=True, fastmath=True)
def _get_local_patch(img, idx, idx_offset):
    
    patch_size = len(idx_offset[0])
    value = np.zeros(patch_size, dtype=img.dtype)
    
    # get patch center
    z, y, x = idx
    
    # populate patch
    for n in range(patch_size):
        dx = idx_offset[2][n]
        dy = idx_offset[1][n]
        dz = idx_offset[0][n]
        value[n] = img[z + dz, y + dy, x + dx]
        
    return value


@nb.njit(cache=True, fastmath=True)
def _lstsq(a, b):
    return np.linalg.solve(np.dot(a.T, a), np.dot(a.T, b))


def _get_local_mesh_grid(size, shape='cuboid', order=2, dx=1):
    """
    Get relative coordinates within a patch.
    
    Args:
        size (scalar, tuple-like): size of the patch. If scalar, assume isotropic patch.
        shape (str): can be "cross", "cuboid" or "ellipse".
        order (int): if 1, calculate gradient; if 2, calculate laplacian.
        
    Returns:
        grid (tuple-of-ndarray-of-float): grid coordinates for a 1-degree polynomial (order = 1) or 2-degree polynomial (order = 2).
        idx_offset (tuple-of-ndarray-of-int): offset wrt the patch center.
    """
    # check if size is scalar
    if isinstance(size, (np.ndarray, list, tuple)) is False:
        size = np.array(3 * [size], dtype=np.float32)
    else:
        size = np.asarray(size, dtype=np.float32)
        
    # check isotropic kernel
    assert len(np.unique(size)) == 1, "Anisotropic Kernel allowed for cross-shaped  and cuboid kernels only!"
        
    axes = [np.flip(-np.arange(-ax // 2 + 1, ax // 2 + 1, dtype=np.float32)) for ax in size]
    idx_offset = [ax.astype(np.int16) for ax in axes]
            
    if shape == 'cross':
        # rescale axes to physical units
        axes = [dx[ax] * axes[ax] for ax in range(3)]
        
        # prepare coordinates
        x, y, z = axes
        xones, yones, zones = np.ones(x.shape, x.dtype), np.ones(y.shape, y.dtype), np.ones(z.shape, z.dtype)
        
        if order == 1:              
            grid = [np.stack((z, zones), axis=-1), np.stack((y, yones), axis=-1), np.stack((x, xones), axis=-1)]
            return grid, idx_offset, None, None
        
        if order == 2:
            x2, y2, z2 =  x**2, y**2, z**2            
            grid = [np.stack((z2, z, zones), axis=-1), np.stack((y2, y, yones), axis=-1), np.stack((x2, x, xones), axis=-1)]
            return grid, idx_offset, None, None
                
    # build cubic grid    
    yy, zz, xx = np.meshgrid(*axes, indexing='xy')
    yy = np.flip(yy, axis=1)
    axes = [xx, yy, zz]
    
    # remove corners
    if shape == 'ellipsoid':
        axes = np.stack(axes, axis=0)
        rr = (axes**2).sum(axis=0)**0.5
        todo = rr <= size[0] // 2
        axes = [ax[todo] for ax in axes]
        
    # get edges
    if shape == 'ellipsoid':
        structure = ndimage.morphology.generate_binary_structure(3, 26)
        patch_mask = todo.astype(np.float32)
        edges = todo.astype(np.float32) - ndimage.binary_erosion(patch_mask, structure=structure)
    elif shape == 'cuboid':
        patch_mask = np.ones(xx.shape, dtype=np.float32)
        edges = np.ones(xx.shape, dtype=np.float32)
        edges[1:-1,1:-1,1:-1] = 0
        
    # flatten axes
    axes = [ax.flatten() for ax in axes]
    idx_offset = [ax.astype(np.int16) for ax in axes] 
    
    # rescale axes to physical units
    axes = [dx[ax] * axes[ax] for ax in range(3)]
    
    # unpack axes
    x, y, z = axes

    if order == 1:
        grid = np.stack([z, y, x, np.ones(x.shape, x.dtype)], axis=-1)
        return grid, idx_offset, patch_mask, edges
    if order == 2:
        x2, y2, z2 = x**2, y**2, z**2
        xy, yz, xz = x * y, y * z, x * z        
        grid = np.stack([z2, y2, x2, z, y, x, xy, yz, xz, np.ones(x.shape, x.dtype)], axis=-1)
        
        return grid, idx_offset, None, None
        

# post processing
def _adaptive_median_filter(input, segmentation, filter_width, filter_shape='cuboid'):
            
    # prepare data
    data_prep = DataReformat(filter_width)
    input, _, segmentation, ind = data_prep.prepare_data(input, np.ones(input.shape, input.dtype), segmentation)
    
    # do laplacian
    filter_operator = PostProcessing(filter_width, filter_shape)
    output = filter_operator(ind, input, segmentation)
    
    # crop
    return data_prep.reformat_data(output)

class PostProcessing:
    
    def __init__(self, size, shape="cuboid"):
                
        # calculate patch grid
        _, idx_offset, _, _ = _get_local_mesh_grid(size, shape)        
        self.idx_offset = idx_offset
                  
    def __apply__(self, ind, phase, seg_mask):
        
        # prepare output
        output = np.zeros(phase.shape, phase.dtype)
        
        # unpack
        idx_offset = self.idx_offset
       
        # actual integration
        for n in range(seg_mask.shape[0]):
            PostProcessing._parallel_convolution(output, phase, seg_mask, ind, idx_offset)
        
        return output
      
    @staticmethod
    @nb.njit(fastmath=True, parallel=True)
    def _parallel_convolution(output, phase, seg_mask, ind, idx_offset):
        
        # general fitting options
        nvoxels, _ = ind.shape
                                
        # loop over voxels
        for n in nb.prange(nvoxels):
            output[ind[n, 0], ind[n, 1], ind[n, 2]] = _local_median(ind[n], idx_offset, seg_mask)
    
    
@nb.njit(fastmath=True)
def _local_median(idx, idx_offset, phase, seg_mask):
    
    # get local phase
    local_phase = _get_local_patch(phase, idx, idx_offset)
            
    # get local segmentation mask
    local_mask = _get_local_patch(seg_mask, idx, idx_offset)
    
    # get weight   
    local_phase = local_phase[local_mask]

    return np.median(local_phase)
    

        

        

        




