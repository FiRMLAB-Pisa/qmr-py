# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:54:03 2022

@author: mcencini
"""
import warnings


import numpy as np
import numba as nb
import matplotlib.pyplot as plt


from scipy import ndimage


from numba import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from scipy.ndimage import gaussian_filter


from skimage.restoration import unwrap_phase as unwrap


from qmrpy.src.inference import utils
from qmrpy.src.inference.field_mapping import b0_multiecho_fitting


# vacuum permeability
mu0 = 4.0e-7 * np.pi # [H/m]


warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


__all__ = ['PhaseBasedLaplacianEPT']


def PhaseBasedLaplacianEPT(input: np.ndarray, resolution: np.ndarray, omega0: float,
                           gaussian_preprocessing_sigma: float = 0.0, gaussian_weight_sigma: float = 0.45,
                           kernel_size: int = 16, kernel_shape='cross', 
                           median_filter_width: int = 0, segmentation_mask: np.ndarray = None,
                           te: np.ndarray = None, fft_shift_along_z: bool = True, local_mask_threshold: float = np.inf) -> np.ndarray:
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
        kernel_size = [kernel_size]
        gaussian_weight_sigma = [gaussian_weight_sigma]
        median_filter_width = [median_filter_width]
   
    # get mask
    mask = segmentation_mask.sum(axis=0)
              
    # preprocess input        
    if gaussian_preprocessing_sigma is not None and gaussian_preprocessing_sigma > 0:
        input = gaussian_filter(input.real, gaussian_preprocessing_sigma) + 1j * gaussian_filter(input.imag, gaussian_preprocessing_sigma)
    
    # get magnitude and phase
    if len(input.shape) == 4:
        magnitude = mask * np.abs(input).sum(axis=0)
    else:
        magnitude = mask * np.abs(input)
        
    mask = mask.astype(bool)
    
    # Normalise magnitude
    magnitude -= magnitude[mask].mean()
    if magnitude[mask].std() != 0: 
        magnitude /=  magnitude[mask].std()
    
    magnitude[np.invert(mask)] = 100
    
    # prepare phase
    if len(input.shape) == 4:
        _, phase = b0_multiecho_fitting(input, te, mask, fft_shift_along_z)
    else:
        phase = np.angle(input)

    # uwrap
    true_phase = phase[phase.shape[0] // 2, phase.shape[1] // 2, phase.shape[2] // 2]
    phase = mask * unwrap(phase)
    phase = phase - phase[phase.shape[0] // 2, phase.shape[1] // 2, phase.shape[2] // 2] + true_phase
    
    # transceive phase approximation
    phase *= -1
    
    # clean phase
    phase = np.nan_to_num(phase).astype(np.float32)
    
    # # remove B0 component
    linear_phase = _get_linear_component(mask, phase.copy())
    phase = phase - linear_phase

    # actual computation
    if len(segmentation_mask.shape) == 3:
        conductivity, laplacian = _PhaseBasedLaplacianEPT(phase, magnitude, segmentation_mask, omega0, resolution, kernel_size, kernel_shape, gaussian_weight_sigma, local_mask_threshold)
        
        # post process
        if median_filter_width is not None and median_filter_width > 0:
            conductivity = _adaptive_median_filter(conductivity, magnitude, segmentation_mask.copy(), median_filter_width, kernel_shape, local_mask_threshold)
            
    else:
        conductivity = []
        laplacian = []
        for n in range(segmentation_mask.shape[0]):
            conductivity_tmp, laplacian_tmp = _PhaseBasedLaplacianEPT(phase, magnitude, segmentation_mask[[n]].copy(), omega0, resolution, kernel_size[n], kernel_shape, gaussian_weight_sigma[n], local_mask_threshold)
            
            # post process
            if median_filter_width[n] is not None and median_filter_width[n] > 0:
                conductivity_tmp = _adaptive_median_filter(conductivity_tmp, magnitude, segmentation_mask[[n]].copy(), median_filter_width[n], kernel_shape, local_mask_threshold)
                
            conductivity.append(conductivity_tmp)
            laplacian.append(laplacian_tmp)
        
        # merge regopms
        conductivity = np.stack(conductivity, axis=0).sum(axis=0)
        laplacian = np.stack(laplacian, axis=0).sum(axis=0)
                
    return conductivity, phase, laplacian


def _PhaseBasedLaplacianEPT(phase, magnitude, segmentation, omega0, resolution=1, kernel_size=26, kernel_shape='cross', gauss_sigma=0.45, local_mask_threshold=np.inf):
    
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
    laplacian = laplacian_operator(ind, phase, magnitude, segmentation, local_mask_threshold)
    
    # calculate conductivity
    conductivity = laplacian[:3].sum(axis=0) / omega0 / mu0
        
    # crop
    laplacian = np.stack([data_prep.reformat_data(term) for term in laplacian], axis=0)
    conductivity = data_prep.reformat_data(conductivity)
        
    return conductivity, laplacian
  
        
#%% plot utils
def show(input, slices, vmin=0, vmax=1, cmap='jet'):
    
    # pad as cubic matrix
    oshape = 3 * [max(input[0].shape)]
    input = [DataReformat._resize(img, oshape) for img in input]
    
    # create matrix
    out = [np.concatenate((img[slices[0]], np.flip(img[:, slices[1]]), np.flip(img[:, :, slices[2]])), axis=-1) for img in input]
    out = np.concatenate(out, axis=0)
    
    # plot
    plt.imshow(out, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.axis('off')
    plt.colorbar()
                                                                                         
                                                                                                                                                                 
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
       
#%% derivative kernel generation
class LocalDerivative:
    
    def __init__(self, size, sigma, shape="cuboid", dx=None, order=1):
                
        # set up dx
        if dx is None:
            dx = np.ones(10, dtype=np.float32)
        
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
            
    def __call__(self, ind, phase, magnitude, seg_mask, local_mask_threshold):
        
        # prepare output
        output = np.zeros(list(magnitude.shape) + [10], magnitude.dtype)
        
        # unpack
        grid = self.grid
        idx_offset = self.idx_offset
        sigma = self.sigma
        _differentiate = self._differentiate
        
        # actual integration
        for n in range(seg_mask.shape[0]):
            LocalDerivative._parallel_convolution(output, phase, magnitude, seg_mask[n], ind[n], idx_offset, sigma, grid, _differentiate, local_mask_threshold)
        
        return np.ascontiguousarray(output.transpose(-1, 0, 1, 2))
      
    @staticmethod
    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _parallel_convolution(output, phase, magnitude, seg_mask, ind, idx_offset, sigma, grid, _differentiate, local_mask_threshold):
        
        # general fitting options
        nvoxels, _ = ind.shape
                                      
        # loop over voxels
        for n in nb.prange(nvoxels):
            tmp = _differentiate(ind[n], idx_offset, phase, magnitude, sigma, seg_mask, grid, local_mask_threshold)
            output[ind[n, 0], ind[n, 1], ind[n, 2], :] = tmp
      
    @staticmethod
    @nb.njit(cache=True, fastmath=True)
    def _differentiate_within_cross(idx, idx_offset, phase, magnitude, sigma, seg_mask, grid, local_mask_threshold):

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
    def _differentiate_within_patch(idx, idx_offset, phase, magnitude, sigma, seg_mask, grid, local_mask_threshold):
        
        # get local phase
        local_phase = _get_local_patch(phase, idx, idx_offset)
        
        # get local magnitude and voxel magnitude
        local_magnitude = _get_local_patch(magnitude, idx, idx_offset)
        
        # get local segmentation mask
        local_mask = _get_local_patch(seg_mask, idx, idx_offset)
        
        # get magnitude difference        
        voxel_magnitude = magnitude[idx[0], idx[1], idx[2]]
        magnitude_diff = local_magnitude - voxel_magnitude
        
        # get weight  
        weights = np.exp(- magnitude_diff**2 / (2 * sigma**2))
        weights *= local_mask
        
        # remove voxels with local magnitude difference >= 20%
        weights *= (np.abs(magnitude_diff / voxel_magnitude) <= local_mask_threshold)
                
        # do fit
        a = grid * np.expand_dims(weights, -1)
        b = local_phase * weights
        try:
            coeffs = _lstsq(a, b)#[:3]
        except:
            coeffs = np.zeros(10, b.dtype)
        
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


def _get_linear_component(mask, input):
    
    # get gridsize
    gridsize = input.shape
        
    # build grid
    axes = [np.flip(-np.arange(-ax // 2 + 1, ax // 2 + 1, dtype=np.float32)) for ax in gridsize]
            
    # build cubic grid    
    zz, yy, xx = np.meshgrid(*axes, indexing='ij')
    yy = np.flip(yy, axis=1)
    
    # flatten 
    xx, yy, zz = xx[mask], yy[mask], zz[mask]
    axes = np.stack([zz, yy, xx, np.ones(zz.shape, zz.dtype)], axis=-1)
        
    # fit phase
    coeff = np.linalg.lstsq(axes, input[mask], rcond=None)[0]

    # # # reshape
    out = np.zeros(input.shape, input.dtype)
    out[mask] =  (coeff * axes).sum(axis=-1)
        
    return out


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
        
    # check if size is scalar
    if isinstance(dx, (np.ndarray, list, tuple)) is False:
        dx = np.array(3 * [dx], dtype=np.float32)
    else:
        dx = np.asarray(dx, dtype=np.float32)
        
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
    x2, y2, z2 = x**2, y**2, z**2
    xy, yz, xz = x * y, y * z, x * z  

    if order == 1:
        grid = np.stack([z, y, x, np.ones(x.shape, x.dtype)], axis=-1)
        return grid, idx_offset, patch_mask, edges
    if order == 2:            
        grid = np.stack([z2, y2, x2, z, y, x, xy, yz, xz, np.ones(x.shape, x.dtype)], axis=-1)

        return grid, idx_offset, None, None
        

# post processing
def _adaptive_median_filter(conductivity, magnitude, segmentation, filter_width, filter_shape='cuboid', local_mask_threshold=np.inf):
                
    # prepare data
    data_prep = DataReformat(filter_width)
    conductivity, magnitude, segmentation, ind = data_prep.prepare_data(conductivity, np.ones(conductivity.shape, conductivity.dtype), segmentation)
    
    # do laplacian
    filter_operator = PostProcessing(filter_width, filter_shape)
    output = filter_operator(ind, conductivity, magnitude, segmentation, local_mask_threshold)
    
    # crop
    return data_prep.reformat_data(output)


class PostProcessing:
    
    def __init__(self, size, shape="cuboid"):
                
        # calculate patch grid
        _, idx_offset, _, _ = _get_local_mesh_grid(size, shape)        
        self.idx_offset = idx_offset
                  
    def __call__(self, ind, conductivity, magnitude, seg_mask, local_mask_threshold):
        
        # prepare output
        output = np.zeros(conductivity.shape, conductivity.dtype)
        
        # unpack
        idx_offset = self.idx_offset
               
        # actual integration
        for n in range(seg_mask.shape[0]):
            PostProcessing._parallel_convolution(output, conductivity, magnitude, seg_mask[n], ind[n], idx_offset, local_mask_threshold)
        
        return output
      
    @staticmethod
    @nb.njit(fastmath=True, parallel=True)
    def _parallel_convolution(output, conductivity, magnitude, seg_mask, ind, idx_offset, local_mask_threshold):
                
        # general fitting options
        nvoxels, _ = ind.shape
        
        # loop over voxels
        for n in nb.prange(nvoxels):
            output[ind[n, 0], ind[n, 1], ind[n, 2]] = _local_median(ind[n], idx_offset, conductivity, magnitude, seg_mask, local_mask_threshold)
    
    
@nb.njit(fastmath=True)
def _local_median(idx, idx_offset, conductivity, magnitude, seg_mask, local_mask_threshold):
    
    # get local conductivity
    local_conductivity = _get_local_patch(conductivity, idx, idx_offset)
    
    # get local magnitude and voxel magnitude
    local_magnitude = _get_local_patch(magnitude, idx, idx_offset)
        
    # get magnitude difference        
    voxel_magnitude = magnitude[idx[0], idx[1], idx[2]]
    magnitude_diff = local_magnitude - voxel_magnitude
                    
    # get local segmentation mask
    local_mask = _get_local_patch(seg_mask, idx, idx_offset)
    
    # remove voxels with local magnitude difference >= 20%
    local_mask *= (np.abs(magnitude_diff / voxel_magnitude) <= local_mask_threshold)
    
    # get conductivity within mask   
    local_conductivity = local_conductivity[local_mask]

    return np.median(local_conductivity)
