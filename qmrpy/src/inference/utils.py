# -*- coding: utf-8 -*-
"""
Utility routines for parameter mapping.

Created on Thu Feb 10 16:33:59 2022

@author: Matteo Cencini
"""
import numpy as np
from scipy import ndimage


__all__ = ['mask']


def mask(input: np.ndarray, threshold: float = 0.05) -> np.ndarray:
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

