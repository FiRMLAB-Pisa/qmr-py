# -*- coding: utf-8 -*-
"""
Fitting routines for field mapping.

Created on Mon Feb  7 14:57:00 2022

@author: Matteo Cencini
"""
import numpy as np


__all__ = ['b1_dam_fitting']


def b1_dam_fitting(input: np.ndarray, fa: float, mask: np.ndarray = None) -> np.ndarray:
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
