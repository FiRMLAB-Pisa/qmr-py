# -*- coding: utf-8 -*-
"""
Fitting routines for field mapping.

Created on Mon Feb  7 14:57:00 2022

@author: Matteo Cencini
"""
import numpy as np


from skimage.restoration import unwrap_phase as unwrap


__all__ = ['b1_dam_fitting', 'b0_multiecho_fitting']


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
    
    # actual calculation (units: [%])
    b1map = 100 * np.rad2deg(np.arccos(cos_flip)) / min_flip
    
    # final cleanup
    b1map = np.nan_to_num(b1map)
    
    # mask
    if mask is not None:
        b1map = mask * b1map
        
    return b1map


def b0_multiecho_fitting(input: np.ndarray, te: float, mask: np.ndarray = None, unipolar_echoes = None, fft_shift_along_z=True) -> np.ndarray:
    
    # preserve input
    input = input.copy()
    
    if unipolar_echoes is not None:
        input = input[unipolar_echoes::2]
        te = te[unipolar_echoes::2]
    
    # fix fftshift
    if fft_shift_along_z is True:
        input = np.fft.ifft(np.fft.fftshift(np.fft.fft(input, axis=1), axes=(1)), axis=1)
        
    # get phase
    phase = np.angle(input)
    
    # mask
    if mask is not None:
        phase = mask * phase
    
    # get number of echoes
    nechoes = len(te)
    
    # uwrap across space
    for n in range(nechoes):
        true_phase = phase[n, phase.shape[1] // 2, phase.shape[2] // 2, phase.shape[3] // 2]
        phase[n] = mask * unwrap(phase[n])
        phase[n] = phase[n] - phase[n, phase.shape[0] // 2, phase.shape[1] // 2, phase.shape[2] // 2] + true_phase
    
    # unwrap across echoes
    true_phase = phase[0]
    phase = np.unwrap(phase, axis=0)
    phase = phase - phase[0] + true_phase
    
    # clean phase
    phase = np.nan_to_num(mask * phase).astype(np.float32)
    
    # fit linear phase
    x = np.stack([te / 1000, np.ones(te.shape, te.dtype)], axis=-1)
    y = phase[:, mask]
    
    # do fit
    p = np.linalg.lstsq(x, y, rcond=None)[0]
    
    # build output
    b0map = np.zeros(phase[0].shape, phase.dtype)
    b0map[mask] = p[0] / 2 / np.pi
    
    b1phase = np.zeros(phase[0].shape, phase.dtype)
    b1phase[mask] = p[1]
    
    return b0map, b1phase


