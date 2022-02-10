# -*- coding: utf-8 -*-
"""
DICOM files reading routines.

Created on Thu Feb 10 16:16:41 2022

@author: Matteo Cencini
"""
from typing import Union, Tuple, List, Dict


import numpy as np


from qmrpy.src.io import utils


__all__ = ['read_dicom']


def read_dicom(dicomdir: Union[str, List, Tuple]) -> Tuple[np.ndarray, Dict]:
    """
    Load multi-contrast images for parameter mapping.
    
    Args:
        dicomdir: string or list of strings with DICOM folders path.
        
    Returns:
        image: ndarray of sorted image data.
        info: dict with the following fields:
            - template: the DICOM template.
            - slice_locations: ndarray of Slice Locations [mm].
            - TI: ndarray of Inversion Times [ms].
            - TE: ndarray of Echo Times [ms].
            - TR: ndarray of Repetition Times [ms].
            - FA: ndarray of Flip Angles [deg].
    """
    # load dicom
    dsets = utils._load_dcm(dicomdir)
        
    # get slice locations
    uSliceLocs, firstSliceIdx, sliceIdx = utils._get_slice_locations(dsets)
        
    # get echo times
    inversionTimes = utils._get_inversion_times(dsets)
    
    # get echo times
    echoTimes = utils._get_echo_times(dsets)
    
    # get repetition times
    repetitionTimes = utils._get_repetition_times(dsets)
    
    # get flip angles
    flipAngles = utils._get_flip_angles(dsets)
    
    # get sequence matrix
    contrasts = np.stack((inversionTimes, echoTimes, repetitionTimes, flipAngles), axis=1)
    
    # get unique contrast and indexes
    uContrasts, contrastIdx = utils._get_unique_contrasts(contrasts)
        
    # extract image tensor
    image = np.stack([dset.pixel_array for dset in dsets], axis=0)
    
    # get size
    n_slices = len(uSliceLocs)
    n_contrasts = uContrasts.shape[0]
    ninstances, ny, nx = image.shape
    
    # fill sorted image tensor
    sorted_image = np.zeros((n_contrasts, n_slices, ny, nx), dtype=np.float32)
    for n in range(ninstances):
        sorted_image[contrastIdx[n], sliceIdx[n], :, :] = image[n]
        
    # get dicom template
    dcm_template = utils._get_dicom_template(dsets, firstSliceIdx)
    
    # unpack sequence
    TI, TE, TR, FA = uContrasts.transpose()
    
    # squeeze
    if sorted_image.shape[0] == 1:
        sorted_image = sorted_image[0]
        
    return sorted_image, {'template': dcm_template, 'TI': TI, 'TE': TE, 'TR': TR, 'FA': FA}
