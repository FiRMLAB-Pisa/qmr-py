# -*- coding: utf-8 -*-
"""
DICOM files reading routines.

Created on Thu Feb 10 16:16:41 2022

@author: Matteo Cencini
"""
import os
import json
import pathlib

from typing import Union, Tuple, List, Dict


import numpy as np


import nibabel as nib


from qmrpy.src.io import utils


__all__ = ['read_data', 'read_segmentation']


def read_data(paths: Union[str, List, Tuple]) -> Tuple[np.ndarray, Dict]:
    """
    Load multi-contrast images for parameter mapping.
    
    Args:
        paths: string or list of strings with image files path.
        
    Returns:
        image: ndarray of sorted image data.
        info: dict with the following fields:
            - nifti_template: the NIFTI template (might be empty).
            - dcm_template: the DICOM template (might be empty).
            - TI: ndarray of Inversion Times [ms].
            - TE: ndarray of Echo Times [ms].
            - TR: ndarray of Repetition Times [ms].
            - FA: ndarray of Flip Angles [deg].
    """
    try:
        return read_dicom(paths)
    except:
        return read_nifti(paths)


def read_segmentation(mask_path: str) -> np.ndarray:
    """
    Load segmentation for better reconstruction conditioning.
    """
    # get full path
    mask_path = os.path.normpath(mask_path)
    
    # split path
    mask_path_parts = mask_path.split(os.sep)
    mask_path = []
    
    # create paths for gm, wm, csf
    for n in range(3):
        filename = 'c' + str(n+1) + mask_path_parts[-1]
        mask_path.append(os.path.join(*mask_path_parts[:-1], filename))
        
    return read_nifti(mask_path)
    
    
def read_nifti(nifti_files: Union[str, List, Tuple]) -> Tuple[np.ndarray, Dict]:
    """
    Load multi-contrast images for parameter mapping.
    
    Args:
        nifti_files: string or list of strings with NIFTI files path.
        
    Returns:
        image: ndarray of sorted image data.
        info: dict with the following fields:
            - nifti_template: the NIFTI template.
            - TI: ndarray of Inversion Times [ms].
            - TE: ndarray of Echo Times [ms].
            - TR: ndarray of Repetition Times [ms].
            - FA: ndarray of Flip Angles [deg].
    """
    if isinstance(nifti_files, (list, tuple)):
        
        # get file path
        nifti_files = [os.path.normpath(file) for file in nifti_files 
                       if file.endswith('.nii') or file.endswith('.nii.gz')]
        file_path = nifti_files
        
        # check for complex images
        try:        
            # magnitude
            try: 
                idx = np.argwhere(np.array(['mag' in name for name in nifti_files])).squeeze()
                img_mag = nib.load(nifti_files[idx])
            except:
                img_mag = None
             
            # phase
            try: 
                idx = np.argwhere(np.array(['phase' in name for name in nifti_files])).squeeze()
                img_phase = nib.load(nifti_files[idx])
            except:
                img_phase = None
            
            # real
            try: 
                idx = np.argwhere(np.array(['real' in name for name in nifti_files])).squeeze()
                img_real = nib.load(nifti_files[idx])
            except:
                img_real = None
             
            # imaginary
            try: 
                idx = np.argwhere(np.array(['imag' in name for name in nifti_files])).squeeze()
                img_imag = nib.load(nifti_files[idx])
            except:
                img_imag = None
                
            # assemble image
            if img_mag is not None and img_phase is not None:
                data = img_mag.get_fdata() * np.exp(1j * img_phase.get_fdata())
                affine = img_mag.affine
                header = img_mag.header
                
            if img_real is not None and img_imag is not None:
                data = img_real.get_fdata() + 1j * img_imag.get_fdata()
                affine = img_real.affine
                header = img_real.header
                
            # correct phase shift along z
            data = np.flip(np.fft.fft(np.fft.fftshift(np.fft.fft(data, axis=-1), axes=-1), axis=-1), axis=-1)

        except:
            img = [nib.load(file) for file in nifti_files]
            data = np.stack([d.get_fdata() for d in img], axis=-1)
            affine = img[0].affine
            header = img[0].header
            
    else:
        file_path = [os.path.normpath(nifti_files)]
        img = nib.load(file_path[0])
        data = img.get_fdata()
        affine = img.affine
        header = img.header
        
    data = np.flip(data.transpose(), axis=-2)
    
    # get json
    try:
        root = os.path.dirname(file_path[0])
        json_paths = [os.path.normpath(str(path)) for path in sorted(pathlib.Path(root).glob('*.json'))]
        
        # init fields
        B0 = []
        TI = []
        EC = []
        TE = []
        TR = []
        FA = []
        
        # iterate over json files
        for json_path in json_paths:
            with open(json_path) as json_file:
                json_dict = json.loads(json_file.read())
             
            B0 = json_dict['MagneticFieldStrength']
            
            # get parameters
            if 'InversionTime' in json_dict:
                TI.append(1e3 * json_dict['InversionTime'])
            else:
                TI.append(np.Inf)
                
            TE.append(1e3 * json_dict['EchoTime'])
            EC.append(json_dict['EchoNumber'])
            TR.append(1e3 * json_dict['RepetitionTime'])
            FA.append(json_dict['FlipAngle'])
            
        # filter repeated entries
        B0 = np.unique(B0)
        TI = np.unique(TI)
        EC = np.unique(EC)
        TE = np.unique(TE)
        TR = np.unique(TR)
        FA = np.unique(FA)
        
        # get scalars
        if len(B0) == 1:
            B0 = B0[0]
        if len(TI) == 1:
            TI = TI[0]
        if len(EC) == 1:
            EC = EC[0]
        if len(TE) == 1:
            TE = TE[0]
        if len(TR) == 1:
            TR = TR[0]
        if len(FA) == 1:
            FA = FA[0]
                                
        return data, {'nifti_template': {'affine': affine, 'header': header, 'json': json_dict}, 'dcm_template': {}, 'B0': B0, 'EC': EC, 'TI': TI, 'TE': TE, 'TR': TR, 'FA': FA}
    
    except:
        return data, {'nifti_template': {'affine': affine, 'header': header, 'json': {}}, 'dcm_template': {}, 'B0': None, 'EC': None, 'TI': None, 'TE': None, 'TR': None, 'FA': None}
        
    
def read_dicom(dicomdir: Union[str, List, Tuple]) -> Tuple[np.ndarray, Dict]:
    """
    Load multi-contrast images for parameter mapping.
    
    Args:
        dicomdir: string or list of strings with DICOM folders path.
        
    Returns:
        image: ndarray of sorted image data.
        info: dict with the following fields:
            - dcm_template: the DICOM template.
            - TI: ndarray of Inversion Times [ms].
            - TE: ndarray of Echo Times [ms].
            - TR: ndarray of Repetition Times [ms].
            - FA: ndarray of Flip Angles [deg].
    """
    # load dicom
    image, dsets = utils._load_dcm(dicomdir)
    
    # get field strength
    B0 = float(dsets[0].MagneticFieldStrength)
        
    # get slice locations
    uSliceLocs, firstSliceIdx, sliceIdx = utils._get_slice_locations(dsets)
        
    # get echo times
    inversionTimes = utils._get_inversion_times(dsets)
    
    # get echo times
    echoTimes = utils._get_echo_times(dsets)
    
    # get echo numbers
    echoNumbers = utils._get_echo_numbers(dsets)
    
    # get repetition times
    repetitionTimes = utils._get_repetition_times(dsets)
    
    # get flip angles
    flipAngles = utils._get_flip_angles(dsets)
    
    # get sequence matrix
    contrasts = np.stack((inversionTimes, echoTimes, echoNumbers, repetitionTimes, flipAngles), axis=1)
    
    # get unique contrast and indexes
    uContrasts, contrastIdx = utils._get_unique_contrasts(contrasts)
            
    # get size
    n_slices = len(uSliceLocs)
    n_contrasts = uContrasts.shape[0]
    ninstances, ny, nx = image.shape
    
    # fill sorted image tensor
    sorted_image = np.zeros((n_contrasts, n_slices, ny, nx), dtype=image.dtype)
    for n in range(ninstances):
        sorted_image[contrastIdx[n], sliceIdx[n], :, :] = image[n]
        
    # get dicom template
    dcm_template = utils._get_dicom_template(dsets, firstSliceIdx)
    
    # unpack sequence
    TI, TE, EC, TR, FA = uContrasts.transpose()
    
    # squeeze
    if sorted_image.shape[0] == 1:
        sorted_image = sorted_image[0]
        
    return sorted_image, {'nifti_template': {}, 'dcm_template': dcm_template, 'B0': B0, 'EC': EC, 'TI': TI, 'TE': TE, 'TR': TR, 'FA': FA}
