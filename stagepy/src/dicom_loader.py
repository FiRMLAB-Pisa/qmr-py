# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:30:28 2022

@author: mcencini
"""
import copy
import os
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool


import numpy as np


import pydicom


def load_ir_se_data(dicomdir):
    """
    Load Inversion Recovery Spin Echo images for gold standard T1 mapping.
    
    Args:
        dicomdir: string or list of strings with DICOM folders path.
        
    Returns:
        image: ndarray of sorted image data.
        info: dict with the following fields:
            - template: the DICOM template.
            - slice_locations: ndarray of Slice Locations [mm].
            - TI: ndarray of Inversion Times [ms].
    """
    # load dicom
    dsets = _load_dcm(dicomdir)
    
    # get slice locations
    uSliceLoc, sliceIdx = _get_slice_location(dsets)
    
    # get echo times
    uInversionTimes, inversionIdx = _get_inversion_time(dsets)
    
    # extract image tensor
    image = np.stack([dset.pixel_array for dset in dsets], axis=0)
    
    # get size
    n_slices = len(uSliceLoc)
    n_inversion_times = len(uInversionTimes)
    ninstances, ny, nx = image.shape
    
    # fill sorted image tensor
    sorted_image = np.zeros((n_inversion_times, n_slices, ny, nx), dtype=np.float32)
    for n in range(ninstances):
        sorted_image[inversionIdx[n], sliceIdx[n], :, :] = image[n]
        
    # get dicom template
    dcm_template = _get_dicom_template(dsets)
    dcm_template.InversionTime = None
    
    return sorted_image, {'template': dcm_template, 'slice_locations': uSliceLoc, 'TI': uInversionTimes}


def load_me_se_data(dicomdir):
    """
    Load Spin Echo images for gold standard T2 mapping.
    
    Args:
        dicomdir: string or list of strings with DICOM folders path.
        
    Returns:
        image: ndarray of sorted image data.
        info: dict with the following fields:
            - template: the DICOM template.
            - slice_locations: ndarray of Slice Locations [mm].
            - TE: ndarray of Echo Times [ms].
    """
    # load dicom
    dsets = _load_dcm(dicomdir)
    
    # get slice locations
    uSliceLoc, sliceIdx = _get_slice_location(dsets)
    
    # get echo times
    uEchoTimes, echoIdx = _get_echo_time(dsets)
    
    # extract image tensor
    image = np.stack([dset.pixel_array for dset in dsets], axis=0)
    
    # get size
    n_slices = len(uSliceLoc)
    n_echo_times = len(uEchoTimes)
    ninstances, ny, nx = image.shape
    
    # fill sorted image tensor
    sorted_image = np.zeros(( n_echo_times, n_slices, ny, nx), dtype=np.float32)
    for n in range(ninstances):
        sorted_image[echoIdx[n], sliceIdx[n], :, :] = image[n]
        
    # get dicom template
    dcm_template = _get_dicom_template(dsets)
    dcm_template.EchoTime = None
    
    return sorted_image, {'template': dcm_template, 'slice_locations': uSliceLoc, 'TE': uEchoTimes}


def load_vfa_gre_data(dicomdir):
    """
    Load Variable Flip Angle Gradient Echo images for T1 mapping or B1+ mapping.
    
    Args:
        dicomdir: string or list of strings with DICOM folders path.
        
    Returns:
        image: ndarray of sorted image data.
        info: dict with the following fields:
            - template: the DICOM template.
            - slice_locations: ndarray of Slice Locations [mm].
            - FA: ndarray of Flip Angles [deg].
    """
    # load dicom
    dsets = _load_dcm(dicomdir)
    
    # get slice locations
    uSliceLoc, sliceIdx = _get_slice_location(dsets)
    
    # get flip angles
    uFlipAngles, flipAngleIdx = _get_flip_angle(dsets)
        
    # extract image tensor
    image = np.stack([dset.pixel_array for dset in dsets], axis=0)
    
    # get size
    n_slices = len(uSliceLoc)
    n_flip = len(uFlipAngles)
    ninstances, ny, nx = image.shape
    
    # fill sorted image tensor
    sorted_image = np.zeros((n_flip, n_slices, ny, nx), dtype=np.float32)
    for n in range(ninstances):
        sorted_image[flipAngleIdx[n], sliceIdx[n], :, :] = image[n]
        
    # get dicom template
    dcm_template = _get_dicom_template(dsets)
    dcm_template.FlipAngle = None
    
    return sorted_image, {'template': dcm_template, 'slice_locations': uSliceLoc, 'FA': uFlipAngles}


def load_me_gre_data(dicomdir):
    """
    Load Multi-Echo Gradient Echo images for QSM and T2* mapping or B0 and Water/Fat mapping.
    
    Args:
        dicomdir: string or list of strings with DICOM folders path.
        
    Returns:
        image: ndarray of sorted image data.
        info: dict with the following fields:
            - template: the DICOM template.
            - slice_locations: ndarray of Slice Locations [mm].
            - TE: ndarray of Echo Times [ms].
    """
    # load dicom
    dsets = _load_dcm(dicomdir)
    
    # get slice locations
    uSliceLoc, sliceIdx = _get_slice_location(dsets)
    
    # get echo times
    uEchoTimes, echoIdx = _get_echo_time(dsets)
    
    # extract image tensor
    image = np.stack([dset.pixel_array for dset in dsets], axis=0)
    
    # get size
    n_slices = len(uSliceLoc)
    n_echo_times = len(uEchoTimes)
    ninstances, ny, nx = image.shape
    
    # fill sorted image tensor
    sorted_image = np.zeros((n_echo_times, n_slices, ny, nx), dtype=np.complex64)
    for n in range(ninstances):
        sorted_image[echoIdx[n], sliceIdx[n], :, :] = image[n]
        
    # get dicom template
    dcm_template = _get_dicom_template(dsets)
    dcm_template.EchoTime = None
    
    return sorted_image, {'template': dcm_template, 'slice_locations': uSliceLoc, 'TE': uEchoTimes}
    

def load_vfa_me_gre_data(dicomdir):
    """
    Load Variable Flip Angle Multi-Echo Gradient Echo images for T1, B1+, QSM and T2* mapping (STAGE).
    
    Args:
        dicomdir: string or list of strings with DICOM folders path.
        
    Returns:
        image: ndarray of sorted image data.
        info: dict with the following fields:
            - template: the DICOM template.
            - slice_locations: ndarray of Slice Locations [mm].
            - FA: ndarray of Flip Angles [deg].
            - TE: ndarray of Echo Times [ms].
    """
    # load dicom
    dsets = _load_dcm(dicomdir)
    
    # get slice locations
    uSliceLoc, sliceIdx = _get_slice_location(dsets)
    
    # get flip angles
    uFlipAngles, flipAngleIdx = _get_flip_angle(dsets)
    
    # get echo times
    uEchoTimes, echoIdx = _get_echo_time(dsets)
    
    # extract image tensor
    image = np.stack([dset.pixel_array for dset in dsets], axis=0)
    
    # get size
    n_slices = len(uSliceLoc)
    n_echo_times = len(uEchoTimes)
    n_flip = len(uFlipAngles)
    ninstances, ny, nx = image.shape
    
    # fill sorted image tensor
    sorted_image = np.zeros((n_flip, n_echo_times, n_slices, ny, nx), dtype=np.complex64)
    for n in range(ninstances):
        sorted_image[flipAngleIdx[n], echoIdx[n], sliceIdx[n], :, :] = image[n]
        
    # get dicom template
    dcm_template = _get_dicom_template(dsets)
    dcm_template.FlipAngle = None
    dcm_template.EchoTime = None
    
    return sorted_image, {'template': dcm_template, 'slice_locations': uSliceLoc, 'FA': uFlipAngles, 'TE': uEchoTimes}


#%% Utils
def _get_dicom_paths(dicomdir):
    """
    Get path to all DICOMs in a directory or a list of directories.
    """
    
    # get all files in dicom dir
    if isinstance(dicomdir, (tuple, list)):
        dcm_paths = _get_full_path(dicomdir[0], os.listdir(dicomdir[0]))
        for d in range(1, len(dicomdir)):
            dcm_paths += _get_full_path(dicomdir[d], os.listdir(dicomdir[d]))
    else:
        dcm_paths = _get_full_path(dicomdir, os.listdir(dicomdir))
                    
    return dcm_paths


def _get_full_path(root, file_list):
    """
    Create list of full file paths from file name and root folder path.
    """
    return [os.path.abspath(os.path.join(root, file)) for file in file_list]


def _load_dcm(dicomdir):
    """    
    load list of dcm files and automatically gather real/imag or magnitude/phase to complex image.
    """
    # get list of dcm files
    dcm_paths = _get_dicom_paths(dicomdir)
    
    # make pool of workers
    pool = ThreadPool(multiprocessing.cpu_count())

    # each thread load a dicom
    dsets = pool.map(_dcmread, dcm_paths)
    
    # cloose pool and wait finish   
    pool.close()
    pool.join() 
    
    # filter None
    dsets = [dset for dset in dsets if dset is not None]

    # cast image to complex
    dsets = _cast_to_complex(dsets)
    
    return dsets


def _dcmread(dcm_path):
    """
    Wrapper to pydicom dcmread to automatically handle not dicom files
    """
    try:
        return pydicom.dcmread(dcm_path)
    except:
        return None


def _cast_to_complex(dsets_in):
    """
    Attempt to retrive complex image with the following priority:
        
        1) Real + 1j Imag
        2) Magnitude * exp(1j * Phase)

    If neither Real / Imag nor Phase are found, returns Magnitude only.
    """
    # initialize
    real = []
    imag = []
    magnitude = []
    phase = []
    
    # allocate template out
    dsets_out = []
    
    # loop over dataset
    for dset in dsets_in:
        if dset[0x0043, 0x102f].value == 0:
            magnitude.append(dset.pixel_array)
            dsets_out.append(dset)
        
        if dset[0x0043, 0x102f].value == 1:
            phase.append(dset.pixel_array)
            
        if dset[0x0043, 0x102f].value == 2:
            real.append(dset.pixel_array)
        
        if dset[0x0043, 0x102f].value == 3:
            imag.append(dset.pixel_array)
            
    if real and imag:
        image = np.stack(real, axis=0).astype(np.float32) + 1j * np.stack(imag, axis=0).astype(np.float32)
    
    if magnitude and phase:
        image = np.stack(magnitude, axis=0).astype(np.float32) * np.exp( 1j * np.stack(phase, axis=0).astype(np.float32))
    else:
        image = np.stack(magnitude, axis=0).astype(np.float32)
        
    # count number of instances
    ninstances = image.shape[0]

    # assign to pixel array
    for n in range(ninstances):
        dsets_out[n].pixel_array[:] = image[n]
        dsets_out[n][0x0025, 0x1007].value = ninstances
        dsets_out[n][0x0025, 0x1019].value = ninstances
        
    return dsets_out


def _get_slice_location(dsets):
    """
    Return array of unique slice locations and slice location index for each dataset in dsets.
    """
    # get unique slice locations
    sliceLoc = np.array([float(dset.SliceLocation) for dset in dsets])
    uSliceLoc = np.unique(sliceLoc)
    
    # get indexes
    sliceIdx = np.zeros(sliceLoc.shape, dtype=np.int)
    
    for n in range(len(uSliceLoc)):
        sliceIdx[sliceLoc == uSliceLoc[n]] = n
        
    return uSliceLoc, sliceIdx
    

def _get_flip_angle(dsets):
    """
    Return array of unique flip angles and flip angle index for each dataset in dsets.
    """
    # get unique flip angles
    flipAngle = np.array([float(dset.FlipAngle) for dset in dsets])
    uFlipAngle = np.unique(flipAngle)
    
    # get indexes
    flipIdx = np.zeros(flipAngle.shape, dtype=np.int)
    
    for n in range(len(uFlipAngle)):
        flipIdx[flipAngle == uFlipAngle[n]] = n
        
    return uFlipAngle, flipIdx


def _get_echo_time(dsets):
    """
    Return array of unique echo times and echo time index for each dataset in dsets.
    """
    # get unique echo times
    echoTime = np.array([float(dset.EchoTime) for dset in dsets])
    uEchoTime = np.unique(echoTime)
    
    # get indexes
    echoIdx = np.zeros(echoTime.shape, dtype=np.int)
    
    for n in range(len(uEchoTime)):
        echoIdx[echoTime == uEchoTime[n]] = n
        
    return uEchoTime, echoIdx


def _get_repetition_time(dsets):
    """
    Return array of unique repetition times and repetition time index for each dataset in dsets.
    """
    # get unique repetition times
    repetitionTime = np.array([float(dset.RepetitionTime) for dset in dsets])
    uRepetitionTime = np.unique(repetitionTime)
    
    # get indexes
    repetitionIdx = np.zeros(repetitionTime.shape, dtype=np.int)
    
    for n in range(len(uRepetitionTime)):
        repetitionIdx[repetitionTime == uRepetitionTime[n]] = n
        
    return uRepetitionTime, repetitionIdx


def _get_inversion_time(dsets):
    """
    Return array of unique inversion times and inversion time index for each dataset in dsets.
    """
    # get unique repetition times
    inversionTime = np.array([float(dset.InversionTime) for dset in dsets])
    uInversionTime = np.unique(inversionTime)
    
    # get indexes
    inversionIdx = np.zeros(inversionTime.shape, dtype=np.int)
    
    for n in range(len(uInversionTime)):
        inversionIdx[inversionTime == uInversionTime[n]] = n
        
    return uInversionTime, inversionIdx


def _get_dicom_template(dsets):
    """
    Get template of Dicom to be used for saving later.
    """
    dset = copy.deepcopy(dsets[0])
    dset.pixel_array[:] = 0.0
    dset.SliceLocation = None


# def _sort_image(image, dcm_template):
#     """
#     Sort image according to slice position and echoes
    
#     Args:
#         image: unsorted image series of size (ninstances, ny, nx)
#         dcm_template: list of pydicom FileDataset
        
#     Returns:
#         image: sorted image of size (nechoes, nslices, ny, nx)
#         dcm_template: list of pydicom FileDataset
#     """    
#     # get echo times
#     echo_times = np.unique(np.stack([float(dcm.EchoTime) for dcm in dcm_template]))
    
#     # get sizes
#     ninstances, ny, nx = image.shape
#     nte = len(echo_times)
#     nslices = ninstances // nte
    
#     # reshape image
#     image = image.reshape(nslices, nte, ny, nx).swapaxes(0, 1)
    
#     # fix number of images in dicom header
#     for dcm in dcm_template:
#         dcm[0x0025, 0x1007].value = ninstances
#         dcm[0x0025, 0x1019].value = ninstances
    
#     # return
#     return np.ascontiguousarray(image), dcm_template
    
    
    

    
    
    