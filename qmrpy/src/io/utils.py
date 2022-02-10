# -*- coding: utf-8 -*-
"""
Utility routines for DICOM files loading and sorting.

Created on Thu Jan 27 13:30:28 2022

@author: Matteo Cencini
"""
import copy
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import os


import numpy as np


import pydicom

    
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
    
    
def _dcmwrite(input):
    """
    Wrapper to pydicom dcmread to automatically handle path / file tuple
    """
    filename, dataset = input
    pydicom.dcmwrite(filename, dataset)



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


def _get_slice_locations(dsets):
    """
    Return array of unique slice locations and slice location index for each dataset in dsets.
    """
    # get unique slice locations
    sliceLocs = _get_relative_slice_position(dsets).round(decimals=4)
    uSliceLocs, firstSliceIdx = np.unique(sliceLocs, return_index=True)
    
    # get indexes
    sliceIdx = np.zeros(sliceLocs.shape, dtype=np.int)
    
    for n in range(len(uSliceLocs)):
        sliceIdx[sliceLocs == uSliceLocs[n]] = n
        
    return uSliceLocs, firstSliceIdx, sliceIdx


def _get_image_orientation(dsets):
    """
    Return image orientation matrix.
    """
    F = np.array(dsets[0].ImageOrientationPatient).reshape(2, 3)
    
    return F


def _get_plane_normal(dsets):
    """
    Return array of normal to imaging plane, as the cross product
    between x and y plane versors.
    """
    x, y = _get_image_orientation(dsets)
    return np.cross(x, y)


def _get_position(dsets):
    """
    Return matrix of image position of size (3, nslices).
    """
    return np.stack([dset.ImagePositionPatient for dset in dsets], axis=1)
    

def _get_relative_slice_position(dsets):
    """
    Return array of slice coordinates along the normal to imaging plane.
    """
    z = _get_plane_normal(dsets)
    position =  _get_position(dsets)
    return z @ position
    
    
def _get_flip_angles(dsets):
    """
    Return array of flip angles for each dataset in dsets.
    """
    # get flip angles
    flipAngles = np.array([float(dset.FlipAngle) for dset in dsets])
      
    return flipAngles


def _get_echo_times(dsets):
    """
    Return array of echo times for each dataset in dsets.
    """
    # get unique echo times
    echoTimes = np.array([float(dset.EchoTime) for dset in dsets])
          
    return echoTimes


def _get_repetition_times(dsets):
    """
    Return array of repetition times for each dataset in dsets.
    """
    # get unique repetition times
    repetitionTimes = np.array([float(dset.RepetitionTime) for dset in dsets])
            
    return repetitionTimes


def _get_inversion_times(dsets):
    """
    Return array of inversion times for each dataset in dsets.
    """
    try:
        # get unique repetition times
        inversionTimes = np.array([float(dset.InversionTime) for dset in dsets])           
    except:
        inversionTimes = np.zeros(len(dsets)) + np.inf
        
    return inversionTimes


def _get_unique_contrasts(constrasts):
    """
    Return ndarray of unique contrasts and contrast index for each dataset in dsets.
    """
    # get unique repetition times
    uContrasts = np.unique(constrasts, axis=0)
    
    # get indexes
    contrastIdx = np.zeros(constrasts.shape[0], dtype=np.int)
    
    for n in range(uContrasts.shape[0]):
        contrastIdx[(constrasts == uContrasts[n]).all(axis=-1)] = n
                 
    return uContrasts, contrastIdx


def _get_dicom_template(dsets, index):
    """
    Get template of Dicom to be used for saving later.
    """
    template = []

    SeriesNumber = dsets[index[0]].SeriesNumber
    
    for n in range(len(index)):
        dset = copy.deepcopy(dsets[index[n]])
        
        dset.pixel_array[:] = 0.0
        dset.PixelData = dset.pixel_array.tobytes()
                
        dset.WindowWidth = None
        dset.WindowCenter = None

        dset.SeriesDescription = None
        dset.SeriesNumber = SeriesNumber
        dset.SeriesInstanceUID = None
    
        dset.SOPInstanceUID = None
        dset.InstanceNumber = None
        dset.ImagesInAcquisition = None
        dset[0x0025, 0x1007].value = None
        dset[0x0025, 0x1019].value = None
        
        dset.InversionTime = '0'
        dset.EchoTime = '0'
        dset.EchoTrainLength = '1'
        dset.RepetitionTime = '0'
        dset.FlipAngle = '0'
        
        template.append(dset)
    
    return template


def _get_nifti_affine(dsets):
    """
    Return affine transform between voxel coordinates and mm coordinates as
    described in https://nipy.org/nibabel/dicom/spm_dicom.html#spm-volume-sorting
    """
    # common parameters
    T = _get_position(dsets)
    T1 = T[:, 0].round(4)
    
    F = _get_image_orientation(dsets)
    dr, dc = np.array(dsets[0].PixelSpacing).round(4)
    
    if len(dsets) == 1: # single slice case
        n = _get_plane_normal(dsets)
        ds = float(dsets[0].SliceThickness)
    
        A = np.stack((np.append(F[0] * dr, 0),
                      np.append(F[1] * dc, 0),
                      np.append(ds * n, 0),
                      np.append(T1, 1)), axis=1)

    else: # multi slice case
        N = len(dsets)
        TN = T[:,-1].round(4)
        A = np.stack((np.append(F[0] * dr, 0),
                      np.append(F[1] * dc, 0),
                      np.append((TN - T1) / (N - 1), 0),
                      np.append(T1, 1)), axis=1)
        
    # fix origin of y axis (wonder if it is true for arbitrary orientations...)
    A[:, -1] = A @ np.array([0, dsets[0].Columns - 1, 0, 1])
        
    return A
        