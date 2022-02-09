# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:30:28 2022

@author: Matteo Cencini
"""
import copy
import os
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
from typing import Union, Tuple, List, Dict


import numpy as np


import pydicom


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
    dsets = _load_dcm(dicomdir)
        
    # get slice locations
    uSliceLocs, firstSliceIdx, sliceIdx = _get_slice_locations(dsets)
        
    # get echo times
    inversionTimes = _get_inversion_times(dsets)
    
    # get echo times
    echoTimes = _get_echo_times(dsets)
    
    # get repetition times
    repetitionTimes = _get_repetition_times(dsets)
    
    # get flip angles
    flipAngles = _get_flip_angles(dsets)
    
    # get sequence matrix
    contrasts = np.stack((inversionTimes, echoTimes, repetitionTimes, flipAngles), axis=1)
    
    # get unique contrast and indexes
    uContrasts, contrastIdx = _get_unique_contrasts(contrasts)
        
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
    dcm_template = _get_dicom_template(dsets, firstSliceIdx)
    
    # unpack sequence
    TI, TE, TR, FA = uContrasts.transpose()
    
    # squeeze
    if sorted_image.shape[0] == 1:
        sorted_image = sorted_image[0]
        
    return sorted_image, {'template': dcm_template, 'TI': TI, 'TE': TE, 'TR': TR, 'FA': FA}


def write_dicom(image: np.ndarray, info: Dict, series_description: str, outpath: str = './output'):
    """
    Write parametric map to dicom.
    
    Args:
        dicomdir: string or list of strings with DICOM folders path.
        
    Returns:
        image: ndarray of image data to be written.
        info: dict with the following fields:
            - template: the DICOM template.
            - slice_locations: ndarray of Slice Locations [mm].
            - TI: ndarray of Inversion Times [ms].
            - TE: ndarray of Echo Times [ms].
            - TR: ndarray of Repetition Times [ms].
            - FA: ndarray of Flip Angles [deg].
        outpath: desired output path
    """
    # generate UIDs
    SeriesInstanceUID = pydicom.uid.generate_uid()
        
    # count number of instances
    ninstances = image.shape[0]
    
    # init dsets
    dsets = info['template']
    
    # cast image
    minval = np.iinfo(np.int16).min
    maxval = np.iinfo(np.int16).max
    image[image < minval] = minval
    image[image > maxval] = maxval
    image = image.astype(np.int16)
        
    # get max value
    windowWidth = np.percentile(np.abs(image), 99)
        
    # set properties
    for n in range(ninstances):
        dsets[n].pixel_array[:] = image[n]
        dsets[n].PixelData = image[n].tobytes()
                
        dsets[n].WindowWidth = str(windowWidth)
        dsets[n].WindowCenter = str(0.5 * windowWidth)

        dsets[n].SeriesDescription = series_description
        dsets[n].SeriesNumber = str(int(dsets[n].SeriesNumber) * 1000)
        dsets[n].SeriesInstanceUID = SeriesInstanceUID
    
        dsets[n].SOPInstanceUID = pydicom.uid.generate_uid()
        dsets[n].InstanceNumber = str(n + 1)
        dsets[n].ImagesInAcquisition = ninstances
        dsets[n][0x0025, 0x1007].value = ninstances
        dsets[n][0x0025, 0x1019].value = ninstances
        
        
    # generate file names
    filename = ['img-' + str(n).zfill(3) + '.dcm' for n in range(ninstances)]
    
    # generate output path
    outpath = os.path.abspath(outpath)
        
    # create output folder
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    # get dicompath
    dcm_paths = [os.path.join(outpath, file) for file in filename]
    
    # generate path / data pair
    path_data = [[dcm_paths[n], dsets[n]] for n in range(ninstances)]
    
    # make pool of workers
    pool = ThreadPool(multiprocessing.cpu_count())

    # each thread write a dicom
    dsets = pool.map(_dcmwrite, path_data)
    
    # cloose pool and wait finish   
    pool.close()
    pool.join()
            


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
    sliceLocs = np.array([round(float(dset.SliceLocation), 4) for dset in dsets])
    uSliceLocs, firstSliceIdx = np.unique(sliceLocs, return_index=True)
    
    # get indexes
    sliceIdx = np.zeros(sliceLocs.shape, dtype=np.int)
    
    for n in range(len(uSliceLocs)):
        sliceIdx[sliceLocs == uSliceLocs[n]] = n
        
    return uSliceLocs, firstSliceIdx, sliceIdx
    

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
    
    
    

    
    
    