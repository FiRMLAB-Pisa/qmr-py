# -*- coding: utf-8 -*-
"""
DICOM and NiFTI files writing routines.

Created on Thu Feb 10 16:16:41 2022

@author: Matteo Cencini
"""
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import os
from typing import Dict


import numpy as np


import nibabel as nib
import pydicom


from qmrpy.src.io import utils


__all__ = ['write_dicom', 'write_nifti']



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
        
    # get level
    windowMin = np.percentile(image, 5)
    windowMax = np.percentile(image, 95)                   
    windowWidth = windowMax - windowMin
        
    # set properties
    for n in range(ninstances):
        dsets[n].pixel_array[:] = image[n]
        dsets[n].PixelData = image[n].tobytes()
                
        dsets[n].WindowWidth = str(windowWidth)
        dsets[n].WindowCenter = str(0.5 * windowWidth)

        dsets[n].SeriesDescription = series_description
        dsets[n].SeriesNumber = str(int(dsets[n].SeriesNumber) * 1000)
        dsets[n].SeriesInstanceUID = SeriesInstanceUID
    
        # dsets[n].SOPInstanceUID = pydicom.uid.generate_uid()
        dsets[n].InstanceNumber = str(n + 1)
        
        try:
            dsets[n].ImagesInAcquisition = ninstances
            dsets[n][0x0025, 0x1007].value = ninstances
            dsets[n][0x0025, 0x1019].value = ninstances
        except:
            pass
        
        dsets[n][0x0018, 0x0086].value = '1' # Echo Number
        dsets[n].InversionTime = '0'
        dsets[n].EchoTime = '0'
        dsets[n].EchoTrainLength = '1'
        dsets[n].RepetitionTime = '0'
        dsets[n].FlipAngle = '0'
        
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
    dsets = pool.map(utils._dcmwrite, path_data)
    
    # cloose pool and wait finish   
    pool.close()
    pool.join()
    
    
def write_nifti(image: np.ndarray, info: Dict, filename: str = 'output.nii', outpath: str = './'):
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
        filename: name of the output nifti file.
        outpath: desired output path
    """
    # generate file name
    if filename.endswith('.nii') is False and filename.endswith('.nii.gz') is False:
        filename += '.nii'
    
    # generate output path
    outpath = os.path.abspath(outpath)
    
    # create output folder
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    # get voxel size
    dx, dy = np.array(info['template'][0].PixelSpacing).round(4)
    dz = round(float(info['template'][0].SliceThickness), 4)
    
    # get affine
    A = utils._get_nifti_affine(info['template'], image.shape[-3:])
    
    # reformat image
    image = np.flip(np.flip(image.transpose(), axis=0), axis=1)
    
    # cast image
    minval = np.iinfo(np.int16).min
    maxval = np.iinfo(np.int16).max
    image[image < minval] = minval
    image[image > maxval] = maxval
    image = image.astype(np.int16)
    
    try:
        windowMin = 0.5 * np.percentile(image[image < 0], 99)
    except:
        windowMin = 0
    try:
        windowMax = 0.5 * np.percentile(image[image > 0], 99)
    except:
        windowMax = 0
        
    # write nifti
    nifti = nib.Nifti1Image(image, A)
    nifti.header['pixdim'][1:4] = np.array([dx, dy, dz])
    nifti.header['sform_code'] = 1
    nifti.header['qform_code'] = 1
    nifti.header['cal_min'] = windowMin 
    nifti.header['cal_max'] = windowMax 
    nifti.header.set_xyzt_units('mm', 'sec')
    nib.save(nifti, os.path.join(outpath, filename))
    