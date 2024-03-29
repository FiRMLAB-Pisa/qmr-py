# -*- coding: utf-8 -*-
"""
DICOM and NiFTI files writing routines.

Created on Thu Feb 10 16:16:41 2022

@author: Matteo Cencini
"""
import copy
import multiprocessing
import os
import json


from multiprocessing.dummy import Pool as ThreadPool
from typing import Dict


import numpy as np


import nibabel as nib
import pydicom


from qmrpy.src.io import utils


__all__ = ['write_dicom', 'write_nifti']



def write_to_numpy(image: np.ndarray, info: Dict, outpath: str = './output'):
    """
    Write parametric map to npz.
    
    Args:
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
    pass

    
def write_dicom(image: np.ndarray, info: Dict, series_description: str, outpath: str = './output', series_number_scale=1000, series_number_offset=0):
    """
    Write parametric map to dicom.
    
    Args:        
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
    if info['dcm_template']:
        # generate UIDs
        SeriesInstanceUID = pydicom.uid.generate_uid()
                    
        # count number of instances
        ninstances = image.shape[0]
        
        # init dsets
        dsets = copy.deepcopy(info['dcm_template'])
        
        # generate series number
        series_number = str(series_number_scale * int(dsets[0].SeriesNumber) + series_number_offset)
        
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
            dsets[n].SeriesNumber = series_number
            dsets[n].SeriesInstanceUID = SeriesInstanceUID
        
            dsets[n].SOPInstanceUID = pydicom.uid.generate_uid()
            dsets[n].InstanceNumber = str(n + 1)
            
            try:
                dsets[n].ImagesInAcquisition = ninstances
            except:
                pass
            try:
                dsets[n][0x0025, 0x1007].value = ninstances
            except:
                pass
            try:
                dsets[n][0x0025, 0x1019].value = ninstances
            except:
                pass        
            
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
    
    
def write_nifti(image: np.ndarray, info: Dict, series_description: str = None, filename: str = 'output.nii', outpath: str = './'):
    """
    Write parametric map to dicom.
    
    Args:
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
        
    # reformat image
    image = np.flip(image.transpose(), axis=1)
    
    # cast image
    minval = np.iinfo(np.int16).min
    maxval = np.iinfo(np.int16).max
    image[image < minval] = minval
    image[image > maxval] = maxval
    image = image.astype(np.int16)
        
    if info['nifti_template']:
        affine = info['nifti_template']['affine']
        header = info['nifti_template']['header']
        json_dict = info['nifti_template']['json']
        nifti = nib.Nifti1Image(image, affine, header)
        
    elif info['dcm_template']:
        
        # we do not have json dict in this case
        json_dict = None
        
        # get voxel size
        dx, dy = np.array(info['dcm_template'][0].PixelSpacing).round(4)
        dz = round(float(info['dcm_template'][0].SliceThickness), 4)
        
        # get affine
        affine, _ = utils._get_nifti_affine(info['dcm_template'], image.shape[-3:])
                    
        try:
            windowMin = 0.5 * np.percentile(image[image < 0], 95)
        except:
            windowMin = 0
        try:
            windowMax = 0.5 * np.percentile(image[image > 0], 95)
        except:
            windowMax = 0
            
        # write nifti
        nifti = nib.Nifti1Image(image, affine)
        nifti.header['pixdim'][1:4] = np.array([dx, dy, dz])
        nifti.header['sform_code'] = 0
        nifti.header['qform_code'] = 2
        nifti.header['cal_min'] = windowMin 
        nifti.header['cal_max'] = windowMax 
        nifti.header.set_xyzt_units('mm', 'sec')
    
    # actual writing
    nib.save(nifti, os.path.join(outpath, filename))
    
    # write json
    if json_dict is not None:
        # fix series description
        if series_description is not None:
            json_dict['SeriesDescription'] = series_description
        jsoname = filename.split('.')[0] + '.json'
        with open(os.path.join(outpath, jsoname), 'w', encoding='utf-8') as f:
            json.dump(json_dict, f, ensure_ascii=False, indent=4)
    
    