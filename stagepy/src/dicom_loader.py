# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 13:30:28 2022

@author: mcencini
"""
import os
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool


import numpy as np


import pydicom


#%% Utils
def _list_dcm(dicomdir):
    # get all files in dicom dir
    raw_list = os.listdir(dicomdir)
    
    # get only '.dcm' files and convert to absolute path
    filtered_list = []
    for item in raw_list:
        if item[-3:] == 'dcm':
            filtered_list.append(os.path.join(dicomdir, item))
            
    return filtered_list


def _load_dcm(dicomdir):
    
    # get list of dcm files
    dcm_list = _list_dcm(dicomdir)
    
    # make pool of workers
    pool = ThreadPool(multiprocessing.cpu_count())

    # each thread load a dicom
    dcm_template = pool.map(pydicom.dcmread, dcm_list)

    # cloose pool and wait finish   
    pool.close()
    pool.join() 

    # get image
    image, dcm_template = _get_image(dcm_template)

    # sort
    image, dcm_template = _sort_image(image, dcm_template)    
    
    return image, dcm_template


def _get_image(dcm_template):
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
    template_out = []
    
    # loop over dataset
    for dcm in dcm_template:
        if dcm[0x0043, 0x102f].value == 0:
            magnitude.append(dcm.pixel_array)
            template_out.append(dcm)
        
        if dcm[0x0043, 0x102f].value == 1:
            phase.append(dcm.pixel_array)
            
        if dcm[0x0043, 0x102f].value == 2:
            real.append(dcm.pixel_array)
        
        if dcm[0x0043, 0x102f].value == 3:
            imag.append(dcm.pixel_array)
            
    if real and imag:
        image = np.stack(real, axis=0).astype(np.float32) + 1j * np.stack(imag, axis=0).astype(np.float32)
        return image, template_out
    
    if magnitude and phase:
        image = np.stack(magnitude, axis=0).astype(np.float32) * np.exp( 1j * np.stack(phase, axis=0).astype(np.float32))
        return image, template_out
    
    image = np.stack(magnitude, axis=0).astype(np.float32) 
    return image, template_out


def _sort_image(image, dcm_template):
    """
    Sort image according to slice position and echoes
    
    Args:
        image: unsorted image series of size (ninstances, ny, nx)
        dcm_template: list of pydicom FileDataset
        
    Returns:
        image: sorted image of size (nechoes, nslices, ny, nx)
        dcm_template: list of pydicom FileDataset
    """    
    # get echo times
    echo_times = np.unique(np.stack([float(dcm.EchoTime) for dcm in dcm_template]))
    
    # get sizes
    ninstances, ny, nx = image.shape
    nte = len(echo_times)
    nslices = ninstances // nte
    
    # reshape image
    image = image.reshape(nslices, nte, ny, nx).swapaxes(0, 1)
    
    # fix number of images in dicom header
    for dcm in dcm_template:
        dcm[0x0025, 0x1007].value = ninstances
        dcm[0x0025, 0x1019].value = ninstances
    
    # return
    return np.ascontiguousarray(image), dcm_template
    
    
    

    
    
    