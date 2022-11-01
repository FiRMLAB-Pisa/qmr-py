# -*- coding: utf-8 -*-
"""
Full reconstruction algorithms

Created on Thu Feb 10 17:33:53 2022

@author: Matteo Cencini
"""
from datetime import timedelta
import os
from time import time


import click
import numpy as np


from tqdm import tqdm


from qmrpy import io, inference


__all__ = ['longitudinal_relaxation', 
           'transverse_relaxation', 
           'transmit_field', 
           'phase_based_laplacian_ept',
           'phase_based_surface_integral_ept',
           'mp2rage_longitudinal_relaxation', 
           'flaws_longitudinal_relaxation']


def longitudinal_relaxation(input_path, output_path='./output', save_dicom=False, save_nifti=False, mask_threshold=0.05):
    """
    Reconstruct quantitative T1 maps from Inversion Recovery Spin-Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * (1 - 2 *exp(-TI / T1))
    
    where TI is the Inversion Time.
    """
    # check input
    if isinstance(input_path, (list, tuple)):
        input_path = [os.path.abspath(path) for path in input_path]
        rootdir = input_path[0].split(os.sep)[-2]
        
    elif input_path.endswith('/') or input_path.endswith('\\'):
        folders = sorted(os.listdir(input_path))
        input_path = [os.path.normpath(os.path.abspath(os.path.join(input_path, folder))) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.normpath(os.path.abspath(os.path.join(output_path, rootdir)))
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_data(input_path)
        ti = info['TI']
        pbar.update(step)
        
        # mask data
        if mask_threshold > 0:
            mask = inference.utils.mask(img)
        else:
            mask = None
            
        pbar.set_description("computing longitudinal relaxation map...")
        longitudinal_relaxation_map = inference.ir_se_t1_fitting(img, ti, mask)
        pbar.update(step)
        
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(longitudinal_relaxation_map, info, output_label + '_qt1', output_path + '_qt1')        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(longitudinal_relaxation_map, info, output_label + '_qt1', output_path + '_qt1')
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return longitudinal_relaxation_map, img

    
def transverse_relaxation(input_path, output_path='./output', save_dicom=False, save_nifti=False, mask_threshold=0.05):
    """
    Reconstruct quantitative T2 / T2* maps from Multi-Echo Spin-Echo / Gradient Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * exp(-TE / T2)
    
    where T2 is replaced by T2* for Gradient Echo data and TE is the Echo Time.
    """
    # check input
    if isinstance(input_path, (list, tuple)):
        input_path = [os.path.abspath(path) for path in input_path]
        rootdir = input_path[0].split(os.sep)[-2]
        
    elif input_path.endswith('/') or input_path.endswith('\\'):
        folders = sorted(os.listdir(input_path))
        input_path = [os.path.normpath(os.path.abspath(os.path.join(input_path, folder))) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.normpath(os.path.abspath(os.path.join(output_path, rootdir)))
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_data(input_path)
        te = info['TE']
        pbar.update(step)
        
        # mask data
        if mask_threshold > 0:
            mask = inference.utils.mask(img)
        else:
            mask = None
            
        pbar.set_description("computing transverse relaxation map...")
        transverse_relaxation_map = inference.me_transverse_relaxation_fitting(img, te, mask)
        pbar.update(step)
        
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(transverse_relaxation_map, info, output_label + '_qt2', output_path + '_qt2')         
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(transverse_relaxation_map, info, output_label + '_qt2', output_path + '_qt2')
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return transverse_relaxation_map, img
    

def transmit_field(input_path, output_path='./output', save_dicom=False, save_nifti=False, mask_threshold=0.05):
    """
    Reconstruct quantitative B1+ maps from Double Angle Spin-Echo / Gradient Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * sin(theta)
    
    where theta is the nominal Flip Angle.
    """
    # check input
    if isinstance(input_path, (list, tuple)):
        input_path = [os.path.abspath(path) for path in input_path]
        rootdir = input_path[0].split(os.sep)[-2]
        
    elif input_path.endswith('/') or input_path.endswith('\\'):
        folders = sorted(os.listdir(input_path))
        input_path = [os.path.normpath(os.path.abspath(os.path.join(input_path, folder))) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.normpath(os.path.abspath(os.path.join(output_path, rootdir)))
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_data(input_path)
        fa = info['FA']
        pbar.update(step)
        
        # mask data
        if mask_threshold > 0:
            mask = inference.utils.mask(img)
        else:
            mask = None
            
        pbar.set_description("computing transmit field magnitude map...")
        transmit_field_map = inference.b1_dam_fitting(img, fa, mask)
        pbar.update(step)
        
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(transmit_field_map, info, output_label + '_qb1', output_path + '_qb1')          
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(transmit_field_map, info, output_label + '_qb1', output_path + '_qb1')
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return transmit_field_map
    

def phase_based_laplacian_ept(input_path, output_path='./output',
                              output_label = None,
                              save_dicom=False, save_nifti=False, 
                              mask_path=None, mask_threshold=0.05, local_mask_threshold=np.inf,
                              gaussian_preprocessing_sigma=0.0, gaussian_weight_sigma=0.45, 
                              laplacian_kernel_width=16, laplacian_kernel_shape='ellipsoid',
                              nclasses=3, merge_wm_csf=False,
                              median_filter_width=0, fft_shift_along_z=True):
    """
    Reconstruct quantitative conductivity maps from bSSFP data.
    
    Use the following signal model:
        
        s(t) = - Nabla Phi / Phi / (2 * omega0 * mu0)
    
    where Phi is bSSFP phase, omega0 is the larmor frequency and mu0 is the vacuum permittivity.
    """
    # check input
    if isinstance(input_path, (list, tuple)):
        input_path = [os.path.abspath(path) for path in input_path]
        rootdir = input_path[0].split(os.sep)[-2]
        
    elif input_path.endswith('/') or input_path.endswith('\\'):
        folders = sorted(os.listdir(input_path))
        input_path = [os.path.normpath(os.path.abspath(os.path.join(input_path, folder))) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    if output_label is None:
        output_label = rootdir
        output_path = os.path.normpath(os.path.abspath(os.path.join(output_path, rootdir)))
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_data(input_path)
        
        # get info
        if info['dcm_template']:
            resolution = np.array([float(info['dcm_template'][0].SliceThickness)] + [float(dr) for dr in info['dcm_template'][0].PixelSpacing]) * 1e-3
            omega0 = 2 * np.pi * float(info['dcm_template'][0].ImagingFrequency) * 1e6
        elif info['nifti_template']:
            resolution = np.flip(info['nifti_template']['header']['pixdim'][1:4]) * 1e-3
            omega0 = 2 * np.pi * float(info['nifti_template']['json']['ImagingFrequency']) * 1e6
        te = info['TE']
        pbar.update(step)
        
        # mask data
        if mask_path is not None:
            
            # get probabilistic segmentation
            segmentation, _ = io.read_segmentation(mask_path)
            
            # merge CSF and WM (inner brain)
            if merge_wm_csf is True:
                segmentation_4_ept = np.stack([segmentation[0], segmentation[1] + segmentation[2]], axis=0)
            else:
                segmentation_4_ept = segmentation.copy()
                    
            # get most probable tissue for each voxels
            winner = (segmentation_4_ept.sum(axis=0) > 0) * (segmentation_4_ept.argmax(axis=0) + 1)
            
            # build binary mask
            mask = np.zeros(segmentation_4_ept.shape, dtype=bool)
            
            for n in range(segmentation_4_ept.shape[0]):
                mask[n][winner == n+1] = True
            
        elif mask_threshold > 0:
            mask = inference.utils.mask(img)
            segmentation = None
        else:
            mask = None
            segmentation = None
        
        # fix kernel width
        if isinstance(laplacian_kernel_width, (list, tuple)) is False:
            if mask is not None and len(mask.shape) == 4:
                laplacian_kernel_width = mask.shape[0] * [laplacian_kernel_width]
                
        if isinstance(median_filter_width, (list, tuple)) is False:
            if mask is not None and len(mask.shape) == 4:
                median_filter_width = mask.shape[0] * [median_filter_width]
            
        pbar.set_description("computing conductivity map...")
        conductivity_map, phase, laplacian = inference.PhaseBasedLaplacianEPT(img, resolution, omega0, 
                                                            gaussian_preprocessing_sigma, gaussian_weight_sigma,
                                                            laplacian_kernel_width, laplacian_kernel_shape,
                                                            median_filter_width, mask, te, fft_shift_along_z,
                                                            local_mask_threshold)                            
        pbar.update(step)
        
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(conductivity_map, info, output_label + '_sigma', output_path + '_sigma')        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(1000*conductivity_map, info, output_label, output_path)
            
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return conductivity_map, img, phase, laplacian, segmentation


def phase_based_surface_integral_ept(input_path, output_path='./output', 
                                     save_dicom=False, save_nifti=False, 
                                     mask_path=None, mask_threshold=0.05, 
                                     gaussian_preprocessing_sigma=0.0, gaussian_weight_sigma=0.45, 
                                     kernel_diff_width=4, kernel_int_width=4, kernel_shape='ellipsoid',
                                     median_filter_width=0):
    """
    Reconstruct quantitative conductivity maps from bSSFP data.
    
    Use the following signal model:
        
        s(t) = - Nabla Phi / Phi / (2 * omega0 * mu0)
    
    where Phi is bSSFP phase, omega0 is the larmor frequency and mu0 is the vacuum permittivity.
    """
    # check input
    if isinstance(input_path, (list, tuple)):
        input_path = [os.path.abspath(path) for path in input_path]
        rootdir = input_path[0].split(os.sep)[-2]
        
    elif input_path.endswith('/') or input_path.endswith('\\'):
        folders = sorted(os.listdir(input_path))
        input_path = [os.path.normpath(os.path.abspath(os.path.join(input_path, folder))) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.normpath(os.path.abspath(os.path.join(output_path, rootdir)))
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_data(input_path)
        
        # get info
        if info['dcm_template']:
            resolution = np.array([float(info['dcm_template'][0].SliceThickness)] + [float(dr) for dr in info['dcm_template'][0].PixelSpacing]) * 1e-3
            omega0 = 2 * np.pi * float(info['dcm_template'][0].ImagingFrequency) * 1e6
        elif info['nifti_template']:
            resolution = np.flip(info['nifti_template']['header']['pixdim'][1:4]) * 1e-3
            omega0 = 2 * np.pi * float(info['nifti_template']['json']['ImagingFrequency']) * 1e6
        pbar.update(step)
        
        # mask data
        if mask_path is not None:
            
            # get probabilistic segmentation
            segmentation, _ = io.read_segmentation(mask_path)
            
            # get most probable tissue for each voxels
            winner = (segmentation.sum(axis=0) > 0) * (segmentation.argmax(axis=0) + 1)
            
            # build binary mask
            mask = np.zeros(segmentation.shape, dtype=bool)
            
            for n in range(3):
                mask[n][winner == n+1] = True
            
        elif mask_threshold > 0:
            mask = inference.utils.mask(img)
        else:
            mask = None

        pbar.set_description("computing conductivity map...")
        conductivity_map = inference.PhaseBasedSurfaceIntegralEPT(img, resolution, omega0, 
                                                                  gaussian_preprocessing_sigma, gaussian_weight_sigma,
                                                                  kernel_diff_width, kernel_int_width, kernel_shape,
                                                                  median_filter_width, mask)                            
        pbar.update(step)
        
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(conductivity_map, info, output_label + '_sigma', output_path + '_sigma')        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(conductivity_map, info, output_label + '_sigma', output_path + '_sigma')
            
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return conductivity_map, mask


def mp2rage_longitudinal_relaxation(inversion_times, tr_flash, flip_angles, input_path, output_path='./output', save_dicom=False, save_nifti=False):
    """
    Reconstruct quantitative T1 maps from MP2RAGEDATA data.
    """
    # check input
    if isinstance(input_path, (list, tuple)):
        input_path = [os.path.abspath(path) for path in input_path]
        rootdir = input_path[0].split(os.sep)[-2]
        
    elif input_path.endswith('/') or input_path.endswith('\\'):
        folders = sorted(os.listdir(input_path))
        input_path = [os.path.normpath(os.path.abspath(os.path.join(input_path, folder))) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.normpath(os.path.abspath(os.path.join(output_path, rootdir)))
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_data(input_path)
        ti = np.asarray(inversion_times, dtype=np.float64)
        fa = np.asarray(flip_angles, dtype=np.float64)
        tr = np.asarray(tr_flash, dtype=np.float64)
        B0 = info['B0']
        pbar.update(step)
                    
        pbar.set_description("computing longitudinal relaxation map...")
        longitudinal_relaxation_map, uni_img = inference.mp2rage_t1_fitting(img, ti, fa, tr, B0)
        pbar.update(step)
        
        # export unified image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(uni_img, info, rootdir + '_uni', output_path + '_uni')        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(uni_img, info, rootdir + '_uni', output_path + '_uni')
        
        # export t1map
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(longitudinal_relaxation_map, info, output_label + '_qt1', output_path + '_qt1')        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(longitudinal_relaxation_map, info, output_label + '_qt1', output_path + '_qt1')
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return longitudinal_relaxation_map, uni_img


def flaws_longitudinal_relaxation(inversion_times, flip_angles, tr_flash, input_path, inversion_efficiency=1.0, beta=0, output_path='./output', save_dicom=False, save_nifti=False):
    """
    Reconstruct quantitative T1 maps from FLAWS data.
    """
    # check input
    if isinstance(input_path, (list, tuple)):
        input_path = [os.path.abspath(path) for path in input_path]
        rootdir = input_path[0].split(os.sep)[-2]
        
    elif input_path.endswith('/') or input_path.endswith('\\'):
        folders = sorted(os.listdir(input_path))
        input_path = [os.path.normpath(os.path.abspath(os.path.join(input_path, folder))) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.normpath(os.path.abspath(os.path.join(output_path, rootdir)))
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_data(input_path)
        ti = np.asarray(inversion_times, dtype=np.float64)
        fa = np.asarray(flip_angles, dtype=np.float64)
        tr_flash = np.asarray(tr_flash, dtype=np.float64)
        tr = info['TR'][0]
        B0 = info['B0']
        pbar.update(step)
                    
        pbar.set_description("computing longitudinal relaxation map...")
        longitudinal_relaxation_map, uni_img, min_img, hc_img, hco_img = inference.mp2rage_t1_fitting(img, ti, fa, tr_flash, tr, B0, beta, inversion_efficiency, sequence='flaws')
        pbar.update(step)
        
        # export wm suppressed image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(np.abs(img[0]), info, 'FLAWS White Matter Suppressed', output_path + '_wms', series_number_offset=100)        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(np.abs(img[0]), info, rootdir + '_wms', output_path + '_wms')
            
        # export csf suppressed image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(np.abs(img[1]), info, 'FLAWS CSF Suppressed', output_path + '_csfs', series_number_offset=200)        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(np.abs(img[1]), info, rootdir + '_csfs', output_path + '_csfs')
        
        # export unified image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(1000 * uni_img, info, 'FLAWS Unified T1w', output_path + '_uni', series_number_offset=500)        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(1000 * uni_img, info, rootdir + '_uni', output_path + '_uni')
            
        # export minimum image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(1000 * min_img, info, 'FLAWS Minimum Image', output_path + '_min', series_number_offset=600)        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(1000 * min_img, info, rootdir + '_min', output_path + '_min')
            
        # export hc image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(1000 * hc_img, info, 'FLAWS High-Contrast Image', output_path + '_hc', series_number_offset=300)        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(1000 * hc_img, info, rootdir + '_hc', output_path + '_hc')
            
        # export hco image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(1000 * hco_img, info, 'FLAWS T1w-MP2RAGE-like image', output_path + '_hco', series_number_offset=400)        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(1000 * hco_img, info, rootdir + '_hco', output_path + '_hco')
        
        # export t1map
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(longitudinal_relaxation_map, info, 'FLAWS T1 map [ms]', output_path + '_qt1', series_number_offset=10)        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(longitudinal_relaxation_map, info, output_label + '_qt1', output_path + '_qt1')
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return longitudinal_relaxation_map, uni_img, min_img, hc_img, hco_img, img



