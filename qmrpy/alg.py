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


# from scipy.signal import medfilt


from tqdm import tqdm


from qmrpy import io, inference
# from qmrpy.src.inference.helmholtz_ept import HelmholtzConductivity


__all__ = ['longitudinal_relaxation', 'transverse_relaxation', 'transmit_field', 'helmholtz_ept', 'mp2rage_longitudinal_relaxation', 'flaws_longitudinal_relaxation']


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
        input_path = [os.path.join(input_path, folder) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.abspath(os.path.join(output_path, rootdir))
            
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
    
    return longitudinal_relaxation_map

    
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
        input_path = [os.path.join(input_path, folder) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.abspath(os.path.join(output_path, rootdir))
            
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
    
    return transverse_relaxation_map
    

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
        input_path = [os.path.join(input_path, folder) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.abspath(os.path.join(output_path, rootdir))
            
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
    

def helmholtz_ept(input_path, output_path='./output', 
                  save_dicom=False, save_nifti=False, 
                  mask_path=None, mask_threshold=0.05, 
                  gaussian_kernel_sigma=0.0, gaussian_weight_sigma=0.0, 
                  laplacian_kernel_width=16, median_filter_width=0,
                  fitting_threshold=50):
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
        input_path = [os.path.join(input_path, folder) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.abspath(os.path.join(output_path, rootdir))
            
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
        if len(mask.shape) == 3:
            conductivity_map = inference.helmholtz_conductivity_fitting(img, resolution, omega0, 
                                                                        gaussian_kernel_sigma, gaussian_weight_sigma,
                                                                        laplacian_kernel_width, median_filter_width,
                                                                        fitting_threshold, mask)
        
        else:
            if np.isscalar(gaussian_kernel_sigma):
                gaussian_kernel_sigma = 3 * [gaussian_kernel_sigma]
            
            if np.isscalar(gaussian_weight_sigma):
                gaussian_weight_sigma = 3 * [gaussian_weight_sigma]
                
            if np.isscalar(laplacian_kernel_width):
                laplacian_kernel_width = 3 * [laplacian_kernel_width]
                                                
            conductivity_map = []
            for n in range(3):
                conductivity_map.append(inference.helmholtz_conductivity_fitting(img, resolution, omega0, 
                                                                                 gaussian_kernel_sigma[n], gaussian_weight_sigma[n],
                                                                                 laplacian_kernel_width[n], 0, fitting_threshold, mask[n]))
                
            # stack result
            conductivity_map = np.stack(conductivity_map, axis=0)
            mask = mask.sum(axis=0) > 0
            conductivity_map = mask * conductivity_map.sum(axis=0)
                            
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
    
    return conductivity_map


def mp2rage_longitudinal_relaxation(inversion_times, tr_flash, input_path, output_path='./output', save_dicom=False, save_nifti=False):
    """
    Reconstruct quantitative T1 maps from MP2RAGEDATA data.
    """
    # check input
    if isinstance(input_path, (list, tuple)):
        input_path = [os.path.abspath(path) for path in input_path]
        rootdir = input_path[0].split(os.sep)[-2]
        
    elif input_path.endswith('/') or input_path.endswith('\\'):
        folders = sorted(os.listdir(input_path))
        input_path = [os.path.join(input_path, folder) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.abspath(os.path.join(output_path, rootdir))
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_data(input_path)
        ti = inversion_times
        fa = info['FA']
        tr = tr_flash
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


def flaws_longitudinal_relaxation(inversion_times, tr_flash, input_path, output_path='./output', save_dicom=False, save_nifti=False):
    """
    Reconstruct quantitative T1 maps from FLAWS data.
    """
    # check input
    if isinstance(input_path, (list, tuple)):
        input_path = [os.path.abspath(path) for path in input_path]
        rootdir = input_path[0].split(os.sep)[-2]
        
    elif input_path.endswith('/') or input_path.endswith('\\'):
        folders = sorted(os.listdir(input_path))
        input_path = [os.path.join(input_path, folder) for folder in folders]
        rootdir = input_path[0].split(os.sep)[-2]
    else:
        input_path = os.path.abspath(input_path)
        rootdir = input_path.split(os.sep)[-1]
        
    # get output
    output_label = rootdir
    output_path = os.path.abspath(os.path.join(output_path, rootdir))
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_data(input_path)
        ti = inversion_times
        fa = info['FA']
        tr = tr_flash
        B0 = info['B0']
        pbar.update(step)
                    
        pbar.set_description("computing longitudinal relaxation map...")
        longitudinal_relaxation_map, uni_img, min_img, hc_img, hco_img = inference.mp2rage_t1_fitting(img, ti, fa, tr, B0, sequence='flaws', t1strategy='hc')
        pbar.update(step)
        
        # export unified image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(uni_img, info, rootdir + '_uni', output_path + '_uni')        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(uni_img, info, rootdir + '_uni', output_path + '_uni')
            
        # export minimum image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(min_img, info, rootdir + '_min', output_path + '_min')        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(min_img, info, rootdir + '_min', output_path + '_min')
            
        # export hc image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(hc_img, info, rootdir + '_hc', output_path + '_hc')        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(hc_img, info, rootdir + '_hc', output_path + '_hc')
            
        # export hco image
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(hco_img, info, rootdir + '_hco', output_path + '_hco')        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(hco_img, info, rootdir + '_hco', output_path + '_hco')
        
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
    
    return longitudinal_relaxation_map, uni_img, min_img, hc_img, hco_img



