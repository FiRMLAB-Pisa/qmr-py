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


import torch
import torch.nn.functional as F


from tqdm import tqdm


from qmrpy import io, inference


__all__ = ['longitudinal_relaxation', 'transverse_relaxation', 'transmit_field', 'helmholtz_ept']


def longitudinal_relaxation(input_path, output_path='./output', output_label='longitudinal_relaxation_map', save_dicom=False, save_nifti=False, mask_threshold=0.05):
    """
    Reconstruct quantitative T1 maps from Inversion Recovery Spin-Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * (1 - 2 *exp(-TI / T1))
    
    where TI is the Inversion Time.
    """
    # check input
    if input_path.endswith('/') or input_path.endswith('\\'):
        folders = os.listdir(input_path)
        input_path = [os.path.join(input_path, folder) for folder in folders]
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_dicom(input_path)
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
            io.write_dicom(longitudinal_relaxation_map, info, output_label, output_path)        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(longitudinal_relaxation_map, info, output_label, output_path)
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return longitudinal_relaxation_map

    
def transverse_relaxation(input_path, output_path='./output', output_label='transverse_relaxation_map', save_dicom=False, save_nifti=False, mask_threshold=0.05):
    """
    Reconstruct quantitative T2 / T2* maps from Multi-Echo Spin-Echo / Gradient Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * exp(-TE / T2)
    
    where T2 is replaced by T2* for Gradient Echo data and TE is the Echo Time.
    """
    # check input
    if input_path.endswith('/') or input_path.endswith('\\'):
        folders = os.listdir(input_path)
        input_path = [os.path.join(input_path, folder) for folder in folders]
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_dicom(input_path)
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
            io.write_dicom(transverse_relaxation_map, info, output_label, output_path)         
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(transverse_relaxation_map, info, output_label, output_path)
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return transverse_relaxation_map
    

def transmit_field(input_path, output_path='./output', output_label='b1_field_map', save_dicom=False, save_nifti=False, mask_threshold=0.05):
    """
    Reconstruct quantitative B1+ maps from Double Angle Spin-Echo / Gradient Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * sin(theta)
    
    where theta is the nominal Flip Angle.
    """
    # check input
    if input_path.endswith('/') or input_path.endswith('\\'):
        folders = os.listdir(input_path)
        input_path = [os.path.join(input_path, folder) for folder in folders]
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_dicom(input_path)
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
            io.write_dicom(transmit_field_map, info, output_label, output_path)          
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(transmit_field_map, info, output_label, output_path)
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return transmit_field_map
    

def helmholtz_ept(input_path, output_path='./output', output_label='conductivity_map', save_dicom=False, save_nifti=False, mask_threshold=0.05, gaussian_kernel_sigma=0.5, laplacian_kernel_width=20, fitting_threshold=20.0, median_filter_width=50):
    """
    Reconstruct quantitative conductivity maps from bSSFP data.
    
    Use the following signal model:
        
        s(t) = - Nabla Phi / Phi / (omega0 * mu0)
    
    where Phi is bSSFP phase, omega0 is the larmor frequency and mu0 is the vacuum permittivity.
    """
    # check input
    if input_path.endswith('/') or input_path.endswith('\\'):
        folders = os.listdir(input_path)
        input_path = [os.path.join(input_path, folder) for folder in folders]
            
    click.echo("starting processing...")
    t_start = time()
    
    # progress bar step size
    step = 1
    
    with tqdm(total=3) as pbar:
        pbar.set_description("loading input data...")
        img, info = io.read_dicom(input_path)
        resolution = np.array([float(info['template'][0].SliceThickness)] + [float(dr) for dr in info['template'][0].PixelSpacing]) * 1e-3
        omega0 = 2 * np.pi * 64.0e6 / 1.5 * float(info['template'][0].MagneticFieldStrength)
        
        # interpolate
        scale = np.array(img.shape[1:]) / 352
        re_img = torch.from_numpy(img.real)
        im_img = torch.from_numpy(img.imag)
        re_img = F.interpolate(re_img.unsqueeze(0).unsqueeze(0), size=[img.shape[0]] + [352, 352])[0][0].numpy()
        im_img = F.interpolate(im_img.unsqueeze(0).unsqueeze(0), size=[img.shape[0]] + [352, 352])[0][0].numpy()
        img = re_img + 1j * im_img
        resolution[1:] *= scale
        
        pbar.update(step)
        
        # mask data
        if mask_threshold > 0:
            mask = inference.utils.mask(img)
        else:
            mask = None
            
        pbar.set_description("computing conductivity map...")
        conductivity_map = inference.helmholtz_conductivity_fitting(img, resolution, omega0, 
                                                                    gaussian_kernel_sigma,
                                                                    laplacian_kernel_width, fitting_threshold, 
                                                                    median_filter_width, mask)
        pbar.update(step)
        
        if save_dicom:
            pbar.set_description("saving output dicom to disk...")
            io.write_dicom(conductivity_map, info, output_label, output_path)        
        if save_nifti:
            pbar.set_description("saving output nifti to disk...")
            io.write_nifti(conductivity_map, info, output_label, output_path)
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return conductivity_map



