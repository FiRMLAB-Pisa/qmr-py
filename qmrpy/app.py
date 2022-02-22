# -*- coding: utf-8 -*-
"""
External interface

Created on Thu Feb 10 17:33:53 2022

@author: Matteo Cencini
"""
from datetime import timedelta
import os
from time import time


import click
from tqdm import tqdm


from qmrpy import io, inference


__all__ = ['longitudinal_relaxation', 'transverse_relaxation', 'transmit_field']


@click.group()
def cli():
    pass


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Inversion Recovery SE data series in DICOM format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--output-label', default='longitudinal_relaxation_map', show_default=True, help='name of the output files')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
def longitudinal_relaxation(input_path, output_path, output_label, save_dicom, save_nifti, mask_threshold):
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
        
        pbar.set_description("saving output to disk...")
        if save_dicom:
            io.write_dicom(longitudinal_relaxation_map, info, output_label, output_path)        
        if save_nifti:
            io.write_nifti(longitudinal_relaxation_map, info, output_label, output_path)
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return longitudinal_relaxation_map

    
# wrap into command line
cli.add_command(longitudinal_relaxation)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Multi-Echo SE or GRE data series in DICOM format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--output-label', default='transverse_relaxation_map', show_default=True, help='name of the output files')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
def transverse_relaxation(input_path, output_path, output_label, save_dicom, save_nifti, mask_threshold):
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
        
        pbar.set_description("saving output to disk...")
        if save_dicom:
            io.write_dicom(transverse_relaxation_map, info, output_label, output_path)         
        if save_nifti:
            io.write_nifti(transverse_relaxation_map, info, output_label, output_path)
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return transverse_relaxation_map
    

# wrap into command line
cli.add_command(transverse_relaxation)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Double Angle SE or GRE data series in DICOM format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--output-label', default='b1_field_map', show_default=True, help='name of the output files')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
def transmit_field(input_path, output_path, output_label, save_dicom, save_nifti, mask_threshold):
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
        
        pbar.set_description("saving output to disk...")
        if save_dicom:
            io.write_dicom(transmit_field_map, info, output_label, output_path)          
        if save_nifti:
            io.write_nifti(transmit_field_map, info, output_label, output_path)
        pbar.update(step)
        
    t_end = time()
    click.echo("reconstruction done! Elapsed time: " + str(timedelta(seconds=(t_end-t_start))))
    
    return transmit_field_map
    

# wrap into command line
cli.add_command(transmit_field)

