# -*- coding: utf-8 -*-
"""
External interface for algorithms.

Created on Thu Feb 10 17:33:53 2022

@author: Matteo Cencini
"""
from datetime import timedelta
import os
from time import time


import click
import numpy as np
from tqdm import tqdm


from qmrpy import io, inference, alg


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
    alg.longitudinal_relaxation(input_path, output_path, output_label, save_dicom, save_nifti, mask_threshold)

    
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
    alg.transverse_relaxation(input_path, output_path, output_label, save_dicom, save_nifti, mask_threshold)

    
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
    alg.transmit_field(input_path, output_path, output_label, save_dicom, save_nifti, mask_threshold)
    

# wrap into command line
cli.add_command(transmit_field)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of bSSFP data series in DICOM format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--output-label', default='conductivity_map', show_default=True, help='name of the output files')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
@click.option( '--laplacian-kernel-width', default=0.05, show_default=True, help='Width of local parabolic phase fitting window (in [voxel])')
@click.option( '--fitting-threshold', default=5, show_default=True, help='Threshold to restrict local parabolic phase fitting [%].')
@click.option( '--median-filter-width', default=0.05, show_default=True, help='Width of adaptive median filter  (in [voxel])')
def helmholtz_ept(input_path, output_path, output_label, save_dicom, save_nifti, mask_threshold, laplacian_kernel_width, fitting_threshold, median_filter_width):
    """
    Reconstruct quantitative conductivity maps from bSSFP data.
    
    Use the following signal model:
        
        s(t) = - Nabla Phi / Phi / (omega0 * mu0)
    
    where Phi is bSSFP phase, omega0 is the larmor frequency and mu0 is the vacuum permittivity.
    """
    alg.helmholtz_ept(input_path, output_path, output_label, save_dicom, save_nifti, mask_threshold, laplacian_kernel_width, fitting_threshold, median_filter_width)
    
    
# wrap into command line
cli.add_command(longitudinal_relaxation)

