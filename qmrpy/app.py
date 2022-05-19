# -*- coding: utf-8 -*-
"""
External interface for algorithms.

Created on Thu Feb 10 17:33:53 2022

@author: Matteo Cencini
"""
import click


from qmrpy import alg


__all__ = ['longitudinal_relaxation', 'transverse_relaxation', 'transmit_field', 'helmholtz_ept', 'mp2rage_longitudinal_relaxation', 'flaws_longitudinal_relaxation']


@click.group()
def cli():
    pass


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Inversion Recovery SE data series in DICOM format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
def longitudinal_relaxation(input_path, output_path, save_dicom, save_nifti, mask_threshold):
    """
    Reconstruct quantitative T1 maps from Inversion Recovery Spin-Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * (1 - 2 *exp(-TI / T1))
    
    where TI is the Inversion Time.
    """
    alg.longitudinal_relaxation(input_path, output_path, save_dicom, save_nifti, mask_threshold)

    
# wrap into command line
cli.add_command(longitudinal_relaxation)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Multi-Echo SE or GRE data series in DICOM format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
def transverse_relaxation(input_path, output_path, save_dicom, save_nifti, mask_threshold):
    """
    Reconstruct quantitative T2 / T2* maps from Multi-Echo Spin-Echo / Gradient Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * exp(-TE / T2)
    
    where T2 is replaced by T2* for Gradient Echo data and TE is the Echo Time.
    """
    alg.transverse_relaxation(input_path, output_path, save_dicom, save_nifti, mask_threshold)

    
# wrap into command line
cli.add_command(transverse_relaxation)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Double Angle SE or GRE data series in DICOM format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
def transmit_field(input_path, output_path, save_dicom, save_nifti, mask_threshold):
    """
    Reconstruct quantitative B1+ maps from Double Angle Spin-Echo / Gradient Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * sin(theta)
    
    where theta is the nominal Flip Angle.
    """
    alg.transmit_field(input_path, output_path, save_dicom, save_nifti, mask_threshold)
    

# wrap into command line
cli.add_command(transmit_field)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of bSSFP data series in DICOM format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
@click.option( '--laplacian-kernel-width', default=0.05, show_default=True, help='Width of local parabolic phase fitting window (in [voxel])')
@click.option( '--fitting-threshold', default=5, show_default=True, help='Threshold to restrict local parabolic phase fitting [%].')
@click.option( '--median-filter-width', default=0.05, show_default=True, help='Width of adaptive median filter  (in [voxel])')
def helmholtz_ept(input_path, output_path, save_dicom, save_nifti, mask_threshold, laplacian_kernel_width, fitting_threshold, median_filter_width):
    """
    Reconstruct quantitative conductivity maps from bSSFP data.
    
    Use the following signal model:
        
        s(t) = - Nabla Phi / Phi / (omega0 * mu0)
    
    where Phi is bSSFP phase, omega0 is the larmor frequency and mu0 is the vacuum permittivity.
    """
    alg.helmholtz_ept(input_path, output_path, save_dicom, save_nifti, mask_threshold, laplacian_kernel_width, fitting_threshold, median_filter_width)
    
    
# wrap into command line
cli.add_command(helmholtz_ept)


@click.command()
@click.option( '--inversion-times', '-ti', multiple=True, required=True, help='MP2RAGE inversion times for the two volumes in [ms]')
@click.option( '--tr-flash', '-tr', multiple=True, required=True, help='FLASH readouts repetition time(s) [ms]')
@click.option( '--flip-angles', '-fa', multiple=True, required=True, help='FLASH readouts flip angle(s) [deg]')
@click.option( '--input-path', required=True, help='location on disk of MP2RAGE data series in DICOM format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
def mp2rage_longitudinal_relaxation(inversion_times, tr_flash, flip_angles, input_path, output_path, save_dicom, save_nifti):
    """
    Reconstruct quantitative T1 maps from MP2RAGEDATA data.
    """
    alg.mp2rage_longitudinal_relaxation(inversion_times, tr_flash, flip_angles, input_path, output_path, save_dicom, save_nifti)

    
# wrap into command line
cli.add_command(mp2rage_longitudinal_relaxation)


@click.command()
@click.option( '--inversion-times', '-ti', multiple=True, required=True, help='FLAWS inversion times for the two volumes in [ms]')
@click.option( '--flip-angles', '-fa', multiple=True, required=True, help='FLASH readouts flip angle(s) [deg]')
@click.option( '--tr-flash', '-tr', required=True, help='FLASH readouts repetition time(s) [ms]')
@click.option( '--input-path', required=True, help='location on disk of FLAWS data series in DICOM format')
@click.option( '--beta', default=0, show_default=True, help='Regularization parameter for background denoising')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--save-dicom', default=True, show_default=True, help='save reconstructed map as DICOM')
@click.option( '--save-nifti', default=True, show_default=True, help='save reconstructed map as NiFTI')
def flaws_longitudinal_relaxation(inversion_times, flip_angles, tr_flash , input_path, beta, output_path, save_dicom, save_nifti):
    """
    Reconstruct quantitative T1 maps from FLAWS data.
    """
    alg.flaws_longitudinal_relaxation(inversion_times, flip_angles, tr_flash, input_path, beta, output_path, save_dicom, save_nifti)

    
# wrap into command line
cli.add_command(flaws_longitudinal_relaxation)


