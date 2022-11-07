# -*- coding: utf-8 -*-
"""
External interface for algorithms.

Created on Thu Feb 10 17:33:53 2022

@author: Matteo Cencini
"""
import click


from qmrpy import alg


__all__ = ['longitudinal_relaxation', 'transverse_relaxation', 'transmit_field', 'phase_based_laplacian_ept', 'water_based_ept', 'mp2rage_longitudinal_relaxation', 'flaws_longitudinal_relaxation']


@click.group()
def cli():
    pass


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Inversion Recovery SE data series in DICOM or NiFti format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
def longitudinal_relaxation(input_path, output_path, mask_threshold):
    """
    Reconstruct quantitative T1 maps from Inversion Recovery Spin-Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * (1 - 2 *exp(-TI / T1))
    
    where TI is the Inversion Time.
    """
    alg.longitudinal_relaxation(input_path, output_path, mask_threshold)

    
# wrap into command line
cli.add_command(longitudinal_relaxation)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Multi-Echo SE or GRE data series in DICOM or NiFti format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--skip-first-echo', default=False, show_default=True, help='If true, discard first echo')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
def transverse_relaxation(input_path, output_path, skip_first_echo, mask_threshold):
    """
    Reconstruct quantitative T2 / T2* maps from Multi-Echo Spin-Echo / Gradient Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * exp(-TE / T2)
    
    where T2 is replaced by T2* for Gradient Echo data and TE is the Echo Time.
    """
    alg.transverse_relaxation(input_path, output_path, skip_first_echo, mask_threshold)

    
# wrap into command line
cli.add_command(transverse_relaxation)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Double Angle SE or GRE data series in DICOM or NiFti format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
def transmit_field(input_path, output_path, mask_threshold):
    """
    Reconstruct quantitative B1+ maps from Double Angle Spin-Echo / Gradient Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * sin(theta)
    
    where theta is the nominal Flip Angle.
    """
    alg.transmit_field(input_path, output_path, mask_threshold)
    

# wrap into command line
cli.add_command(transmit_field)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of Double Angle SE or GRE data series in DICOM or NiFti format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data')
@click.option( '--fix-phase-along-z', default=False, show_default=True, help='If needed, fix phase wrap along z.')
def static_field(input_path, output_path, mask_threshold, fix_phase_along_z):
    """
    Reconstruct quantitative B0 maps from double echo Gradient Echo data.
    
    Use the following signal model:
        
        s(t) = M0 * exp(1i * gamma * B0)
        
    """
    alg.static_field(input_path, output_path, mask_threshold, fix_phase_along_z)
    

# wrap into command line
cli.add_command(static_field)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of bSSFP data series in DICOM or NiFti format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--segmentation-path', default=None, show_default=True, 
              help="""Use external tissue segmentation (stored in --segmentation-path=/path/to/segmentation/Segmentation.nii) to condition fit. 
                      Assumes segmentation is done with SPM and will search for /path/to/segmentation/cnSegmentation.nii'), with cn = c1, c2, c3,...
                      as described in SPM documentation and 'n' is given by the n_tissue_classes argument.""")
@click.option( '--n-tissue-classes', default=3, show_default=True, help='Number of SPM-based tissue masks to be loaded. By default, 1 is GM, 2 is WM and 3 is CSF.')
@click.option( '--merge-wm-csf', default=False, show_default=True, help='If true, will treat composite WM+CSF masks as a single mask.') 
@click.option( '--mask-threshold', default=0.05, show_default=True, help='Threshold to mask input data. Used if SPM segmentation is not provided.')
@click.option( '--gaussian-preprocessing-sigma', default=0.0, show_default=True, help='Sigma of Gaussian filter used to smooth input image before fitting. If 0, skip this step')
@click.option( '--gaussian-weight-sigma', default=0.45, show_default=True, help='Sigma of Gaussian kernel used to weight the input data according to magnitude difference during fitting')
@click.option( '--laplacian-kernel-width', default=16, show_default=True, help='Width of local parabolic phase fitting window (in [voxel])')
@click.option( '--laplacian-kernel-shape', default='ellipsoid', show_default=True, help='Shape of the laplacian kernel (options are cross, cuboid and sigmoid)')
@click.option( '--median-filter-width', default=0.05, show_default=True, help='Width of adaptive median filter  (in [voxel])')
@click.option( '--fix-phase-along-z', default=False, show_default=True, help='If needed, fix phase wrap along z.')
def phase_based_laplacian_ept(input_path, output_path, segmentation_path, n_tissue_classes, merge_wm_csf, mask_threshold, 
                              gaussian_preprocessing_sigma, gaussian_weight_sigma, 
                              laplacian_kernel_width, laplacian_kernel_shape, 
                              median_filter_width, fix_phase_along_z):
    """
    Reconstruct quantitative conductivity maps from bSSFP data.
    
    Use the following signal model:
        
        s(t) = - Nabla Phi / Phi / (omega0 * mu0)
    
    where Phi is bSSFP phase, omega0 is the larmor frequency and mu0 is the vacuum permittivity.
    """
    alg.phase_based_laplacian_ept(input_path, output_path, segmentation_path, n_tissue_classes, merge_wm_csf, mask_threshold, 
                                  gaussian_preprocessing_sigma, gaussian_weight_sigma, 
                                  laplacian_kernel_width, laplacian_kernel_shape, 
                                  median_filter_width, fix_phase_along_z)
    
    
# wrap into command line
cli.add_command(phase_based_laplacian_ept)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of bSSFP data series in DICOM or NiFti format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--anatomic-region', default='brain', show_default=True, help='Anatomical region corresponding to acquisition')
@click.option( '--units', default='ms', show_default='ms', help='Units of the input T1 map')
@click.option( '--t1-index', default=0, show_default=True, help='For multi-parametric input, index corresponding to T1 map')
def water_based_ept(input_path, output_path, anatomic_region, units, t1_index):
    """
    Reconstruct electric properties maps from quantitative T1 map.
    
    First, a water concentration map is estimated from the T1 map according to [1]:
        
        - 1/water_map = A + B/t1_map

    Then, water concentration map is used to compute electric properties,    
    assuming the following relations between water concentration and tissue 
    conductivity and relative permittivity [2]:
        
        - \sigma = c_1 + c_2 * exp(c_3 * W)
        - \epsilon_r = p_1 * W^2 + p_2 * W + p3
    
    where sigma is the conductivityty, W the percentage
    water concentration and c_i the model coefficients.
    """
    alg.water_based_ept(input_path, output_path, anatomic_region, units, t1_index)
    
    
# wrap into command line
cli.add_command(water_based_ept)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of MP2RAGE data series in DICOM or NiFti format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--inversion-times', '-ti', multiple=True, required=True, help='MP2RAGE inversion times for the two volumes in [ms]')
@click.option( '--tr-flash', '-tr', multiple=True, required=True, help='FLASH readouts repetition time(s) [ms]')
@click.option( '--flip-angles', '-fa', multiple=True, required=True, help='FLASH readouts flip angle(s) [deg]')
@click.option( '--inversion-efficiency', default=1.0, show_default=True, help='Inversion efficiency for imperfect pulse')
@click.option( '--beta', default=0, show_default=True, help='Regularization parameter for background denoising')
def mp2rage_longitudinal_relaxation(input_path, output_path, inversion_times, tr_flash, flip_angles, inversion_efficiency, beta):
    """
    Reconstruct quantitative T1 maps from MP2RAGEDATA data.
    """
    alg.mp2rage_longitudinal_relaxation(input_path, output_path, inversion_times, tr_flash, flip_angles, inversion_efficiency, beta)

    
# wrap into command line
cli.add_command(mp2rage_longitudinal_relaxation)


@click.command()
@click.option( '--input-path', required=True, help='location on disk of FLAWS data series in DICOM or NiFti format')
@click.option( '--output-path', default='./output', show_default=True, help='path for the stored output')
@click.option( '--inversion-times', '-ti', multiple=True, required=True, help='FLAWS inversion times for the two volumes in [ms]')
@click.option( '--flip-angles', '-fa', multiple=True, required=True, help='FLASH readouts flip angle(s) [deg]')
@click.option( '--tr-flash', '-tr', required=True, help='FLASH readouts repetition time(s) [ms]')
@click.option( '--inversion-efficiency', default=1.0, show_default=True, help='Inversion efficiency for imperfect pulse')
@click.option( '--beta', default=0, show_default=True, help='Regularization parameter for background denoising')
def flaws_longitudinal_relaxation(input_path, output_path, inversion_times, flip_angles, tr_flash, inversion_efficiency, beta):
    """
    Reconstruct quantitative T1 maps from FLAWS data.
    """
    alg.flaws_longitudinal_relaxation(input_path, output_path, inversion_times, flip_angles, tr_flash, inversion_efficiency, beta)

    
# wrap into command line
cli.add_command(flaws_longitudinal_relaxation)


