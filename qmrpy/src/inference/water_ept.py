#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This lightweight library allow to calculate electric properties maps
starting from quantitative t1 maps. This approach is inspired by:
    
    1. Michel, E., Hernandez, D. and Lee, S.Y. (2017), 
    Electrical conductivity and permittivity maps of brain tissues 
    derived from water content based on T1-weighted acquisition. 
    Magn. Reson. Med., 77: 1094-1103. 
    https://doi.org/10.1002/mrm.26193
    
Differently from [1], here we directly use quantitative T1 maps 
(derived from fast T1 map technique such as MR Fingerprinting, MP2RAGE or VFA)
to retrieve water content maps of the tissue, which are used to compute 
electric properties maps.

Other references:
    2. Fatouros PP, Marmarou A. 
    Use of magnetic resonance imaging for in vivo measurements 
    of water content in human brain: method and normal values. 
    J Neurosurg. 1999 Jan;90(1):109-15. 
    doi: 10.3171/jns.1999.90.1.0109. PMID: 10413163.
    
Created on Mon Sep 27 15:11:56 2021

@author: Matteo Cencini
@mail: matteo.cencini@gmail.com
"""
import numpy as np
import scipy.optimize as opt


__all__ = ['water_ept_fitting']


#%% Physical Constants
# Speed of light [m/s].
c0 = 299792458.0

# Vacuum permeability [H/m].
mu0 = 4.0e-7 * np.pi

# Vacuum permeability [F/m].
eps0 = 1.0 / mu0 / c0**2

# gyromagnetic factor [Hz/T]
gamma = 64e6 / 1.5

# T1 to water content converter ([1] for A,B @3T; [2] for A,B @1.5T, @7T: computed from literature T1 values in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6590660/pdf/HBM-40-1786.pdf)
water_lut = {'1.5': {'A': 0.935, 'B': 0.341},
             '3.0': {'A': 0.89, 'B': 0.50},
             '7.0': {'A': 0.93, 'B': 0.62}}

# Tabulated water contents [%] (source: https://www.sciencedirect.com/science/article/pii/S1053811922001434)
literature_water = {'wm': 69.0, 'gm': 80.0, 'csf': 100.0} # our submission
# literature_water = {'wm': 70.0, 'gm': 81.0, 'csf': 100.0} # average from https://doi.org/10.1016/j.neuroimage.2022.119014 (table 2)
# literature_water = {'wm': 69.57, 'gm': 83.41, 'csf': 98.8} # Michel et al.

# 4 Cole-Cole model parameters (N De Geeter et al 2012 Phys. Med. Biol. 57 2169)
brain_params = {'wm': {'epsInf': 4.0,
                       'deps': np.array([32.0, 100.0, 4e4, 3.5e7]),
                       'tau': np.array([7.96e-12, 7.96e-9, 53.05e-6, 7.958e-3]),
                       'alpha': np.array([0.10, 0.10, 0.30, 0.02]),
                       'sigma': 0.02},
                 'gm': {'epsInf': 4.0,
                        'deps': np.array([45.0, 400.0, 2e5, 4.5e7]) ,
                        'tau': np.array([7.96e-12, 15.92e-9, 106.10e-6, 5.305e-3]),
                        'alpha': np.array([0.10, 0.15, 0.22, 0.00]),
                        'sigma': 0.02},
                'csf': {'epsInf': 4.0,
                        'deps': np.array([65.0, 40.0, 00.0, 0.0]),
                        'tau': np.array([7.958e-12, 1.592e-9, 0.0, 0.0]),
                        'alpha': np.array([0.10, 0.0, 0.0, 0.0]),
                        'sigma': 2.0}}

cole_cole_model_params = {'brain': brain_params}
 

#%% Functions   
def water_ept_fitting(input: np.ndarray, field_strength: float, anatomic_region: str = 'brain') -> np.ndarray:
    """ Compute electric properties maps from quantitative T1 map.
    
    First, a water concentration map is estimated from the T1 map according to [1]:
        
        - 1/water_map = A + B/t1_map

    Then, water concentration map is used to compute electric properties,    
    assuming the following relations between water concentration and tissue 
    conductivity and relative permittivity [2]:
        
        - \sigma = c_1 + c_2 * exp(c_3 * W)
        - \epsilon_r = p_1 * W^2 + p_2 * W + p3
    
    where sigma is the conductivityty, W the percentage
    water concentration and c_i the model coefficients [1].
    
    Args:
        input (ndarray): quantitative T1 map of the tissues (units: [ms]).
        field_strength (float): B0 field strength (units: [T]).
        anatomic_region (str): anatomy of interest (default: brain).
    
    Returns:
        conductivity (ndarray): conductivity map of the tissues (units: [S/m])
        permittivity (ndarray): relative permittivity map of the tissues.

    References:
        1. Fatouros PP, Marmarou A. 
        Use of magnetic resonance imaging for in vivo measurements 
        of water content in human brain: method and normal values. 
        J Neurosurg. 1999 Jan;90(1):109-15. 
        doi: 10.3171/jns.1999.90.1.0109. PMID: 10413163.
        
        2. Michel, E., Hernandez, D. and Lee, S.Y. (2017), 
        Electrical conductivity and permittivity maps of brain tissues 
        derived from water content based on T1-weighted acquisition. 
        Magn. Reson. Med., 77: 1094-1103. 
        https://doi.org/10.1002/mrm.261    
    """      
    # calibrate conductivity curve
    _c = _calibrate_conductivity(field_strength, anatomic_region)
    
    # calibrate permittivity curve
    _p = _calibrate_permittivity(field_strength, anatomic_region)
            
    # get water map
    water_map = _convert_t1_map_to_water_map(input, field_strength)
    
    # get conductivity
    conductivity = _calculate_conductivity(water_map, _c[0], _c[1], _c[2])
    
    # get permittivity
    permittivity = _calculate_permittivity(water_map, _p[0], _p[1], _p[2])
    
    return conductivity, permittivity

      
def _calculate_conductivity(water, c1, c2, c3):
    """ Water - conductivity lookup table.
    
    Assume the following relation between water concentration and tissue 
    conductivity [1]:
        
        \sigma = c_1 + c_2 * exp(c_3 * W)
    
    where sigma is the conductivity, W the percentage
    water concentration and c_i the model coefficients [1].
    
    Args:
        water (ndarray): percentage water concentration (units: [%]).
        c1, c2, c3 (float): water - conductivity model coefficients.
    
    Returns:
        sigma (ndarray): tissue conductivity. 
    """
    return c1 + c2*np.exp(c3 * water)


def _calibrate_conductivity(field_strength, anatomic_region):
    """ Calibrate water - conductivity lookup table.
    
    Assume the following relation between water concentration and tissue 
    conductivity [1]:
        
        \sigma = c_1 + c_2 * exp(c_3 * W)
    
    where sigma is the conductivityty, W the percentage
    water concentration and c_i the model coefficients [1].
    
    Args:
        field_strength (float): B0 field strength (units: [T])..
        anatomic_region (str): anatomy of interest (default: brain).
    
    Returns:
        c1, c2, c3 (float): water - conductivity model coefficients.
    """
    # get electrical properties
    conductivity_dict = _get_complex_dielectric_properties(field_strength, anatomic_region)['conductivity']
    
    # get array
    water = []
    conductivity = []
    for tissue in literature_water.keys():
        water.append(literature_water[tissue]) 
        conductivity.append(conductivity_dict[tissue])
        
    # convert to ndarray
    water = np.array(water)
    conductivity = np.array(conductivity)
    
    # do fit
    cout, _ = opt.curve_fit(_calculate_conductivity, water, conductivity, p0=[0.268, 1.526e-5, 11.852e-2])
    
    # unpack
    c1, c2, c3 = cout[0], cout[1], cout[2] 
    
    return c1, c2, c3


def _calculate_permittivity(water, p1, p2, p3):
    """ Water - permittivity lookup table.
    
    Assume the following relation between water concentration and tissue 
    relative permittivity [1]:
        
        \epsilon_r = p_1 * W^2 + p_2 * W + p3
    
    where epsilon_r is the relative permittivity, W the percentage
    water concentration and p_i the model coefficients [1].
    
    Args:
        water (ndarray): percentage water concentration (units: [%]).
        p1, p2, p3 (float): water - permittivity model coefficients.
    
    Returns:
        epsilon_r (ndarray): tissue relative permittivity. 
    """
    return p1 * water**2 + p2 * water + p3


def _calibrate_permittivity(field_strength, anatomic_region):
    """ Calibrate water - permittivity lookup table.
    
    Assume the following relation between water concentration and tissue 
    dielectric properties [1]:
        
        \epsilon_r = p_1 * W^2 + p_2 * W + p3
    
    where epsilon_r is the relative permittivity, W the percentage
    water concentration and p_i the model coefficients [1].
    
    Args:
        field_strength (float): B0 field strength (units: [T])..
        anatomic_region (str): anatomy of interest (default: brain).
    
    Returns:
        p1, p2, p3 (float): water - permittivity model coefficients.
    """
    # get electrical properties
    permittivity_dict = _get_complex_dielectric_properties(field_strength, anatomic_region)['permittivity']
    
    # get array
    water = []
    permittivity = []
    for tissue in literature_water.keys():
        water.append(literature_water[tissue]) 
        permittivity.append(permittivity_dict[tissue])
        
    # convert to ndarray
    water = np.array(water)
    permittivity = np.array(permittivity)
    
    # do fit
    pout, _ = opt.curve_fit(_calculate_permittivity, water, permittivity, p0=[-287e-4, 591e-2, -220])
    
    # unpack
    p1, p2, p3 = pout[0], pout[1], pout[2] 
    
    return p1, p2, p3


def _get_complex_dielectric_properties(field_strength, anatomic_region):
    """ Calculate theoretical complex dielectric properties.
    
    Assume 4 Cole-Cole model for dielectric properties.
    
    Args:
        field_strength (float): B0 field strength (units: [T])..
        anatomic_region (str): anatomy of interest (default: brain).
    
    Returns:
        dielectric_properties (dict): dictionary with the following fields:
            
                                        - conductivity: representative conductivity 
                                                        values for the selected
                                                        anatomy and field strength 
                                                        (units: [S/m]).
                                                 
                                        - permittivity: representative permittivity 
                                                        values for the selected
                                                        anatomy and field strength.
    """
    # get Larmor frequency [rad/s]
    omega = 2 * np.pi * field_strength * gamma
    
    # initialize output dict
    dielectric_properties = {'permittivity': {}, 'conductivity': {}}
    
    # get params
    try:
        params = cole_cole_model_params[anatomic_region]
        
    except:
        print('Not implemented!')
   
    # loop over representative tissues
    for tissue in params.keys():
        tmp = params[tissue]
        epsInf = tmp['epsInf']
        eps = tmp['deps'] / (1 + (1j * omega * tmp['tau'])**(1-tmp['alpha']))
        sigma = tmp['sigma'] / (1j * omega * eps0)
        
        # get complex electical properties
        complex_ep = epsInf + eps.sum() + sigma
        
        # assign
        dielectric_properties['permittivity'][tissue] = complex_ep.real
        dielectric_properties['conductivity'][tissue] = -complex_ep.imag * omega * eps0
        
    return dielectric_properties
    

def _convert_t1_map_to_water_map(t1map, field_strength=1.5):
    """ Convert T1 map [ms] to water content map [%].
    
    Assume the following relation between T1 and water concentration [2]:
        
        - 1/water_map = A + B/t1_map
        
    Args:
        t1map (ndarray): quantitative T1 map of the tissues (units: [ms]).
        field_strength (float): B0 field strength (units: [T]).
    
    Returns:
        water_map (ndarray): water content map of the tissues (percentage).
    """
    # get A, B as [1,2]
    try:
        A = water_lut[str(field_strength)]['A']
        B = water_lut[str(field_strength)]['B']
    except:
        print(f'Field strength (={field_strength} T) not available!')

    # calculate water map as [2]
    water_map_inv = A + B / (t1map / 1000.0) # ms -> s
    water_map = 1 / water_map_inv
    
    # clean NaN and convert to [%]
    water_map = 100 * np.nan_to_num(water_map)
    
    return water_map