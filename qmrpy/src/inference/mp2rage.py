# -*- coding: utf-8 -*-
"""
Fitting routines and signal model for MP2RAGE and FLAWS data.

Created on Tue May 17 14:18:49 2022

@author: Matteo Cencini; Marta Lancione
"""
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

import numpy as np


__all__ = ['mp2rage_t1_fitting']


def mp2rage_t1_fitting(input, ti, fa, tr_flash, tr_mp2rage, B0, beta=0, inversion_efficiency=1.0, Rz=1, sequence='mp2rage'):
    """
    Calculate t2/t2* maps from multiecho spin echo / gradient echo data.
    
    Args:
        input (ndarray): complex data of size (2, nz, ny, nx)
        ti (ndarray): array of inversion times [ms].
        fa (float or ndarray): FLASH flip angles (can be different for the two blocks) [deg]-
        tr_flash (float or ndarray): FLASH readout repetition times (can be different for the two blocks) in [ms].
        beta (float): regularization parameters for de-noise in low SNR regions.
        B0 (float): Static field strength [T].
        inversion_efficiency (float): actual inversion efficiency (1 for perfect inversion).
        sequence (str): if 'mp2rage', returns 'unified t1w' and 't1 map'; for 'flaws', returns multiple contrasts (see paper).
        t1strategy (str): if 'uni', uses unified t1w for t1 estimation; if 'hc', uses hc image as in FLAWS paper.
        
    Returns:
        t1map (ndarray): T1 map of size (nz, ny, nx) in [ms].
        uni (ndarray): Unified image (T1w for MP2RAGE).
        min (ndarray, optional): WM and CSF suppressed image for FLAWS.
        hc (ndarray, optional): High-Contrast image from FLAWS.
        hco (ndarray, optional): Reversed High-Contrast image from FLAWS (MP2RAGEuni-like).
    """    
    # preserve input
    input = np.abs(input.copy())
    nslices = input.shape[0]
    nslices = np.ceil(nslices / Rz)
    nslices = int(nslices)
    
    # get unified image
    uni_img = -MP2RAGE.uni_image(input[0], input[1], beta)
        
    # get additional images
    if sequence == 'mp2rage':
        t1map = MP2RAGE.fit_t1(uni_img, ti, fa, tr_flash, tr_mp2rage, nslices, B0, inversion_efficiency, strategy='uni')
        return t1map, uni_img
    
    elif sequence == 'flaws':
        _, min_img = MP2RAGE().min_image(input[0], input[1], uni_img)
        _, hc_img = MP2RAGE().hc_image(input[0], input[1], uni_img)
        _, hco_img = MP2RAGE().hco_image(input[0], input[1], uni_img)
        
        # actual t1 fit
        t1map = MP2RAGE.fit_t1(hc_img, ti, fa, tr_flash, tr_mp2rage, nslices, B0, inversion_efficiency, strategy='hc')
            
        return t1map, uni_img, min_img, hc_img, hco_img

        
class MP2RAGE:
    """
    MP2RAGE related routines.
    """
    @staticmethod
    def uni_image(inv1_img: np.ndarray, inv2_img: np.ndarray, beta: float = 0) -> np.ndarray:
        """
        Default Unified MP2RAGE image as in Marques et al.
        
        Args:
            inv1_img: first Inversion Pulse volume.
            inv2_img: second Inversion Pulse volume.
            beta: de-noising parameter for low SNR regions.
            
        Returns:
            uni_image: unified image (T1w for standard MP2RAGE).
        """
        return ((inv1_img.conj() * inv2_img).real - beta) / ((np.abs(inv1_img)**2 + np.abs(inv2_img)**2) + 2 * beta)

    @staticmethod
    def min_image(inv1_img: np.ndarray, inv2_img: np.ndarray, uni_image: np.ndarray = None):
        """
        Minimum image for both WM and CSF suppression in FLAWS.
        
        Args:
            inv1_img: first Inversion Pulse volume.
            inv2_img: second Inversion Pulse volume.
            uni_image: unified image.
            
        Returns:
            min_image: minimum image.
            min_image_fullrange: range-restored minimum image.
        """  
        # get magnitude images
        S1 = np.abs(inv1_img)
        S2 = np.abs(inv2_img)
        
        tmp = np.stack((S1, S2), axis=0)
        min_image = tmp.min(axis=0) / (S1 + S2)
        
        # restore full range
        if uni_image is not None:
            case1 = ((uni_image >= 0) * (S1 < S2)).astype(bool)
            case2 = ((uni_image > 0) * (S1 > S2)).astype(bool)
            
            min_image_fullrange = min_image.copy()
            min_image_fullrange[case1] = -min_image[case1]
            min_image_fullrange[case2] = -min_image[case2]
        
            return min_image, min_image_fullrange
        
        return min_image, None
        
    @staticmethod
    def hc_image(inv1_img: np.ndarray, inv2_img: np.ndarray, uni_image: np.ndarray = None):
        """
        High Contrast image for FLAWS.
        
        Args:
            inv1_img: first Inversion Pulse volume.
            inv2_img: second Inversion Pulse volume.
            uni_image: unified image.
            
        Returns:
            hc_image: high-contrast image.
            hc_image_fullrange: range-restored minimum image.
        """  
        # get magnitude images
        S1 = np.abs(inv1_img)
        S2 = np.abs(inv2_img)
        
        hc_image = (S1 - S2) / (S1 + S2)
        
        # restore full range
        if uni_image is not None:
            case1 = ((uni_image >= 0) * (S1 < S2)).astype(bool)
            case2 = ((uni_image > 0) * (S1 > S2)).astype(bool)
            
            hc_image_fullrange = hc_image.copy()
            hc_image_fullrange[case1] = -2 - hc_image[case1]
            hc_image_fullrange[case2] = 2 - hc_image[case2]
            
            return hc_image, hc_image_fullrange
            
        return hc_image, None
    
    @staticmethod
    def hco_image(inv1_img: np.ndarray, inv2_img: np.ndarray, uni_image: np.ndarray = None):
        """
        High Contrast image to get MP2RAGEuni-like contrast in FLAWS.
        
        Args:
            inv1_img: first Inversion Pulse volume.
            inv2_img: second Inversion Pulse volume.
            uni_image: unified image.
            
        Returns:
            hco_image: high-contrast image.
            hco_image_fullrange: range-restored minimum image.
        """
        # get magnitude images
        S1 = np.abs(inv1_img)
        S2 = np.abs(inv2_img)
        
        hco_image = (S2 - S1) / (S1 + S2)
        
        if uni_image is not None:
            # restore full range
            case1 = ((uni_image >= 0) * (S1 < S2)).astype(bool)
            case2 = ((uni_image > 0) * (S1 > S2)).astype(bool)
            
            hco_image_fullrange = hco_image.copy()
            hco_image_fullrange[case1] = 2 - hco_image[case1]
            hco_image_fullrange[case2] = -2 - hco_image[case2]
            
            return hco_image, hco_image_fullrange
        
        return hco_image, None
    
    @staticmethod
    def fit_t1(image, ti, fa, tr_flash, tr_mp2rage, nslices, B0, inversion_efficiency=0.96, strategy='uni'):

        # get lookup table
        intensity, t1_grid = MP2RAGE.lookup_table(ti, fa, tr_flash, tr_mp2rage, nslices, B0, inversion_efficiency, strategy)
        
        t1_grid = t1_grid[np.argsort(t1_grid)]
        intensity = np.sort(intensity)

        # perform inference
        t1map = np.interp(image.flatten(), intensity, t1_grid)
        
        # clean-up result
        t1map = np.nan_to_num(t1map.reshape(image.shape))

        # convert to milliseconds
        t1map *= 1000

        return t1map
    
    @staticmethod
    def signal_model(ti, fa, tr_flash, tr_mp2rage, nslices, T1s, B0, inversion_efficiency):
    
        # handle input parameters
        ti = np.atleast_1d(ti) * 1e-3 # [ms] -> s
        tr_mp2rage *= 1e-3 # [ms] -> [s]
        tr_flash *= 1e-3 # [ms] -> [s]

   
        # flip angle
        fa = np.atleast_1d(fa)    
        fa = np.deg2rad(fa)
    
        if len(fa) != 2:
            fa = np.repeat(fa, 2)
        
        # number of slices        
        nslices = np.atleast_1d(nslices)
    
        if len(nslices) == 2:
            nZ_bef = nslices[0]
            nZ_aft = nslices[1]
            nslices = sum(nslices);
    
        elif len(nslices)==1:
            nZ_bef = nslices / 2
            nZ_aft = nslices / 2
    
        # calculate operators
        E_1 = np.exp(-tr_flash / T1s)
        TA_bef = nZ_bef * tr_flash
        TA_aft = nZ_aft * tr_flash

        TD = np.zeros(3)
        TD[0] = ti[0] - TA_bef      
        TD[1] = ti[1] - ti[0] - (TA_aft + TA_bef)
        TD[2] = tr_mp2rage - ti[1] - TA_aft

        E_TD = np.exp(-TD[:, None] / T1s[None, :])

        cosalfaE1 = np.cos(fa)[:, None] * E_1    
        sinalfa = np.sin(fa)[:, None]
    
        MZsteadystate = 1. / (1 + inversion_efficiency * (np.prod(cosalfaE1, axis=0))**(nslices) * np.prod(E_TD, axis=0))
        
        MZsteadystatenumerator = (1 - E_TD[0])
        MZsteadystatenumerator *= cosalfaE1[0]**nslices + (1 - E_1) * (1 - (cosalfaE1[0])**nslices) / (1 - cosalfaE1[0])        
        MZsteadystatenumerator = MZsteadystatenumerator * E_TD[1] + (1 - E_TD[1])
        MZsteadystatenumerator *= cosalfaE1[1]**nslices + (1 - E_1) * (1 - (cosalfaE1[1])**nslices) / (1 - cosalfaE1[1])        
        MZsteadystatenumerator = MZsteadystatenumerator * E_TD[2] + (1 - E_TD[2])
    
        MZsteadystate = MZsteadystate * MZsteadystatenumerator
        
        # get signal
        signal = np.zeros((2, len(T1s)), dtype=np.float32)
    
        # signal for first volume
        temp = (-inversion_efficiency * MZsteadystate * E_TD[0] + (1-E_TD[0])) * (cosalfaE1[0])**(nZ_bef) + (1 - E_1) * (1 - (cosalfaE1[0])**(nZ_bef)) / (1 - (cosalfaE1[0]))
        signal[0] = sinalfa[0] * temp
    
        # signal for second volume
        temp *= (cosalfaE1[1])**(nZ_aft) + (1 - E_1[1]) * (1 - (cosalfaE1[1])**(nZ_aft))  / (1 - (cosalfaE1[1]))
        temp = (temp * E_TD[1] + (1 - E_TD[1])) * (cosalfaE1[1])**(nZ_bef) + (1 - E_1) * (1 - (cosalfaE1[1])**(nZ_bef)) / (1 - (cosalfaE1[1]))
        signal[1] = sinalfa[1] * temp
    
        return signal        

    @staticmethod
    def lookup_table(ti, fa, tr_flash, tr_mp2rage, nslices, B0, inversion_efficiency, strategy):
    
        # get parameters search grid
        t1_grid = np.arange(0.05, 5.05, 0.05)
        signal = MP2RAGE.signal_model(ti, fa, tr_flash, tr_mp2rage, nslices, t1_grid, B0, inversion_efficiency)
            
        # get unified signal
        intensity = (signal[0] * signal[1].conj()).real / (np.abs(signal[0]**2) + np.abs(signal[1]**2))
        uni_signal = intensity
        _, hc_signal = MP2RAGE.hc_image(signal[0], signal[1], uni_signal)
        
        if strategy != 'hc':
            # get monotonic part
            minindex = np.argmax(np.abs(hc_signal))
            maxindex = np.argmin(np.abs(hc_signal))
                    
            hc_signal = hc_signal[minindex:maxindex+1]
            t1_grid = t1_grid[minindex:maxindex+1]
    
        return hc_signal, t1_grid
