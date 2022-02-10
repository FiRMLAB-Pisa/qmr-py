# -*- coding: utf-8 -*-
"""
Unit tests for parameter inference routines.

Created on Thu Feb 10 16:50:12 2022

@author: Matteo Cencini
"""
import pytest


import numpy as np


from qmrpy.src.inference import inversion_recovery, multiecho


@pytest.mark.parametrize("nslices", [1, 2])
def test_ir_t1_mapping(nslices):
    
    # set up parameters
    ti = np.array([50, 250, 400, 700, 1200, 2000, 3000], dtype=np.float32)
    t1true = 1000.0
    
    # generate test signal
    signal = (1 - 2 * np.exp(-ti / t1true))[:, None, None, None] * np.ones((nslices, 2, 2), dtype=np.float32)
    
    # measure t1
    t1measured = inversion_recovery.ir_se_t1_fitting(signal, ti)
    
    assert np.allclose(t1measured, t1true)
    

@pytest.mark.parametrize("nslices", [1, 2])
def test_multiecho_transverse_mapping(nslices):
    
    # set up parameters
    te = np.array([10, 30, 80, 130, 180, 300, 500], dtype=np.float32)
    t2true = 80.0
    
    # generate test signal
    signal = np.exp(-te / t2true)[:, None, None, None] * np.ones((nslices, 2, 2), dtype=np.float32)
    
    # measure t1
    t2measured = multiecho.me_transverse_relaxation_fitting(signal, te)
    
    assert np.allclose(t2measured, t2true)
    

