# -*- coding: utf-8 -*-
"""
Plotting utils.

Created on Thu Feb 10 19:06:26 2022

@author: Matteo Cencini
"""
import matplotlib.pyplot as plt
import numpy as np

__all__ = ['show_images']


def show_images(qmap, slice_idx, title=None, cmap='gray'):
    """
    Display quantitative map.
    
    Args:
        qmap (ndarray): input quantitative map of shape (nz, ny, nx)
        slice_idx (int): slice to be shown.
        cmap (str): color map.
        title (str): Plot title.
    """
    # preserve input
    qmap = qmap.copy()
        
    # be sure image is cubic
    max_dim = np.max(qmap.shape)
    pad_sz = max_dim - np.array(qmap.shape)
    pad_sz = [[0, pad_sz[n]] for n in range(len(pad_sz))]
    qmap = np.pad(qmap, pad_sz)
        
    if isinstance(slice_idx, list) or isinstance(slice_idx, tuple):
        
        assert len(slice_idx) == 3, "Either provide a single slice index or an index for each dimension!"
        
        x1 = qmap[slice_idx[0]]
        x2 = qmap[:,slice_idx[1]]
        x3 = qmap[:,:,slice_idx[2]]
        x = np.concatenate((x1, x2, x3), axis=1)
        
    else:
        x = qmap[slice_idx]
        
    # get colorscale
    try:
        vmin = 0.5 * np.percentile(x[x < 0], 99)
    except:
        vmin = 0      
    try:
        vmax = 0.5 * np.percentile(x[x > 0], 99)
    except:
        vmax = 0
      
    # show
    plt.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='lanczos'), plt.axis('off'), plt.colorbar()
    
    if title is not None:
        plt.title(title)
