# -*- coding: utf-8 -*-
"""
Utility routines for DICOM files loading and sorting.

Created on Thu Jan 27 13:30:28 2022

@author: Matteo Cencini
"""
import copy
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
import os


import numpy as np


import pydicom


def _load_dcm(dicomdir):
    """    
    load list of dcm files and automatically gather real/imag or magnitude/phase to complex image.
    """
    # get list of dcm files
    dcm_paths = _get_dicom_paths(dicomdir)
    
    # check inside paths for subfolders
    dcm_paths = _probe_dicom_paths(dcm_paths)
        
    # make pool of workers
    pool = ThreadPool(multiprocessing.cpu_count())

    # each thread load a dicom
    dsets = pool.map(_dcmread, dcm_paths)
    
    # cloose pool and wait finish   
    pool.close()
    pool.join()
        
    # filter None
    dsets = [dset for dset in dsets if dset is not None]
    
    # cast image to complex
    image, dsets = _cast_to_complex(dsets)
    
    return image, dsets

    
def _get_dicom_paths(dicomdir):
    """
    Get path to all DICOMs in a directory or a list of directories.
    """
    # get all files in dicom dir
    if isinstance(dicomdir, (tuple, list)):
        dcm_paths = _get_full_path(dicomdir[0], sorted(os.listdir(dicomdir[0])))
        for d in range(1, len(dicomdir)):
            dcm_paths += _get_full_path(dicomdir[d], sorted(os.listdir(dicomdir[d])))
    else:
        dcm_paths = _get_full_path(dicomdir, sorted(os.listdir(dicomdir)))
                    
    return dcm_paths


def _get_full_path(root, file_list):
    """
    Create list of full file paths from file name and root folder path.
    """
    return [os.path.abspath(os.path.join(root, file)) for file in file_list]


def _probe_dicom_paths(dcm_paths_in):
    """
    For each element in list, check if it is a folder and read dicom paths inside it.
    """
    dcm_paths_out = []
    
    # loop over paths in input list
    for path in dcm_paths_in:
        if os.path.isdir(path):
            dcm_paths_out += _get_dicom_paths(path)
        else:
            dcm_paths_out.append(path) 
        
    return dcm_paths_out


def _dcmread(dcm_path):
    """
    Wrapper to pydicom dcmread to automatically handle not dicom files
    """
    try:
        return pydicom.dcmread(dcm_path)
    except:
        return None
    
    
def _dcmwrite(input):
    """
    Wrapper to pydicom dcmread to automatically handle path / file tuple
    """
    filename, dataset = input
    pydicom.dcmwrite(filename, dataset)


def _cast_to_complex(dsets_in):
    """
    Attempt to retrive complex image, with the following priority:
        
        1) Real + 1j Imag
        2) Magnitude * exp(1j * Phase)

    If neither Real / Imag nor Phase are found, returns Magnitude only.
    """
    # get vendor
    vendor = _get_vendor(dsets_in[0])
    
    # actual conversion
    if vendor == 'GE':
        return _cast_to_complex_ge(dsets_in)
    
    if vendor == 'Philips':
        return _cast_to_complex_philips(dsets_in)
    
    if vendor == 'Siemens':
        return _cast_to_complex_siemens(dsets_in)
    
    
def _get_vendor(dset):
    """
    Get vendor from DICOM header.
    """
    if dset.Manufacturer == 'GE MEDICAL SYSTEMS':
        return 'GE'
    
    if dset.Manufacturer == 'Philips Medical Systems':
        return 'Philips'
    
    if dset.Manufacturer == 'SIEMENS':
        return 'Siemens'
    

def _cast_to_complex_ge(dsets_in):
    """
    Attempt to retrive complex image for GE DICOM, with the following priority:
        
        1) Real + 1j Imag
        2) Magnitude * exp(1j * Phase)

    If neither Real / Imag nor Phase are found, returns Magnitude only.
    """
    # initialize
    real = []
    imag = []
    magnitude = []
    phase = []
    do_recon = True
    
    # allocate template out
    dsets_out = []
    
    # loop over dataset
    for dset in dsets_in:
        if dset[0x0043, 0x102f].value == 0:
            magnitude.append(dset.pixel_array)
            dsets_out.append(dset)
        
        if dset[0x0043, 0x102f].value == 1:
            phase.append(dset.pixel_array)
            
        if dset[0x0043, 0x102f].value == 2:
            real.append(dset.pixel_array)
        
        if dset[0x0043, 0x102f].value == 3:
            imag.append(dset.pixel_array)
            
    if real and imag and do_recon:
        image = np.stack(real, axis=0).astype(np.float32) + 1j * np.stack(imag, axis=0).astype(np.float32)
        do_recon = False
    
    if magnitude and phase and do_recon:
        scale = 2 * np.pi / 4095
        offset = -np.pi
        image = np.stack(magnitude, axis=0).astype(np.float32) * np.exp( 1j * (scale * np.stack(phase, axis=0) + offset).astype(np.float32))
        do_recon = False
    elif do_recon:
        image = np.stack(magnitude, axis=0).astype(np.float32)
        
    # fix phase shift along z
    if np.iscomplexobj(image):
        phase = np.angle(image)
        phase[..., 1::2, :, :] = ((1e5 * (phase[..., 1::2, :, :] + 2 * np.pi)) % (2 * np.pi * 1e5)) / 1e5 - np.pi
        image = np.abs(image) * np.exp(1j * phase)
        
    # count number of instances
    ninstances = image.shape[0]
        
    # assign to pixel array
    for n in range(ninstances):
        dsets_out[n].pixel_array[:] = 0.0
        dsets_out[n][0x0025, 0x1007].value = ninstances
        dsets_out[n][0x0025, 0x1019].value = ninstances
        
    return image, dsets_out


def _cast_to_complex_philips(dsets_in):
    """
    Attempt to retrive complex image for Philips DICOM:
    If Phase is not found, returns Magnitude only.
    """
    # initialize
    magnitude = []
    phase = []
    
    # allocate template out
    dsets_out = []
    
    # loop over dataset
    for dset in dsets_in:
        if dset.ImageType[-2] == 'M':
            magnitude.append(dset.pixel_array)
            dsets_out.append(dset)
        
        if dset.ImageType[-2] == 'P':
            phase.append(dset.pixel_array)
                           
    if magnitude and phase:
        scale = 2 * np.pi / 4095
        offset = -np.pi
        image = np.stack(magnitude, axis=0).astype(np.float32) * np.exp( 1j * (scale * np.stack(phase, axis=0) + offset).astype(np.float32))
    else:
        image = np.stack(magnitude, axis=0).astype(np.float32)
        
    # count number of instances
    ninstances = image.shape[0]
        
    # assign to pixel array
    for n in range(ninstances):
        dsets_out[n].pixel_array[:] = 0.0
        
    return image, dsets_out


def _cast_to_complex_siemens(dsets_in):
    """
    Attempt to retrive complex image for Siemens DICOM:
    If Phase is not found, returns Magnitude only.
    """
    # initialize
    magnitude = []
    phase = []
    
    # allocate template out
    dsets_out = []
    
    # loop over dataset
    for dset in dsets_in:
        if dset.ImageType[2] == 'M':
            magnitude.append(dset.pixel_array)
            dsets_out.append(dset)
        
        if dset.ImageType[2] == 'P':
            phase.append(dset.pixel_array)
                           
    if magnitude and phase:
        scale = 2 * np.pi / 4095
        offset = -np.pi
        image = np.stack(magnitude, axis=0).astype(np.float32) * np.exp( 1j * (scale * np.stack(phase, axis=0) + offset).astype(np.float32))
    else:
        image = np.stack(magnitude, axis=0).astype(np.float32)
        
    # count number of instances
    ninstances = image.shape[0]
        
    # assign to pixel array
    for n in range(ninstances):
        dsets_out[n].pixel_array[:] = 0.0
        
    return image, dsets_out


def _get_slice_locations(dsets):
    """
    Return array of unique slice locations and slice location index for each dataset in dsets.
    """
    # get unique slice locations
    sliceLocs = _get_relative_slice_position(dsets).round(decimals=4)
    uSliceLocs, firstSliceIdx = np.unique(sliceLocs, return_index=True)
    
    # get indexes
    sliceIdx = np.zeros(sliceLocs.shape, dtype=np.int)
    
    for n in range(len(uSliceLocs)):
        sliceIdx[sliceLocs == uSliceLocs[n]] = n
        
    return uSliceLocs, firstSliceIdx, sliceIdx


def _get_image_orientation(dsets):
    """
    Return image orientation matrix.
    """
    F = np.array(dsets[0].ImageOrientationPatient).reshape(2, 3)
    
    return F


def _get_plane_normal(dsets):
    """
    Return array of normal to imaging plane, as the cross product
    between x and y plane versors.
    """
    x, y = _get_image_orientation(dsets)
    return np.cross(x, y)


def _get_position(dsets):
    """
    Return matrix of image position of size (3, nslices).
    """
    return np.stack([dset.ImagePositionPatient for dset in dsets], axis=1)
    

def _get_relative_slice_position(dsets):
    """
    Return array of slice coordinates along the normal to imaging plane.
    """
    z = _get_plane_normal(dsets)
    position =  _get_position(dsets)
    return z @ position
    
    
def _get_flip_angles(dsets):
    """
    Return array of flip angles for each dataset in dsets.
    """
    # get flip angles
    flipAngles = np.array([float(dset.FlipAngle) for dset in dsets])
      
    return flipAngles


def _get_echo_times(dsets):
    """
    Return array of echo times for each dataset in dsets.
    """
    # get unique echo times
    echoTimes = np.array([float(dset.EchoTime) for dset in dsets])
          
    return echoTimes


def _get_repetition_times(dsets):
    """
    Return array of repetition times for each dataset in dsets.
    """
    # get unique repetition times
    repetitionTimes = np.array([float(dset.RepetitionTime) for dset in dsets])
            
    return repetitionTimes


def _get_inversion_times(dsets):
    """
    Return array of inversion times for each dataset in dsets.
    """
    try:
        # get unique repetition times
        inversionTimes = np.array([float(dset.InversionTime) for dset in dsets])           
    except:
        inversionTimes = np.zeros(len(dsets)) + np.inf
        
    return inversionTimes


def _get_unique_contrasts(constrasts):
    """
    Return ndarray of unique contrasts and contrast index for each dataset in dsets.
    """
    # get unique repetition times
    uContrasts = np.unique(constrasts, axis=0)
    
    # get indexes
    contrastIdx = np.zeros(constrasts.shape[0], dtype=np.int)
    
    for n in range(uContrasts.shape[0]):
        contrastIdx[(constrasts == uContrasts[n]).all(axis=-1)] = n
                 
    return uContrasts, contrastIdx


def _get_dicom_template(dsets, index):
    """
    Get template of Dicom to be used for saving later.
    """
    template = []

    SeriesNumber = dsets[index[0]].SeriesNumber
    
    for n in range(len(index)):
        dset = copy.deepcopy(dsets[index[n]])
        
    #     dset.pixel_array[:] = 0.0
    #     dset.PixelData = dset.pixel_array.tobytes()
                
    #     dset.WindowWidth = None
    #     dset.WindowCenter = None

    #     dset.SeriesDescription = None
        dset.SeriesNumber = SeriesNumber
    #     dset.SeriesInstanceUID = None
    
    #     dset.SOPInstanceUID = None
    #     dset.InstanceNumber = None
        
    #     try:
    #         dset.ImagesInAcquisition = None
    #         dset[0x0025, 0x1007].value = None
    #         dset[0x0025, 0x1019].value = None
    #     except:
    #         pass
        
    #     dset.InversionTime = '0'
        # dset[0x0018, 0x0086].value = '1' # Echo Number
    #     dset.EchoTime = '0'
    #     dset.EchoTrainLength = '1'
    #     dset.RepetitionTime = '0'
    #     dset.FlipAngle = '0'
        
        template.append(dset)
    
    return template


def _get_nifti_affine(dsets, shape):
    """
    Return affine transform between voxel coordinates and mm coordinates as
    described in https://nipy.org/nibabel/dicom/spm_dicom.html#spm-volume-sorting
    """
    # common parameters
    T = _get_position(dsets)
    T1 = T[:, 0].round(4)
    
    F = _get_image_orientation(dsets)
    dr, dc = np.array(dsets[0].PixelSpacing).round(4)
    
    if len(dsets) == 1: # single slice case
        n = _get_plane_normal(dsets)
        ds = float(dsets[0].SliceThickness)
    
        A0 = np.stack((np.append(F[0] * dr, 0),
                       np.append(F[1] * dc, 0),
                       np.append(ds * n, 0),
                       np.append(T1, 1)), axis=1)

    else: # multi slice case
        N = len(dsets)
        TN = T[:,-1].round(4)
        A0 = np.stack((np.append(F[0] * dr, 0),
                       np.append(F[1] * dc, 0),
                       np.append((TN - T1) / (N - 1), 0),
                       np.append(T1, 1)), axis=1)
        
    # get orientation
    axial_orientation, coronal_orientation, sagittal_orientation = __calculate_slice_orientation__(A0)
    
    # fix affine matrix
    A = np.eye(4)
    A[:, 0] = A0[:, sagittal_orientation.normal_component]
    A[:, 1] = A0[:, coronal_orientation.normal_component]
    A[:, 2] = A0[:, axial_orientation.normal_component]
    point = [0, 0, 0, 1]
            
    # If the orientation of coordinates is inverted, then the origin of the "new" image
    # would correspond to the last voxel of the original image
    # First we need to find which point is the origin point in image coordinates
    # and then transform it in world coordinates
    if not axial_orientation.x_inverted:
        A[:, 0] = - A[:, 0]
        point[sagittal_orientation.normal_component] = shape[sagittal_orientation.normal_component] - 1

    if axial_orientation.y_inverted:
        A[:, 1] = - A[:, 1]
        point[coronal_orientation.normal_component] = shape[coronal_orientation.normal_component] - 1

    if coronal_orientation.y_inverted:
        A[:, 2] = - A[:, 2]
        point[axial_orientation.normal_component] = shape[axial_orientation.normal_component] - 1

    A[:, 3] = np.dot(A0, point)

    return A


class SliceOrientation:
    """
    Class containing the orientation of a slice.
    """
    x_component = None
    y_component = None
    normal_component = None
    x_inverted = False
    y_inverted = False
    

def __calculate_slice_orientation__(affine):
    # Not all image data has the same orientation
    # We use the affine matrix and multiplying it with one component
    # of the slice we can find the correct orientation
    affine_inverse = np.linalg.inv(affine)
    transformed_x = np.transpose(np.dot(affine_inverse, [[1], [0], [0], [0]]))[0]
    transformed_y = np.transpose(np.dot(affine_inverse, [[0], [1], [0], [0]]))[0]
    transformed_z = np.transpose(np.dot(affine_inverse, [[0], [0], [1], [0]]))[0]

    # calculate the most likely x,y,z direction
    x_component, y_component, z_component = __calc_most_likely_direction__(transformed_x,
                                                                           transformed_y,
                                                                           transformed_z)

    # Find slice orientiation for the axial size
    # Find the index of the max component to know which component is the direction in the size
    axial_orientation = SliceOrientation()
    axial_orientation.normal_component = z_component
    axial_orientation.x_component = x_component
    axial_orientation.x_inverted = np.sign(transformed_x[axial_orientation.x_component]) < 0
    axial_orientation.y_component = y_component
    axial_orientation.y_inverted = np.sign(transformed_y[axial_orientation.y_component]) < 0
    
    # Find slice orientiation for the coronal size
    # Find the index of the max component to know which component is the direction in the size
    coronal_orientation = SliceOrientation()
    coronal_orientation.normal_component = y_component
    coronal_orientation.x_component = x_component
    coronal_orientation.x_inverted = np.sign(transformed_x[coronal_orientation.x_component]) < 0
    coronal_orientation.y_component = z_component
    coronal_orientation.y_inverted = np.sign(transformed_z[coronal_orientation.y_component]) < 0
    
    # Find slice orientation for the sagittal size
    # Find the index of the max component to know which component is the direction in the size
    sagittal_orientation = SliceOrientation()
    sagittal_orientation.normal_component = x_component
    sagittal_orientation.x_component = y_component
    sagittal_orientation.x_inverted = np.sign(transformed_y[sagittal_orientation.x_component]) < 0
    sagittal_orientation.y_component = z_component
    sagittal_orientation.y_inverted = np.sign(transformed_z[sagittal_orientation.y_component]) < 0
    
    # Assert that the slice normals are not equal
    assert axial_orientation.normal_component != coronal_orientation.normal_component
    assert coronal_orientation.normal_component != sagittal_orientation.normal_component
    assert sagittal_orientation.normal_component != axial_orientation.normal_component
    
    return axial_orientation, coronal_orientation, sagittal_orientation
        

def __calc_most_likely_direction__(transformed_x, transformed_y, transformed_z):
    """
    Calculate which is the most likely component for a given direction
    """
    # calculate the x component
    tx_dot_x = np.abs(np.dot(transformed_x, [1, 0, 0, 0]))
    tx_dot_y = np.abs(np.dot(transformed_x, [0, 1, 0, 0]))
    tx_dot_z = np.abs(np.dot(transformed_x, [0, 0, 1, 0]))
    x_dots = [tx_dot_x, tx_dot_y, tx_dot_z]
    x_component = np.argmax(x_dots)
    x_max = np.max(x_dots)

    # calculate the y component
    ty_dot_x = np.abs(np.dot(transformed_y, [1, 0, 0, 0]))
    ty_dot_y = np.abs(np.dot(transformed_y, [0, 1, 0, 0]))
    ty_dot_z = np.abs(np.dot(transformed_y, [0, 0, 1, 0]))
    y_dots = [ty_dot_x, ty_dot_y, ty_dot_z]
    y_component = np.argmax(y_dots)
    y_max = np.max(y_dots)

    # calculate the z component
    tz_dot_x = np.abs(np.dot(transformed_z, [1, 0, 0, 0]))
    tz_dot_y = np.abs(np.dot(transformed_z, [0, 1, 0, 0]))
    tz_dot_z = np.abs(np.dot(transformed_z, [0, 0, 1, 0]))
    z_dots = [tz_dot_x, tz_dot_y, tz_dot_z]
    z_component = np.argmax(z_dots)
    z_max = np.max(z_dots)

    # as long as there are duplicate directions try to correct
    while x_component == y_component or x_component == z_component or y_component == z_component:
        if x_component == y_component:
            # keep the strongest one and change the other
            if x_max >= y_max:  # update the y component
                y_dots[y_component] = 0
                y_component = np.argmax(y_dots)
                y_max = np.max(y_dots)
            else:  # update the x component
                x_dots[x_component] = 0
                x_component = np.argmax(x_dots)
                x_max = np.max(x_dots)

        if x_component == z_component:
            # keep the strongest one and change the other
            if x_max >= z_max:  # update the z component
                z_dots[z_component] = 0
                z_component = np.argmax(z_dots)
                z_max = np.max(z_dots)
            else:  # update the x component
                x_dots[x_component] = 0
                x_component = np.argmax(x_dots)
                x_max = np.max(x_dots)

        if y_component == z_component:
            # keep the strongest one and change the other
            if y_max >= z_max:  # update the z component
                z_dots[z_component] = 0
                z_component = np.argmax(z_dots)
                z_max = np.max(z_dots)
            else:  # update the y component
                y_dots[y_component] = 0
                y_component = np.argmax(y_dots)
                y_max = np.max(y_dots)

    return x_component, y_component, z_component