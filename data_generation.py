# -*- coding: utf-8 -*-
"""
data_generation.py - The data generation module of the project
==============================================================

This module generates the requested database for each experiment, using the method 'fetch_data'.
"""
import numpy as np
import pandas as pd
import os
import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from Infrastructure.enums import DBType
from Infrastructure.utils import ex, Union, Dict, Vector, Matrix, ThreeDMatrix, List
from nilearn.image import resample_img


@ex.capture
def fetch_covid19_db(resources_path: str, covid19_ct_scans_config: Dict[str, str],
                     db_size: Union[int, None] = None) -> ThreeDMatrix:
    """
    This function generates the database of COVID-19 images,
    taken from https://www.kaggle.com/andrewmvd/covid19-ct-scans.

    Args:
        db_size(int or None): The number of images to create. When given None, 
            or when db_size is larger then the database size, it creates 
            all available images in the database.

    Returns:
        A 3D matrix.
    """
    db_path: str = covid19_ct_scans_config["db_path"]
    db_file_name: str = covid19_ct_scans_config["database_file_name"]
    db = pd.read_csv(os.path.join(resources_path, db_path, db_file_name))
    db = db['ct_scan']
    db_size: int = db_size if db_size is not None else len(db)
    data = list()
    for image_path in db[:db_size]:
        actual_path: str = image_path[3:]
        current_image = nib.load(os.path.join(resources_path, db_path, actual_path))
        # print(current_image.shape)
        current_image = resample_img(current_image, target_affine=np.eye(3)*4., interpolation='nearest')
        # print(current_image.shape)
        image_pixels: Matrix = current_image.get_fdata()[:, :, 120 // 4]  # Using the mid-images in the volume as 2D images.
        arr = np.rot90(np.array(image_pixels))
        # Re-scaling all images, because of RAM limitations.
        # arr = rescale(arr, scale=0.5, mode='reflect', multichannel=False)
        arr = (arr - arr.min()) / (arr.max() - arr.min())  # Normalizing each image
        data.append(arr)
    data_as_matrix: ThreeDMatrix = np.stack(data)
    return data_as_matrix


@ex.capture
def fetch_medical_images_kaggle_db(resources_path: str, ct_medical_images_kaggle_config: Dict[str, str],
                                   db_size: Union[int, None] = None) -> ThreeDMatrix:
    """
    This function generates the database of CT images, taken from https://www.kaggle.com/kmader/siim-medical-images.

    Args:
        db_size(int or None): The number of images to create. When given None, 
            or when db_size is larger then the database's size, it creates 
            all available images in the database.

    Returns:
        A 3D matrix.
    """
    db_path: str = ct_medical_images_kaggle_config["db_path"]
    db_file_name: str = ct_medical_images_kaggle_config["database_file_name"]
    db = pd.read_csv(os.path.join(resources_path, db_path, db_file_name))['dicom_name']
    db_size: int = db_size if db_size is not None else len(db)
    data = list()
    for image_path in db[:db_size]:
        x = pydicom.dcmread(os.path.join(resources_path, db_path, "dicom_dir", image_path))
        arr = apply_modality_lut(x.pixel_array, x)  # Shift values to Hounsfield Units.
        # Re-scaling all images, because of RAM limitations.
        arr = rescale(arr, scale=0.25, mode='reflect', multichannel=False)
        arr = (arr - arr.min()) / (arr.max() - arr.min())  # Normalizing each image
        data.append(arr)
    data: ThreeDMatrix = np.array(data, order='C')
    return data


@ex.capture
def fetch_shepp_logan_phantom(shepp_logan_scaling_factors: List[float], db_size: int = 1) -> ThreeDMatrix:
    """
    This function generates scaled copies of the Shepp-Logan phantom.

    Args:
        db_size(int or None): The number of images to create. When given None, 
            or when db_size is larger then the database's size, it creates 
            all available images in the database.

    Returns:
        A 3D matrix.
    """
    images = list()
    orig_image: Matrix = shepp_logan_phantom()
    orig_image = rescale(orig_image, scale=0.1, mode='reflect', multichannel=False)
    scaling_factors = shepp_logan_scaling_factors[:db_size]
    for scaling_factor in scaling_factors:
        if scaling_factor < 1:
            smaller_image = rescale(orig_image, scale=scaling_factor, mode='reflect', multichannel=False)
            pad_width = int((1 - scaling_factor) * max(orig_image.shape[0], orig_image.shape[1])) // 2
            image = np.pad(smaller_image, pad_width, pad_with_zeros)
        else:
            image = orig_image
        images.append(image.copy())
    return np.stack(images)


def _zero_outside_circle(data: ThreeDMatrix) -> ThreeDMatrix:
    """
    This function nullifies every cell in every image of the array, which lies
    outside the circle at the center, with the image minimal side length as its diameter.

    Args:
        data(ThreeDMatrix): A 3D array, where data[i] is the i-th image.

    Returns:
        A 3D matrix, with shape identical to the shape of data.
    """
    img_shape: Vector = np.array(data.shape[1:])
    radius: float = min(img_shape) // 2
    coords = np.array(np.ogrid[:img_shape[0], :img_shape[1]], dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    indices_mask = dist > radius ** 2
    
    data[:, indices_mask] = 0
    return data
    

def fetch_data(db_type: str, db_size: Union[int, None] = None) -> ThreeDMatrix:
    """
    This function generates the desired database.

    Args:
        db_type(str): A DBType, the name of the requested database.
        db_size(int or None): The number of images to create. When given None, 
            or when db_size is larger then the database's size, it creates 
            all available images in the database.

    Returns:
        A 3D matrix, the requested database.
    """
    data = None
    if db_type == DBType.SheppLogan:
        data: ThreeDMatrix = fetch_shepp_logan_phantom(db_size=db_size)
    elif db_type == DBType.COVID19_CT_Scans:
        data: ThreeDMatrix = fetch_covid19_db(db_size=db_size)
    elif db_type == DBType.CT_Medical_Images:
        data: ThreeDMatrix = fetch_medical_images_kaggle_db(db_size=db_size)
    print(data.shape)
    return _zero_outside_circle(data)


def pad_with_zeros(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
