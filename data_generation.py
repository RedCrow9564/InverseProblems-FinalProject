import numpy as np
import pandas as pd
import os
import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from Infrastructure.enums import DBType
from Infrastructure.utils import ex, Union, Dict, Scalar, Vector, Matrix, ThreeDMatrix, List
import matplotlib.pyplot as plt


@ex.capture
def fetch_covid19_db(resources_path: str, covid19_ct_scans_config: Dict[str, str], db_size: Union[int, None]=None) -> ThreeDMatrix:
    db_path: str = covid19_ct_scans_config["db_path"]
    db_file_name: str = covid19_ct_scans_config["database_file_name"]
    db = pd.read_csv(os.path.join(resources_path, db_path, db_file_name))
    db = db['ct_scan']
    db_size: int = db_size if db_size is not None else len(db)
    data = list()
    for image_path in db[:db_size]:
        actual_path: str = image_path[3:]
        current_image = nib.load(os.path.join(resources_path, db_path, actual_path))
        image_pixels: Matrix  = current_image.get_fdata()[:, :, 120]  # Using the mid-images in the volume as 2D images.
        arr = np.rot90(np.array(image_pixels))
        data.append(arr)
    data: ThreeDMatrix = np.array(data, order='C')
    return data


@ex.capture
def fetch_medical_images_kaggle_db(resources_path: str, ct_medical_images_kaggle_config: Dict[str, str], db_size: Union[int, None]=None) -> ThreeDMatrix:
    db_path: str = ct_medical_images_kaggle_config["db_path"]
    db_file_name: str = ct_medical_images_kaggle_config["database_file_name"]
    db = pd.read_csv(os.path.join(resources_path, db_path, db_file_name))['dicom_name']
    db_size: int = db_size if db_size is not None else len(db)
    data = list()
    for image_path in db[:db_size]:
        x = pydicom.dcmread(os.path.join(resources_path, db_path, "dicom_dir", image_path))
        arr = apply_modality_lut(x.pixel_array, x)  # Shift values to Hounsfield Units.
        data.append(arr)
    data: ThreeDMatrix = np.array(data, order='C')
    return data


@ex.capture
def fetch_shepp_logan_phantom(shepp_logan_scaling_factors: List[float], db_size: int=1) -> ThreeDMatrix:
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
    img_shape: Vector = np.array(data.shape[1:])
    radius: float = min(img_shape) // 2
    coords = np.array(np.ogrid[:img_shape[0], :img_shape[1]],
                      dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    indices_mask = dist > radius ** 2
    
    data[:, indices_mask] = 0
    return data
    

def fetch_data(db_type: str, db_size: Union[int, None]=None) -> ThreeDMatrix:
    if db_type == DBType.Random:
        pass  # TODO: Create random data generation function
    elif db_type == DBType.SheppLogan:
        data = fetch_shepp_logan_phantom(db_size=db_size)
    elif db_type == DBType.COVID19_CT_Scans:
        data = fetch_covid19_db(db_size=db_size)
    elif db_type == DBType.CT_Medical_Images:
        data = fetch_medical_images_kaggle_db(db_size=db_size)
    # from matplotlib import pyplot as plt
    # a = _zero_outside_circle(data)
    # plt.imshow(a[0], cmap=plt.cm.Greys_r)
    # plt.show()
    return _zero_outside_circle(data)


def pad_with_zeros(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value
