import numpy as np
import pandas as pd
import os
import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
from skimage.data import shepp_logan_phantom
from skimage.transform import rescale
from Infrastructure.enums import DBType
from Infrastructure.utils import ex, Union, Dict, Scalar, Vector, Matrix, ThreeDMatrix


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


def fetch_shepp_logan_phantom() -> ThreeDMatrix:
    image: Matrix = shepp_logan_phantom()
    image = rescale(image, scale=0.4, mode='reflect', multichannel=False)
    image: ThreeDMatrix = image.reshape((1, image.shape[0], image.shape[1]))
    return image    
    

def fetch_data(db_type: str, db_size: Union[int, None]=None) -> ThreeDMatrix:
    if db_type == DBType.Random:
        pass  # TODO: Create random data generation function
    elif db_type == DBType.SheppLogan:
        return fetch_shepp_logan_phantom()
    elif db_type == DBType.COVID19_CT_Scans:
        return fetch_covid19_db(db_size=db_size)
    elif db_type == DBType.CT_Medical_Images:
        return fetch_medical_images_kaggle_db(db_size=db_size)
