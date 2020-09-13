from itertools import product
import numpy as np
from skimage.transform import radon
from Infrastructure.enums import LogFields
from Infrastructure.utils import Scalar, Vector, Matrix, ThreeDMatrix, List, DataLog


def error_in_circle_pixels(true_images: Matrix, estimated_images: Matrix) -> float:
    indices_mask: Matrix = np.zeros_like(true_images[0])
    middle_index: int = int(max(true_images.shape[1],  true_images.shape[2]) / 2)
    for i, j in product(range(true_images.shape[1]), range(true_images.shape[2])):
        if (middle_index - i)  ** 2 + (middle_index - j) ** 2 < middle_index ** 2:
            indices_mask[i, j] = 1
    
    error_per_image: Vector = list()
    for true_image, estimated_image in zip(true_images, estimated_images):
        current_error: float = np.mean((true_image[indices_mask == 1] - 
            estimated_image[indices_mask == 1]) ** 2)
        error_per_image.append(current_error)
    error: float = np.sqrt(np.mean(error_per_image))
    return error


class BaseExperiment:
    def __init__(self, original_images: ThreeDMatrix, data_type: str, save_estimated_images: bool = False):
        self._true_images: ThreeDMatrix = original_images
        self._data_type: str = data_type
        self._save_estimated_images: bool = save_estimated_images
        self.data_log = DataLog(LogFields)

    @staticmethod
    def radon_transform_all_images(images: ThreeDMatrix, thetas: Vector) -> ThreeDMatrix:
        sinograms = list()
        for image in images:
            sinogram: Matrix = radon(image, thetas, circle=True)
            sinograms.append(sinogram)

        sinograms = np.array(sinograms, dtype=np.float64)
        return sinograms

    @staticmethod
    def estimate_error(true_images: ThreeDMatrix, estimated_images: ThreeDMatrix) -> float:
        return error_in_circle_pixels(true_images, estimated_images)
