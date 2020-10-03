from itertools import product
import numpy as np
from numba import jit
import pylops
from numpy.random import Generator, PCG64
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


def add_noise_by_snr(images: ThreeDMatrix, snr: float, random_generator) -> ThreeDMatrix:
    noisy_images: ThreeDMatrix = images.copy()
    for image in noisy_images:
        image_amplitude: float = np.linalg.norm(image, ord='fro')
        image += random_generator.normal(0, np.sqrt(1 / snr) * image_amplitude / float(image.size), image.shape)
        print("Raw image fro norm: {}, noised image fro norm: {}".format(image_amplitude, np.linalg.norm(image, ord='fro')))
    # if snr > 0.1:
    #     a = 1/0
    return noisy_images


def _radon_curve_image(image_shape):
    ny, nx = image_shape
    
    @jit(nopython=True)
    def radoncurve(x, r, theta):
        return (r - ny//2)/(np.sin(np.deg2rad(theta))+1e-15) + np.tan(np.deg2rad(90 - theta))*x  + ny//2
    return radoncurve


class BaseExperiment:
    def __init__(self, original_images: ThreeDMatrix, data_type: str, 
                 seed: int, log_fields: List[str], save_estimated_images: bool = False):
        self._true_images: ThreeDMatrix = original_images
        self._data_type: str = data_type
        self._save_estimated_images: bool = save_estimated_images
        self.data_log = DataLog(log_fields)
        self._rng = Generator(PCG64(seed))  # Set random generator.

    @staticmethod
    def radon_transform_all_images(images: ThreeDMatrix, thetas: Vector, 
                                   method:str) -> ThreeDMatrix:
        sinograms = list()
        ntheta: int = len(thetas)
        
        if method =='Scikit-Image':
            for image in images:
                sinogram: Matrix = radon(image, thetas, circle=True)
                sinograms.append(sinogram)
            sinograms = np.array(sinograms, dtype=np.float64)
            return sinograms
        
        elif method == 'Pylops':
            ny, nx = images[0].shape
            c = _radon_curve_image(images[0].shape)
            R = pylops.signalprocessing.Radon2D(
              np.arange(ny), np.arange(nx), thetas, kind=c, centeredh=True,
              interp=False, engine='numba', dtype='float64').H
            
            for image in images:
                y: Vector = R * image.T.ravel()
                sinogram = y.reshape(ntheta, ny).T
                sinograms.append(sinogram)

            sinograms = np.array(sinograms, dtype=np.float64)
            return sinograms, R
