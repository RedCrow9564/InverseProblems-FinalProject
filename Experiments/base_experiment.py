# -*- coding: utf-8 -*-
"""
base_experiment.py - The basis for all experiments in this project.
===================================================================

This module contains the base class for all possible experiments in this project.
"""
import numpy as np
from numpy.random import Generator, PCG64
from skimage.transform import radon
from Infrastructure.utils import Scalar, Vector, Matrix, ThreeDMatrix, List, DataLog


def error_in_circle_pixels(true_images: ThreeDMatrix, estimated_images: ThreeDMatrix) -> Scalar:
    """
    This method estimated the error between the true images, and the estimated images.
    
    Args:
        true_images(ThreeDMatrix): The true images.
        estimated_images(ThreeDMatrix): The estimated images.

    Returns:
        A scalar value of error.
    """
    img_shape: Vector = np.array(true_images.shape[1:])
    radius: float = min(img_shape) // 2
    coords = np.array(np.ogrid[:img_shape[0], :img_shape[1]], dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    indices_mask = dist <= radius ** 2
    error: Scalar = np.mean(np.power(true_images[:, indices_mask] -
                            estimated_images[:, indices_mask], 2))
    error = np.sqrt(np.mean(error))
    print(error)
    return error


def add_noise_by_snr(images: ThreeDMatrix, snr: float, random_generator) -> ThreeDMatrix:
    """
    This method applies Radon transform to all input images, at the requested angles.
    Then it generates noise according to the input SNR and adds it to the noise.

    Args:
        images(ThreeDMatrix): Input images.
        snr(float): A value of SNR for generating noise.
        random_generator: A random generator for creating noise.

    Returns:
        A 3D matrix, which are the sinograms of the input images.
    """
    noisy_images: ThreeDMatrix = images.copy()
    if snr in (0.0, np.inf):
        return noisy_images
    
    image_size: int = float(images[0].size)
    for image in noisy_images:
        image_amplitude: float = np.linalg.norm(image, ord='fro')
        noise_magnitude_per_pixel: float = np.sqrt(1 / snr) * image_amplitude
        noise_magnitude_per_pixel /= image_size
        image += random_generator.normal(0, noise_magnitude_per_pixel, image.shape)
    return noisy_images


class BaseExperiment:
    def __init__(self, original_images: ThreeDMatrix, data_type: str, 
                 seed: int, log_fields: List[str], save_estimated_images: bool = False):
        self._true_images: ThreeDMatrix = original_images
        self._data_type: str = data_type
        self._save_estimated_images: bool = save_estimated_images
        self.data_log = DataLog(log_fields)
        self._rng = Generator(PCG64(seed))  # Set random generator.
        self._calculated_output_images = None

    @staticmethod
    def radon_transform_all_images(images: ThreeDMatrix, thetas: Vector) -> ThreeDMatrix:
        """
        This method applies Radon transform to all input images, at the requested angles.
        Args:
            images(ThreeDMatrix): Input images.
            thetas(Vector): A list of angles at which the projections are taken.

        Returns:
            A 3D matrix, which are the sinograms of the input images.
        """
        sinograms: ThreeDMatrix = np.empty(
            (images.shape[0], images.shape[1], len(thetas)), dtype=np.float64)
        
        for i, image in enumerate(images):
            sinograms[i] = radon(image, thetas, circle=True)
        weights: Matrix = create_weights(images.shape[1:], thetas, images.shape[1])
        return sinograms, weights
        

def _get_w_for_ray(image_shape, theta, iproj):
    """
    Compute the projection of an image along a ray.
    Parameters
    ----------
    image : 2D array, dtype=float
        Image to project.
    theta : float
        Angle of the projection
    ray_position : float
        Position of the ray within the projection
    Returns
    -------
    projected_value : float
        Ray sum along the projection
    norm_of_weights :
        A measure of how long the ray's path through the reconstruction
        circle was
    """
    theta = theta / 180. * np.pi
    radius = image_shape[0] // 2 - 1
    projection_center = image_shape[0] // 2
    rotation_center = image_shape[0] // 2
    # (s, t) is the (x, y) system rotated by theta
    t = iproj - projection_center
    # s0 is the half-length of the ray's path in the reconstruction circle
    s0 = np.sqrt(radius * radius - t * t) if radius*radius >= t*t else 0.
    Ns = 2 * int(np.ceil(2 * s0))  # number of steps along the ray

    W = np.zeros(image_shape)
    if Ns > 0:
        # step length between samples
        ds = 2 * s0 / Ns
        dx = -ds * np.cos(theta)
        dy = -ds * np.sin(theta)
        # point of entry of the ray into the reconstruction circle
        x0 = s0 * np.cos(theta) - t * np.sin(theta)
        y0 = s0 * np.sin(theta) + t * np.cos(theta)
        for k in range(Ns + 1):
            x = x0 + k * dx
            y = y0 + k * dy
            index_i = x + rotation_center
            index_j = y + rotation_center
            i = int(np.floor(index_i))
            j = int(np.floor(index_j))
            di = index_i - np.floor(index_i)
            dj = index_j - np.floor(index_j)
            # Use linear interpolation between values
            # Where values fall outside the array, assume zero
            if i > 0 and j > 0:
                weight = (1. - di) * (1. - dj) * ds
                # ray_sum += weight * image[i, j]
                # weight_norm += weight * weight
                W[i, j] = W[i, j] + weight
            if i > 0 and j < image_shape[1] - 1:
                weight = (1. - di) * dj * ds
                # ray_sum += weight * image[i, j+1]
                # weight_norm += weight * weight
                W[i, j + 1] = W[i, j + 1] + weight
            if i < image_shape[0] - 1 and j > 0:
                weight = di * (1 - dj) * ds
                # ray_sum += weight * image[i+1, j]
                # weight_norm += weight * weight
                W[i + 1, j] = W[i + 1, j] + weight
            if i < image_shape[0] - 1 and j < image_shape[1] - 1:
                weight = di * dj * ds
                W[i + 1, j + 1] = W[i + 1, j + 1] + weight
                # ray_sum += weight * image[i+1, j+1]
                # weight_norm += weight * weight
    W = W.flatten()
    return W


def create_weights(image_shape, theta, number_of_projections): 
    
    a1 = theta.shape[0] * number_of_projections
    a2 = image_shape[0] * image_shape[1]
    weights = np.zeros((a1, a2), dtype=np.float)
    counter = 0
    for angle_index in range(theta.shape[0]):
        for i_proj in range(number_of_projections):
            v = _get_w_for_ray(image_shape, theta[angle_index], i_proj)
            weights[counter, :] = v
            counter += 1
    return weights
