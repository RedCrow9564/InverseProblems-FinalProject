# -*- coding: utf-8 -*-
"""
fbp_and_sart.py - Non-iterative mthods module.
==============================================

This module applies the methods of FBP and SART. It uses tha available functions of
the Scikit-Image package (https://scikit-image.org/).
"""
import numpy as np
from skimage.transform import iradon, iradon_sart
from Infrastructure.utils import Scalar, Vector, Matrix, ThreeDMatrix


def filtered_back_projection(sinogram: Matrix, thetas: Vector, filter_name: str) -> Matrix:
    """
    This mehtod applies the FBP algorithm.
    Args:
        sinogram(Matrix): A given sinogram.
        thetas(Vector): A list of angles at which the projections were taken.
        filter_name(str): A string which represents the requested filter for this method. 
            For exmaple 'ramp' or 'hamming'.

    Returns:
        A 2D matrix.
    """
    return iradon(sinogram, thetas, circle=True, filter=filter_name)


def sart(sinogram: ThreeDMatrix, thetas: Vector, R: Matrix, initial_image: Matrix = None) -> Matrix:
    """
    This mehtod applies the SART algorithm.
    Args:
        sinogram(Matrix): A given sinogram.
        thetas(Vector): A list of angles at which the projections were taken.
        initial_image(Matrix): A matrix which is used as the starting point for this method.

    Returns:
        A 2D matrix.
    """
    estimated_image: Matrix = iradon_sart(sinogram, thetas, image=initial_image)
    sinogram_error: Scalar = np.linalg.norm(R.dot(estimated_image.flatten()) - sinogram.flatten('F'), ord=2)
    return estimated_image, sinogram_error
