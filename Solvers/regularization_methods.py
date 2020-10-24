# -*- coding: utf-8 -*-
"""
regularization_methods.py - Regularization algorithms module.
================================================================

This module contains methods for regularization methods for the
Inverse Radon-Transform problem. Available methods are L1, L2 and TV
regularization, in addition to the truncated-SVD method.

"""
import numpy as np
from scipy.sparse.linalg import aslinearoperator
import pylops
from pylops.optimization.sparsity import SplitBregman
from pylops.optimization.leastsquares import RegularizedInversion
from Infrastructure.utils import Scalar, Vector, Matrix


def l1_regularization(sinogram: Matrix, alpha: Scalar, image_shape, R: Matrix, initial_image: Matrix) -> Matrix:
    """
    L1-Regularization methods.

    Args:
        sinogram(Matrix): A Numpy 2D matrix.
        alpha(Scalar): A real scalar, the regularization parameter.
        image_shape(2-Tuple): The shape of the expected image.
        R(Matrix): The Discrete Radon-Transform operator.
        initial_image(Matrix): An initial guess for the reconstructed image.
    
    Returns:
        A 2D Matrix, the reconstructed image.
    """
    vectorized_sinogram: Vector = sinogram.flatten('F')
    Dop = [pylops.Identity(np.prod(image_shape))]
    
    # TV params
    lamda = [0.1]
    niterinner = 3

    R = aslinearoperator(R)
    estimated_image, _ = SplitBregman(
        R, Dop, vectorized_sinogram, 1, niterinner, mu=2.0/alpha, 
        epsRL1s=lamda, tol=1e-4, tau=1., show=False,
        x0=initial_image.flatten(), **dict(iter_lim=1, damp=1e-10))
    sinogram_error: Scalar = np.linalg.norm(R * estimated_image - vectorized_sinogram, ord=2)

    # Reshaping the output to the expected image shape
    estimated_image = np.real(estimated_image.reshape(image_shape))
    return estimated_image, sinogram_error


def total_variation_regularization(sinogram: Matrix, alpha: Scalar, image_shape, R: Matrix,
                                   initial_image: Matrix) -> Matrix:
    """
    Total-Variation Regularization methods.

    Args:
    sinogram(Matrix): A Numpy 2D matrix.
    alpha(Scalar): A real scalar, the regularization parameter.
    image_shape(2-Tuple): The shape of the expected image.
    R(Matrix): The Discrete Radon-Transform operator.
    initial_image(Matrix): An initial guess for the reconstructed image.
    """
    image_size: int = np.prod(image_shape)
    vectorized_sinogram: Vector = sinogram.flatten('F')
    Dop = [
        pylops.FirstDerivative(image_size, dims=image_shape, dir=0, edge=False, 
                               kind='backward', dtype=np.float64),
        pylops.FirstDerivative(image_size, dims=image_shape, dir=1, edge=False, 
                               kind='backward', dtype=np.float64)
    ]
    
    # TV params
    lamda = [0.1, 0.1]
    niterinner = 7

    R = aslinearoperator(R)
    estimated_image, _ = SplitBregman(
        R, Dop, vectorized_sinogram, 1, niterinner, mu=2.0/alpha, 
        epsRL1s=lamda, tol=1e-4, tau=1., show=False,
        x0=initial_image.flatten(), **dict(iter_lim=1, damp=1e-10))
    sinogram_error: Scalar = np.linalg.norm(R * estimated_image - vectorized_sinogram, ord=2)
    estimated_image = np.real(estimated_image.reshape(image_shape))
    return estimated_image, sinogram_error


def l2_regularization(sinogram: Matrix, alpha: Scalar, image_shape, R: Matrix, initial_image: Matrix) -> Matrix:
    image_size: int = np.prod(image_shape)
    vectorized_sinogram: Vector = sinogram.flatten('F')
    R = aslinearoperator(R)
    estimated_image: Matrix = RegularizedInversion(
        R, [pylops.Identity(image_size)], vectorized_sinogram,
        x0=initial_image.flatten(), 
        **dict(damp=alpha, iter_lim=1))
    sinogram_error: Scalar = np.linalg.norm(R * estimated_image - vectorized_sinogram, ord=2)
    estimated_image = np.real(estimated_image.reshape(image_shape))
    return estimated_image, sinogram_error


def TSVD(sinogram: Matrix, alpha: Scalar, image_shape, R: Matrix, initial_image) -> Matrix:
    u, s, vh = np.linalg.svd(R, full_matrices=False)
    cut_index: int = int(np.floor(s.shape[0] * alpha))
    s[cut_index:] = 0
    s = np.reciprocal(s, where=s > 0)
    r_inverse_truncated: Matrix = vh.T @ np.diag(s) @ u.T
    
    vectorized_sinogram: Vector = sinogram.flatten('F')
    estimated_image = r_inverse_truncated @ vectorized_sinogram
    estimated_image = np.real(estimated_image).reshape(image_shape)
    sinogram_error: Scalar = np.linalg.norm(R * estimated_image - vectorized_sinogram, ord=2)
    return estimated_image, sinogram_error
