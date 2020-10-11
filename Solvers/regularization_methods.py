import numpy as np
import pylops
from pylops.optimization.sparsity import SplitBregman
from pylops.optimization.leastsquares import RegularizedInversion
from Infrastructure.utils import Scalar, Vector, Matrix


def l1_regularization(
        sinogram: Matrix, theta: Vector, alpha: Scalar, 
        image_shape, R, initial_image: Matrix) -> Matrix:
    ny, nx = sinogram.shape
    vectorized_sinogram: Vector = sinogram.flatten('F')
    Dop = [pylops.Identity(ny * ny)]
    
    # TV params
    lamda = [0.1]
    niterinner = 3

    estimated_image, niter = SplitBregman(
        R, Dop, vectorized_sinogram, 1, niterinner, mu=2.0/alpha, 
        epsRL1s=lamda, tol=1e-4, tau=1., show=False, 
        x0=initial_image.flatten('F'), **dict(iter_lim=1, damp=1e-2))
    estimated_image = np.real(estimated_image.reshape(ny, ny))
    return estimated_image


def total_variation_regularization(
        sinogram: Matrix, theta: Vector, alpha: Scalar, 
        image_shape, R, initial_image: Matrix) -> Matrix:
    ny, nx = sinogram.shape
    vectorized_sinogram: Vector = sinogram.flatten('F')
    Dop = [
        pylops.FirstDerivative(ny ** 2, dims=(ny, ny), dir=0, edge=False, 
                               kind='backward', dtype=np.float64),
        pylops.FirstDerivative(ny ** 2, dims=(ny, ny), dir=1, edge=False, 
                               kind='backward', dtype=np.float64)
    ]
    
    # TV params
    lamda = [0.1, 0.1]
    niterinner = 7

    estimated_image, niter = SplitBregman(
        R, Dop, vectorized_sinogram, 1, niterinner, mu=2.0/alpha, 
        epsRL1s=lamda, tol=1e-4, tau=1., show=False, 
        x0=initial_image.flatten('F'), **dict(iter_lim=1, damp=1e-2))
    estimated_image = np.real(estimated_image.reshape(ny, ny))
    return estimated_image


def l2_regularization(
        sinogram: Matrix, theta: Vector, alpha: Scalar, 
        image_shape, R, initial_image: Matrix)  -> Matrix:
    ny, nx = sinogram.shape
    vectorized_sinogram: Vector = sinogram.flatten('F')
    estimated_image: Matrix = RegularizedInversion(
        R, [pylops.Identity(ny * ny)],  vectorized_sinogram,
        x0=initial_image.flatten('F'), 
        **dict(damp=alpha, iter_lim=1))
    estimated_image = np.real(estimated_image.reshape(ny, ny))
    return estimated_image
