import numpy as np
from scipy.sparse.linalg import aslinearoperator
import pylops
from pylops.optimization.sparsity import SplitBregman
from pylops.optimization.leastsquares import RegularizedInversion
from Infrastructure.utils import Scalar, Vector, Matrix


def l1_regularization(
        sinogram: Matrix, alpha: Scalar, image_shape, R, 
        initial_image: Matrix) -> Matrix:
    ny, nx = image_shape
    vectorized_sinogram: Vector = sinogram.flatten('F')
    Dop = [pylops.Identity(ny * nx)]
    
    # TV params
    lamda = [0.1]
    niterinner = 3

    R = aslinearoperator(R)
    estimated_image, niter = SplitBregman(
        R, Dop, vectorized_sinogram, 1, niterinner, mu=2.0/alpha, 
        epsRL1s=lamda, tol=1e-4, tau=1., show=False,
        x0=initial_image.flatten(), **dict(iter_lim=1, damp=1e-10))
    estimated_image = np.real(estimated_image.reshape(ny, ny))
    return estimated_image


def total_variation_regularization(
        sinogram: Matrix, alpha: Scalar, image_shape, R, 
        initial_image: Matrix) -> Matrix:
    ny, nx = image_shape
    vectorized_sinogram: Vector = sinogram.flatten('F')
    Dop = [
        pylops.FirstDerivative(ny ** 2, dims=(ny, nx), dir=0, edge=False, 
                               kind='backward', dtype=np.float64),
        pylops.FirstDerivative(ny ** 2, dims=(ny, nx), dir=1, edge=False, 
                               kind='backward', dtype=np.float64)
    ]
    
    # TV params
    lamda = [0.1, 0.1]
    niterinner = 7

    R = aslinearoperator(R)
    estimated_image, niter = SplitBregman(
        R, Dop, vectorized_sinogram, 1, niterinner, mu=2.0/alpha, 
        epsRL1s=lamda, tol=1e-4, tau=1., show=False,
        x0=initial_image.flatten(), **dict(iter_lim=1, damp=1e-10))
    estimated_image = np.real(estimated_image.reshape(ny, ny))
    return estimated_image


def l2_regularization(
        sinogram: Matrix, alpha: Scalar, image_shape, R, 
        initial_image: Matrix)  -> Matrix:
    ny, nx = image_shape
    vectorized_sinogram: Vector = sinogram.flatten('F')
    R = aslinearoperator(R)
    estimated_image: Matrix = RegularizedInversion(
        R, [pylops.Identity(ny * nx)],  vectorized_sinogram,
        x0=initial_image.flatten(), 
        **dict(damp=alpha, iter_lim=1))
    estimated_image = np.real(estimated_image.reshape(ny, ny))
    return estimated_image


def TSVD(sinogram: Matrix, alpha: Scalar, image_shape, R, initial_image)  -> Matrix:
    u, s, vh = np.linalg.svd(R, full_matrices=False)
    cutIndex: int = int(np.floor(s.shape[0] * alpha))
    s[cutIndex:] = 0
    s = np.reciprocal(s, where=s > 0)
    r_inverse_truncated: Matrix = vh.T @ np.diag(s) @ u.T
    
    vectorized_sinogram: Vector = sinogram.flatten('F')
    estimated_image = r_inverse_truncated @ vectorized_sinogram
    estimated_image = np.real(estimated_image).reshape(image_shape)
    return estimated_image
