import numpy as np
from numba import jit
import pylops
from skimage.transform import iradon
from Infrastructure.utils import Callable, Dict, Scalar, Vector, Matrix, ThreeDMatrix
from Infrastructure.enums import SolverName


def _radon_curve_image(image_shape):
    ny, nx = image_shape
    
    @jit(nopython=True)
    def radoncurve(x, r, theta):
        return (r - ny//2)/(np.sin(np.deg2rad(theta))+1e-15) + np.tan(np.deg2rad(90 - theta))*x  + ny//2
    return radoncurve


def l1_regularization(sinograms: ThreeDMatrix, iterations_num: int, 
                      theta: Vector, alpha: Scalar, image_shape, R):
    images_num = sinograms.shape[0]
    ny, nx = image_shape
    vectorized_sinograms: Vector = sinograms.flatten('F')  #reshape((images_num, -1)).T
    print(vectorized_sinograms.shape)
    pfista, niterf = pylops.optimization.sparsity.FISTA(
        R, vectorized_sinograms, niter=iterations_num, eps=alpha, tol=1e-7)
    xinv = np.real(pfista.reshape(images_num, nx, ny))
    xinv = np.transpose(xinv, axes=(0, 2, 1))
    return xinv


_name_to_solver: Dict = {
        SolverName.FBP: iradon,
        SolverName.L1Regularization: l1_regularization
    }
    
def get_solver(solver_name: str) -> Callable:    
    return _name_to_solver[solver_name]