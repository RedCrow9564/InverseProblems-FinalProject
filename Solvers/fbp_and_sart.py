from skimage.transform import iradon, iradon_sart
from Infrastructure.utils import Vector, Matrix, ThreeDMatrix


def filtered_back_projection(sinogram: Matrix, thetas: Vector, 
                             filter_name: str) -> Matrix:
    return iradon(sinogram, thetas, circle=True, filter_name=filter_name)


def sart(sinogram: ThreeDMatrix, thetas: Vector, 
         initial_image: Matrix = None) -> Matrix:
    return iradon_sart(sinogram, thetas, image=initial_image,
                       dtype=sinogram.dtype)
