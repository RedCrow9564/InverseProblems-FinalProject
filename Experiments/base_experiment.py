import numpy as np
from numpy.random import Generator, PCG64
from skimage.transform import radon
from Infrastructure.utils import Scalar, Vector, Matrix, ThreeDMatrix, List, DataLog


def error_in_circle_pixels(true_images: Matrix, estimated_images: Matrix) -> float:
    img_shape: Vector = np.array(true_images.shape[1:])
    radius: float = min(img_shape) // 2
    coords = np.array(np.ogrid[:img_shape[0], :img_shape[1]],
                      dtype=object)
    dist = ((coords - img_shape // 2) ** 2).sum(0)
    indices_mask = dist <= radius ** 2
    error: Scalar = np.mean(np.power(true_images[:, indices_mask] - \
              estimated_images[:, indices_mask], 2))
    error = np.sqrt(np.mean(error))
    print(error)
    return error



def add_noise_by_snr(images: ThreeDMatrix, snr: float, random_generator) -> ThreeDMatrix:
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
        sinograms: ThreeDMatrix = np.empty(
            (images.shape[0], images.shape[1], len(thetas)), dtype=np.float64)
        
        for i, image in enumerate(images):
            sinograms[i] = radon(image, thetas, circle=True)
        weights: Matrix = create_weights(image.shape, thetas, image.shape[0])
        return sinograms, weights
        

def _get_w_for_ray(imageShape, theta, iproj):
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
    radius =imageShape[0] // 2 - 1
    projection_center = imageShape[0] // 2
    rotation_center = imageShape[0] // 2
    # (s, t) is the (x, y) system rotated by theta
    t = iproj - projection_center
    # s0 is the half-length of the ray's path in the reconstruction circle
    s0 = np.sqrt(radius * radius - t * t) if radius*radius >= t*t else 0.
    Ns = 2 * int(np.ceil(2 * s0))  # number of steps
                                               # along the ray

    W = np.zeros(imageShape)
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
                #ray_sum += weight * image[i, j]
              #  weight_norm += weight * weight
                W[i,j] = W[i,j] +  weight
            if i > 0 and j < imageShape[1] - 1:
                weight = (1. - di) * dj * ds
              #  ray_sum += weight * image[i, j+1]
              #  weight_norm += weight * weight
                W[i,j + 1] = W[i,j + 1] + weight
            if i < imageShape[0] - 1 and j > 0:
                weight = di * (1 - dj) * ds
             #   ray_sum += weight * image[i+1, j]
             #   weight_norm += weight * weight
                W[i + 1,j] = W[i + 1,j] + weight
            if i < imageShape[0] - 1 and j < imageShape[1] - 1:
                weight = di * dj * ds
                W[i + 1,j + 1] = W[i + 1,j + 1] + weight
               # ray_sum += weight * image[i+1, j+1]
               # weight_norm += weight * weight
    W = W.flatten()
    return W


def create_weights(imageShape, theta, numerOfProjections):
    """Inverse radon transform.

    Reconstruct an image from the radon transform, using a single iteration of
    the Simultaneous Algebraic Reconstruction Technique (SART) algorithm.

    Parameters
    ----------
    radon_image : 2D array, dtype=float
        Image containing radon transform (sinogram). Each column of
        the image corresponds to a projection along a different angle. The
        tomography rotation axis should lie at the pixel index
        ``radon_image.shape[0] // 2`` along the 0th dimension of
        ``radon_image``.
    theta : 1D array, dtype=float, optional
        Reconstruction angles (in degrees). Default: m angles evenly spaced
        between 0 and 180 (if the shape of `radon_image` is (N, M)).
    image : 2D array, dtype=float, optional
        Image containing an initial reconstruction estimate. Shape of this
        array should be ``(radon_image.shape[0], radon_image.shape[0])``. The
        default is an array of zeros.
    projection_shifts : 1D array, dtype=float, optional
        Shift the projections contained in ``radon_image`` (the sinogram) by
        this many pixels before reconstructing the image. The i'th value
        defines the shift of the i'th column of ``radon_image``.
    clip : length-2 sequence of floats, optional
        Force all values in the reconstructed tomogram to lie in the range
        ``[clip[0], clip[1]]``
    relaxation : float, optional
        Relaxation parameter for the update step. A higher value can
        improve the convergence rate, but one runs the risk of instabilities.
        Values close to or higher than 1 are not recommended.

    Returns
    -------
    reconstructed : ndarray
        Reconstructed image. The rotation axis will be located in the pixel
        with indices
        ``(reconstructed.shape[0] // 2, reconstructed.shape[1] // 2)``.

    Notes
    -----
    Algebraic Reconstruction Techniques are based on formulating the tomography
    reconstruction problem as a set of linear equations. Along each ray,
    the projected value is the sum of all the values of the cross section along
    the ray. A typical feature of SART (and a few other variants of algebraic
    techniques) is that it samples the cross section at equidistant points
    along the ray, using linear interpolation between the pixel values of the
    cross section. The resulting set of linear equations are then solved using
    a slightly modified Kaczmarz method.

    When using SART, a single iteration is usually sufficient to obtain a good
    reconstruction. Further iterations will tend to enhance high-frequency
    information, but will also often increase the noise.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] AH Andersen, AC Kak, "Simultaneous algebraic reconstruction
           technique (SART): a superior implementation of the ART algorithm",
           Ultrasonic Imaging 6 pp 81--94 (1984)
    .. [3] S Kaczmarz, "Angenäherte auflösung von systemen linearer
           gleichungen", Bulletin International de l’Academie Polonaise des
           Sciences et des Lettres 35 pp 355--357 (1937)
    .. [4] Kohler, T. "A projection access scheme for iterative
           reconstruction based on the golden section." Nuclear Science
           Symposium Conference Record, 2004 IEEE. Vol. 6. IEEE, 2004.
    .. [5] Kaczmarz' method, Wikipedia,
           https://en.wikipedia.org/wiki/Kaczmarz_method

    """

    
    
    a1 = theta.shape[0] * numerOfProjections
    a2 =  imageShape[0] * imageShape[1]
    weights = np.zeros((a1,a2), dtype=np.float)
    counter = 0
    for angle_index in range(theta.shape[0]):
        for i_proj in range(numerOfProjections):
            v = _get_w_for_ray(imageShape, theta[angle_index], i_proj)
            weights[counter, :] = v
            counter += 1
    return weights
