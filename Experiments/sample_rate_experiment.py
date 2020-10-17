import numpy as np
from Experiments.base_experiment import BaseExperiment, error_in_circle_pixels, add_noise_by_snr
from copy import deepcopy
from Infrastructure.enums import LogFields, SolverName
from Infrastructure.utils import ex, Scalar, Vector, Matrix, ThreeDMatrix, DataLog
from Solvers import get_solver
from itertools import product
from matplotlib import pyplot as plt
import random
from skimage.transform import resize

ITERATIONS = 7

class SampleRateExperiment(BaseExperiment):
    """
    Experiments what happens when the sampling rate is change. 
    Two values are taken into consideration:
    1. The angle sampling rate
    2. The displacement sampling rate
    Given a sinogram with axes projection degree and projection displacement,
    it just mean taking some of the information (taking coarser grid)
    """
    @ex.capture(prefix="sample_rate_experiment_config")
    def __init__(self, original_images, data_type, projections_number: int,
                 reconstruction_algorithm: str, snr_list: Vector,
                 theta_rates: Vector, displacement_rates: Vector, _seed: int):
        """
        Initializes the experiment with images, metadata, parameters, etc.
        .param original_images A 3 dimensional array that represent the
                               database to test upon. First dimension is index
                               of image in DB, second and third are the images'
                               rows and columns respectively.
        """
        log_fields = [LogFields.SolverName, LogFields.ProjectionsNumber, 
                      LogFields.SNR, LogFields.DataType, 
                      LogFields.ImageIndex, LogFields.ThetaRate,
                      LogFields.DisplacementRate, LogFields.RMSError]
        super(SampleRateExperiment, self).__init__(
            original_images, data_type, _seed, log_fields)
        self._thetas: Vector = np.linspace(0., 180., projections_number, endpoint=False)
        self._displacement_rates = displacement_rates
        self._theta_rates = theta_rates
        self._solver = reconstruction_algorithm
        self._snr_list = snr_list
        self._iterations = ITERATIONS

    def run(self) -> DataLog:
        """
        Runs the experiment
        """
        output_images = list()

        sinograms, R = BaseExperiment.radon_transform_all_images(self._true_images, self._thetas)
        
        # Experiment with given algorithm ("solvers")
        solver = get_solver(self._solver)

        self._noisy_sinograms_by_snr_index = list()

        # Experiment with different SNRs
        for snr in self._snr_list:
            noisy_sinograms: ThreeDMatrix = add_noise_by_snr(
                sinograms, snr=snr, random_generator=self._rng)
            self._noisy_sinograms_by_snr_index.append(noisy_sinograms)

            # Experiment for every (noisy) sinogram on the given DB
            # for true_image, sinogram in zip(self._true_images,
            #                                 noisy_sinograms):
            for image_index, (true_image, sinogram) in enumerate(zip(self._true_images, noisy_sinograms)):
                
                # Downsample each sinogram according to given sampling rates
                for theta_rate, displacement_rate in product(self._theta_rates, 
                                                            self._displacement_rates):

                    # Attempt reconstructing image by using solver on the downsampled sinogram
                    downsampled_sinogram = sinogram[::displacement_rate, ::theta_rate]
                    downsampled_thetas = self._thetas[::theta_rate]
                    print("Size of downsampled sinogram is {}".format(downsampled_sinogram.shape))
                    print("Size of downsampled thetas is {}".format(downsampled_thetas.shape))
                    if self._solver == SolverName.FBP:
                        estimated_image: Matrix = solver(downsampled_sinogram, self._thetas[::theta_rate], 'ramp')
                    elif self._solver in (SolverName.L1Regularization, SolverName.L2Regularization, SolverName.TVRegularization):
                        ALPHA = 0.01
                        # interpolated_downsampled_sinogram = resize(downsampled_sinogram, sinogram.shape, anti_aliasing=False)
                        print("Size of R is {}".format(R.shape))
                        num_of_rays = R.shape[0] // len(self._thetas)
                        reshaped_R = R.reshape((len(self._thetas), num_of_rays, R.shape[1]))
                        downsampled_R = reshaped_R[::theta_rate, :, :].reshape((R.shape[0] // theta_rate), R.shape[1])
                        estimated_image = np.zeros_like(true_image)
                        for _ in range(self._iterations):
                            estimated_image: Matrix = solver(downsampled_sinogram,
                                                            ALPHA, true_image.shape, 
                                                            downsampled_R, 
                                                            estimated_image)
                    elif self._solver == SolverName.SART:
                        estimated_image = np.zeros_like(true_image)
                        for _ in range(self._iterations):
                            estimated_image: Matrix = solver(downsampled_sinogram, downsampled_thetas, np.zeros_like(estimated_image))
                    else:
                        raise ValueError("Invalid solver {}".format(self._solver))
                    
                    # Calculate error from original image
                    print("Shape of true_image is {}".format(true_image.shape))
                    print("Shape of estimated_image is {}".format(estimated_image.shape))
                    # Calculates error by comparing reconstructed downsampled singoram to downsampled true image
                    # error: Scalar = error_in_circle_pixels(true_image[::displacement_rate,::displacement_rate].reshape((1, estimated_image.shape[0], estimated_image.shape[1])), 
                    #                                         estimated_image.reshape((1, estimated_image.shape[0], estimated_image.shape[1])))
                    # Calculates error by comparing rescaled reconstructed downsampled sinogram to true images
                    error: Scalar = error_in_circle_pixels(np.expand_dims(true_image, 0), 
                                                           np.expand_dims(resize(estimated_image, true_image.shape, 
                                                                                 anti_aliasing=False), 0))
                    print("Error is {}".format(error))
                    # Place results in log object
                    self.data_log.append_dict({
                        LogFields.SolverName: self._solver,
                        LogFields.ProjectionsNumber: len(self._thetas),
                        LogFields.SNR: snr,
                        LogFields.DataType: self._data_type,
                        LogFields.ImageIndex: image_index,
                        LogFields.ThetaRate: theta_rate,
                        LogFields.DisplacementRate: displacement_rate,
                        LogFields.RMSError: error
                    })

                    output_images.append(deepcopy(estimated_image))
        
        self._calculated_output_images = output_images
        return self.data_log, output_images

    # ASSUMES ONLY ONE IMAGE!!!
    def plot(self, plot_name=None):
        if self._calculated_output_images is None:
            print("Can't plot because experiment didn't run")
            return

        # Calculated sinogram and noised sinograms for comparison:
        sinograms_fig, axes = plt.subplots(len(self._true_images), len(self._snr_list))
        axes = np.reshape(axes, ((len(self._true_images), len(self._snr_list))))
        for i, j in product(range(len(self._true_images)), range(len(self._snr_list))):
            axes[i, j].set_title("Pic {} with SNR {}".format(i, self._snr_list[j]))
            axes[i, j].imshow(self._noisy_sinograms_by_snr_index[j][i], cmap="gray")
        if plot_name is not None:
            plt.savefig(plot_name + '_noised_sinograms.jpg')

        # Graphs of RMSError by projection downsampling rate (number of thetas) with 2 SNRs
        error_graph_fig = plt.figure(constrained_layout=True)
        rms_error_matrix = np.array(self.data_log._data[LogFields.RMSError]).reshape((len(self._snr_list), 
                                                                                      len(self._true_images), 
                                                                                      len(self._theta_rates)))
        print(rms_error_matrix.shape)
        for i, j in product(range(len(self._snr_list)), range(len(self._true_images))):
            #normalized_rms_error_vector = rms_error_vector / rms_error_vector[0]
            print((i, j))
            print (rms_error_matrix[i, j, :].shape)
            plt.plot(self._theta_rates, rms_error_matrix[i, j, :], 
                     ['b', 'g', 'r', 'c', 'm', 'y', 'k'][(i + len(self._snr_list) * j)], 
                     label="Pic {} with SNR {}".format(j, self._snr_list[i]))
        plt.xlabel('Proj rate', fontsize=18)
        plt.ylabel('Normalized RMS error', fontsize=16)
        plt.legend()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        if plot_name is not None:
            plt.savefig(plot_name + '_error_graph.jpg')

        # Graphs of RMSError by projection number with 2 SNRs
        error_graph2_fig = plt.figure(constrained_layout=True)
        for i, j in product(range(len(self._snr_list)), range(len(self._true_images))):
            #normalized_rms_error_vector = rms_error_vector / rms_error_vector[0]
            plt.plot(len(self._thetas) / np.array(self._theta_rates), rms_error_matrix[i, j, :], 
                     ['b', 'g', 'r', 'c', 'm', 'y', 'k'][(i + len(self._snr_list) * j)], 
                     label="Pic {} with SNR {}".format(j, self._snr_list[i]))
        plt.xlabel('Proj num', fontsize=18)
        plt.ylabel('Normalized RMS error', fontsize=16)
        plt.legend()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        if plot_name is not None:
            plt.savefig(plot_name + '_error_graph2.jpg')

        # True image with 6 reconstructions (low/medium/high downsampling rate, 2 SNRs)
        image_reconst_cmp_fig = plt.figure(constrained_layout=True)
        image_reconst_cmp_gs = image_reconst_cmp_fig.add_gridspec(3, 5)
        true_image_ax = image_reconst_cmp_fig.add_subplot(image_reconst_cmp_gs[:, :3])
        l_rate_without_snr_ax = image_reconst_cmp_fig.add_subplot(image_reconst_cmp_gs[0, 3])
        m_rate_without_snr_ax = image_reconst_cmp_fig.add_subplot(image_reconst_cmp_gs[1, 3])
        h_rate_without_snr_ax = image_reconst_cmp_fig.add_subplot(image_reconst_cmp_gs[2, 3])
        l_rate_with_snr_ax = image_reconst_cmp_fig.add_subplot(image_reconst_cmp_gs[0, 4])
        m_rate_with_snr_ax = image_reconst_cmp_fig.add_subplot(image_reconst_cmp_gs[1, 4])
        h_rate_with_snr_ax = image_reconst_cmp_fig.add_subplot(image_reconst_cmp_gs[2, 4])

        true_image_ax.set_title("True image")
        true_image_ax.imshow(self._true_images[0], cmap="gray")

        reconst_matrix = np.array(self._calculated_output_images).reshape((len(self._snr_list), 
                                                                          len(self._true_images),
                                                                          len(self._theta_rates),
                                                                          *self._true_images[0].shape))
        mid = len(self._theta_rates) // 2
        l_rate_without_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[0], self._snr_list[0]))
        l_rate_without_snr_ax.imshow(reconst_matrix[0, 0, 0], cmap="gray")
        m_rate_without_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[mid], self._snr_list[0]))
        m_rate_without_snr_ax.imshow(reconst_matrix[0, 0, mid], cmap="gray")
        h_rate_without_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[-1], self._snr_list[0]))
        h_rate_without_snr_ax.imshow(reconst_matrix[0, 0, -1], cmap="gray")

        if len(self._snr_list) > 1:
            l_rate_with_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[0], self._snr_list[1]))
            l_rate_with_snr_ax.imshow(reconst_matrix[1, 0, 0], cmap="gray")
            m_rate_with_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[mid], self._snr_list[1]))
            m_rate_with_snr_ax.imshow(reconst_matrix[1, 0, mid], cmap="gray")
            h_rate_with_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[-1], self._snr_list[1]))
            h_rate_with_snr_ax.imshow(reconst_matrix[1, 0, -1], cmap="gray")

        image_reconst_cmp_fig.tight_layout()
        if plot_name is not None:
            plt.savefig(plot_name + '_image_comp.jpg')
        plt.show()

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)