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
from binascii import hexlify
from hashlib import sha256
import os

"""
Represents a sample rate experiment for CT scan simulation.
"""
class SampleRateExperiment(BaseExperiment):
    @ex.capture(prefix="sample_rate_experiment_config")
    def __init__(self, original_images, data_type, projections_number: int,
                 reconstruction_algorithm: str, snr_list: Vector,
                 theta_rates: Vector, displacement_rates: Vector, is_deterministic: bool,
                 iterations: int, fbp_filter: str, regularization_param_alpha: float, _seed: int):
        """
        Initiates a sample rate experiment, to be executed with a given
        configuration from main.py. Configuration is given via parameter
        [sample_rate_experiment_config].

        Args:
            original_images(ThreeDMatrix): A collection of 2D images
        """
        log_fields = [LogFields.SolverName, LogFields.ProjectionsNumber, 
                      LogFields.SNR, LogFields.DataType, LogFields.ImageIndex,
                      LogFields.ThetaRate, LogFields.DisplacementRate, 
                      LogFields.RMSError, LogFields.Iterations,
                      LogFields.FilterName, LogFields.RegularizarionParameter]
        super(SampleRateExperiment, self).__init__(
            original_images, data_type, _seed, log_fields)
        self._thetas: Vector = np.linspace(0., 180., projections_number, endpoint=False)
        self._displacement_rates = displacement_rates
        self._theta_rates = theta_rates
        self._solver = reconstruction_algorithm
        self._snr_list = snr_list
        self._is_deterministic = is_deterministic
        self._iterations = iterations
        self._filter = fbp_filter
        self._alpha = regularization_param_alpha

    def run(self) -> DataLog:
        """
        A method for performing the sample-experiment.

        Returns:
            A DataLog object, containing all sampled results of this experiment.

        """
        output_images = list()

        sinograms, R = BaseExperiment.radon_transform_all_images(self._true_images, self._thetas)
        
        # Experiment with given algorithm ("solvers") and make solver use cache
        solver = lambda *args : cached_solver(get_solver(self._solver), *args)

        self._noisy_sinograms_by_snr_index = list()

        # Experiment with different SNRs
        for snr in self._snr_list:
            noisy_sinograms: ThreeDMatrix = add_noise_by_snr(
                sinograms, snr=snr, random_generator=self._rng, is_deterministic=True)
            self._noisy_sinograms_by_snr_index.append(noisy_sinograms)

            # Experiment for every (noisy) sinogram on the given DB
            for image_index, (true_image, sinogram) in enumerate(zip(self._true_images, noisy_sinograms)):
                
                # Downsample each sinogram according to given sampling rates
                for theta_rate, displacement_rate in product(self._theta_rates, 
                                                             self._displacement_rates):

                    downsampled_sinogram = sinogram[::displacement_rate, ::theta_rate]
                    downsampled_thetas = self._thetas[::theta_rate]
                    estimated_image = np.zeros_like(true_image)
                    
                    print("About to run solver {}".format(self._solver))
                    print("Shape of downsampled sinogram is {}".format(downsampled_sinogram.shape))
                    print("Shape of downsampled thetas is {}".format(downsampled_thetas.shape))
                    
                    # Attempt reconstructing image by using solver on the downsampled sinogram
                    if self._solver == SolverName.FBP:
                        estimated_image: Matrix = solver(downsampled_sinogram, downsampled_thetas, self._filter)
                    elif self._solver in (SolverName.L1Regularization, SolverName.L2Regularization, 
                                          SolverName.TVRegularization, SolverName.TruncatedSVD):
                        print("Shape of R is {}".format(R.shape))
                        num_of_rays = R.shape[0] // len(self._thetas)
                        reshaped_R = R.reshape((len(self._thetas), num_of_rays, R.shape[1]))
                        downsampled_R = reshaped_R[::theta_rate, :, :].reshape((R.shape[0] // theta_rate),
                                                                                R.shape[1])
                        
                        for _ in range(self._iterations):
                            estimated_image: Matrix = solver(downsampled_sinogram,
                                                             self._alpha,
                                                             true_image.shape, 
                                                             downsampled_R, 
                                                             estimated_image)
                    elif self._solver == SolverName.SART:
                        for _ in range(self._iterations):
                            estimated_image: Matrix = solver(downsampled_sinogram, 
                                                             downsampled_thetas, 
                                                             estimated_image)
                            
                    else:
                        raise ValueError("Invalid solver {}".format(self._solver))
                    
                    # Calculate error from original image
                    print("Done running solver")
                    print("Shape of true_image is {}".format(true_image.shape))
                    print("Shape of estimated_image is {}".format(estimated_image.shape))

                    # Calculates error by comparing rescaled reconstructed downsampled sinogram to true images
                    error: Scalar = error_in_circle_pixels(np.expand_dims(true_image, 0), 
                                                           np.expand_dims(estimated_image, 0))
                    print("RMS error is {}".format(error))
                    print("Saving results to data_log")
                    # Place results in log object
                    self.data_log.append_dict({
                        LogFields.SolverName: self._solver,
                        LogFields.ProjectionsNumber: len(self._thetas),
                        LogFields.SNR: snr,
                        LogFields.DataType: self._data_type,
                        LogFields.ImageIndex: image_index,
                        LogFields.ThetaRate: theta_rate,
                        LogFields.DisplacementRate: displacement_rate,
                        LogFields.RMSError: error,
                        LogFields.Iterations: self._iterations,
                        LogFields.FilterName: self._filter,
                        LogFields.RegularizarionParameter: self._alpha
                    })

                    output_images.append(deepcopy(estimated_image))
        
        self._calculated_output_images = output_images
        return self.data_log, output_images


    def plot(self, plot_name=None):
        """
        A method for plotting the graphs we are interested in, for this specific experiment.

        Args:
            plot_name(str): String to be appended to names of files generated by this method.
        """
        if self._calculated_output_images is None:
            print("Can't plot because experiment didn't run")
            return

        # Use TeX for labels in plots
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif"
            })

        # Plot (and save to file) sinograms and noised sinograms that were generated
        # from images and used in experiment
        _, axes = plt.subplots(len(self._true_images), len(self._snr_list))
        axes = np.reshape(axes, ((len(self._true_images), len(self._snr_list))))
        for i, j in product(range(len(self._true_images)), range(len(self._snr_list))):
            plot_label = 'no noise' if self._snr_list[j] == np.inf else 'SNR = {}'.format(self._snr_list[j])
            plot_label = 'Image {} with {}'.format(i, plot_label)
            axes[i, j].set_title(textit(plot_label))
            axes[i, j].imshow(self._noisy_sinograms_by_snr_index[j][i], cmap="gray")
        save_graph(plot_name, 'noised_sinograms')

        # Plot (and save to file) graphs of RMS error by projection downsampling rate.
        # Note: Assumes 7 values from {Images X SNRs}. More than that will cause exception.
        plt.figure(constrained_layout=True)
        rms_error_matrix = np.array(self.data_log._data[LogFields.RMSError]).reshape((len(self._snr_list), 
                                                                                      len(self._true_images), 
                                                                                      len(self._theta_rates)))
        for i, j in product(range(len(self._snr_list)), range(len(self._true_images))):
            plot_label = 'no noise' if self._snr_list[i] == np.inf else 'SNR = {}'.format(self._snr_list[i])
            plot_label = 'Image {} with {}'.format(j, plot_label)
            plt.plot(self._theta_rates, rms_error_matrix[i, j, :], 
                     ['b', 'g', 'r', 'c', 'm', 'y', 'k'][(i + len(self._snr_list) * j)], 
                     label=plot_label)
        plt.xlabel(textit('Projection sampling rate'), fontsize=18)
        plt.ylabel(textit('RMS error'), fontsize=16)
        plt.legend()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        save_graph(plot_name, 'error_by_rate_graph')

        # Plot (and save to file) graphs of RMS error by number of projections.
        # Note: Assumes 7 values from {Images X SNRs}. More than that will cause exception.
        plt.figure(constrained_layout=True)
        for i, j in product(range(len(self._snr_list)), range(len(self._true_images))):
            plot_label = 'no noise' if self._snr_list[i] == np.inf else 'SNR = {}'.format(self._snr_list[i])
            plot_label = 'Image {} with {}'.format(j, plot_label)
            plt.yscale('log')
            plt.plot(len(self._thetas) / np.array(self._theta_rates), rms_error_matrix[i, j, :], 
                     ['b', 'g', 'r', 'c', 'm', 'y', 'k'][(i + len(self._snr_list) * j)], 
                     label=textit(plot_label))
        plt.xlabel(textit('Number of projections'), fontsize=18)
        plt.ylabel(textit('RMS error'), fontsize=16)
        plt.legend()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0, ymax=1.0)
        save_graph(plot_name, 'error_by_projections_graph')

        
        # Plot a method for plotting images of reconstructions with respect to their theta rates
        self.plot_snr_theta_rate_matrix(len(self._true_images), None, plot_name)
        
        plt.show()


    def plot_snr_theta_rate_matrix(self, num_of_plots=1, max_snr_cols=None, plot_name=None):
        """
        # Plots (and save to file):
        # * True image with 6 reconstructions (low/middle/high downsampling rate, 2 SNRs)
        # * Vector of reconstructions of images (taking on reconstruction for every 2 
        #   theta rates in configuration)

        Args:
            num_of_plots(int): Number of images to plot for.
            max_snr_cols(int or None): Number of SNRs to plot for. If None plot for all of them.
            plot_name(str): String to be appended to names of files generated by this method.
        """
        # Determine number of images show plots for and number of SNRs to include in figures.
        num_of_plots = min(num_of_plots, len(self._true_images))
        snr_cols = len(self._snr_list) if max_snr_cols is None else min(max_snr_cols, len(self._snr_list))
        
        # Plot for every image
        for plot_i in range(num_of_plots):
            # Create grid for plot of true image with reconstructions of
            # different SNRs and theta rates
            image_reconst_cmp_fig = plt.figure(constrained_layout=True, figsize=(10, 5))
            image_reconst_cmp_gs = image_reconst_cmp_fig.add_gridspec(3, 3 + snr_cols)
            true_image_ax = image_reconst_cmp_fig.add_subplot(image_reconst_cmp_gs[:, :3])
            for i in range(snr_cols):
                for j in range(3):
                    image_reconst_cmp_fig.add_subplot(image_reconst_cmp_gs[j, 3 + i])

            # Give figure labels and plot true image
            true_image_ax.set_title(textit("True image {}".format(plot_i)))
            true_image_ax.imshow(self._true_images[plot_i], cmap="gray")
            
            # Create reconstructions matrix for image
            reconst_matrix = np.array(self._calculated_output_images).reshape((len(self._snr_list), 
                                                                                len(self._true_images),
                                                                                len(self._theta_rates),
                                                                                *self._true_images[plot_i].shape))
            mid = len(self._theta_rates) // 2
            # Taking axes without true image
            axes = np.array(image_reconst_cmp_fig.axes[1:]).reshape((snr_cols, 3)).transpose()
            # Plot reconstruction matrix (with 3 selected theta rates: first, middle and last)
            for snr_i in range(len(self._snr_list)):
                for ax, rate_i in zip(axes[:, snr_i], (0, mid, -1)):
                    if self._snr_list[snr_i] == np.inf:
                        snr_info = 'no noise'
                    else:
                        snr_info = 'SNR={}'.format(self._snr_list[snr_i])
                    ax.set_title(textit("ProjRate={}, {}".format(self._theta_rates[rate_i], snr_info)))
                    ax.imshow(reconst_matrix[snr_i, plot_i, rate_i], cmap="gray")

            # Save graph to file
            save_graph(plot_name, 'image_{}_comp'.format(plot_i))

            # Plot reconstruction vector (taking one reconstruction for every 2 theta rates.)
            for snr_i in range(len(self._snr_list)):
                downsampled_reconstructions_fig, downsampled_reconstructions_axes = plt.subplots(1, len(self._theta_rates) // 2)

                if self._snr_list[snr_i] == np.inf:
                    snr_info = 'no noise'
                else:
                    snr_info = 'SNR={}'.format(self._snr_list[snr_i])
                downsampled_reconstructions_fig.suptitle(textit('Reconstructions by number of projections for image {}, {}'.format(plot_i, snr_info)))
                for rate_i, ax in enumerate(downsampled_reconstructions_axes):
                    ax.axis('off')
                    ax.set_title(textit("{}".format(len(self._thetas) // self._theta_rates[rate_i * 2])))
                    ax.imshow(reconst_matrix[snr_i, plot_i, rate_i * 2], cmap="gray")

                save_graph(plot_name, 'reconstructions_vector_image_{}_snr_{}_comp'.format(plot_i, snr_i))


def textit(text):
    """
    # Style text to be processed as TeX

    Args:
        text(str): Text to style

    Returns:
        str: Styled text
    """
    return r'\textit{' + text + '}'  


def cached_solver(solver, *args):
    """
    # Use solver with given arges. 
    # If cache file of specific run exists, use the its content as result.
    # Otherwise, run solver, save result in cache file and return result.

    Args:
        solver(Callable): Inverse radon transform algorithm to execute
        args(collection of objects): Arguments for solver

    Returns:
        str: Result running the solver
    """
    # Make sure cache directory exists
    if not os.path.exists('cache'):
        os.makedirs('cache')
    
    # Compute cache file name of solver call (from hash of solver + args)
    m = sha256()
    for component in (solver.__name__,) + args:
        if isinstance(component, np.ndarray):
            component_bytearray = component.tostring()
        else:
            component_bytearray = str(component).encode('utf-8')
        m.update(component_bytearray)
    file_id = str(hexlify(m.digest()).decode('utf-8'))
    print("Hash is {}".format(file_id))
    file_path = r'cache\{}.npy'.format(file_id)
    
    # Take result from cache or create cache, and then return result
    if not os.path.isfile(file_path):
        res = solver(*args)
        np.save(file_path, res)
        print('Saved cache file {}'.format(file_path))
    else:
        res = np.load(file_path)
        print('Loaded cache file {}'.format(file_path))
    return res


def save_graph(plot_name, graph_name):
    """
    # Save figure to jpg file with name {plot_name}_{graph_name}

    Args:
        plot_name(str): Prefix for file name (name of experiment, image etc.)
        graph_name(str): Postfix for file name (name of graph)
    """
    if plot_name is not None:
        print("Saving {} to file".format(graph_name))
        plt.savefig('{}_{}.jpg'.format(plot_name, graph_name), dpi=800)