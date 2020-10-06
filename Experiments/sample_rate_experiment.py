import numpy as np
from Experiments.base_experiment import BaseExperiment, error_in_circle_pixels, add_noise_by_snr
from Infrastructure.enums import LogFields
from copy import deepcopy
from Infrastructure.enums import LogFields, SolverName
from Infrastructure.utils import ex, Scalar, Vector, Matrix, ThreeDMatrix, DataLog
from Solvers import get_solver
from itertools import product
from matplotlib import pyplot as plt


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
                 compared_algorithms: Vector, snr_list: Vector,
                 theta_rates: Vector, displacement_rates: Vector, _seed: int):
        log_fields = [LogFields.SolverName, LogFields.ProjectionsNumber, 
                      LogFields.SNR, LogFields.DataType, LogFields.ThetaRate,
                      LogFields.DisplacementRate, LogFields.RMSError]
        super(SampleRateExperiment, self).__init__(
            original_images, data_type, _seed, log_fields)
        self._thetas: Vector = np.linspace(0., 180., projections_number, endpoint=False)
        self._displacement_rates = displacement_rates
        self._theta_rates = theta_rates
        self._solvers_list = compared_algorithms
        self._snr_list = snr_list
        self.calculated_output_images = None


    def run(self) -> DataLog:
        output_images = list()
        sinograms: ThreeDMatrix = BaseExperiment.radon_transform_all_images(
            self._true_images, self._thetas, method="Scikit-Image")
        solver = get_solver(SolverName.FBP)

        self._noisy_sinograms = list()

        # Experiment with different SNRs
        for snr in self._snr_list:
            noisy_sinograms: ThreeDMatrix = add_noise_by_snr(
                sinograms, snr=snr, random_generator=self._rng)
            self._noisy_sinograms.append(noisy_sinograms[0])
            
            # Experiment with different algorithms ("solvers")
            for solver_name in self._solvers_list:
                solver = get_solver(solver_name)

                # Experiment for every (noisy) sinogram on the given DB
                for true_image, sinogram in zip(self._true_images, 
                                                noisy_sinograms):
                    
                    # Downsample each sinogram according to given sampling rates
                    for theta_rate, displacement_rate in product(self._theta_rates, 
                                                                self._displacement_rates):

                        # Attempt reconstructing image by using solver on the downsampled sinogram
                        downsampled_sinogram = sinogram[::displacement_rate, ::theta_rate]
                        estimated_image: Matrix = solver(downsampled_sinogram, theta=self._thetas[::theta_rate])
                        
                        # Calculate error from original image
                        print("Shape of true_image is {}".format(true_image.shape))
                        print("Shape of estimated_image is {}".format(estimated_image.shape))
                        error: Scalar = error_in_circle_pixels(true_image[::displacement_rate,::displacement_rate].reshape((1, estimated_image.shape[0], estimated_image.shape[1])), 
                                                               estimated_image.reshape((1, estimated_image.shape[0], estimated_image.shape[1])))
                        # Place results in log object
                        self.data_log.append_dict({
                            LogFields.SolverName: solver_name,
                            LogFields.ProjectionsNumber: len(self._thetas),
                            LogFields.SNR: snr,
                            LogFields.DataType: self._data_type,
                            LogFields.ThetaRate: theta_rate,
                            LogFields.DisplacementRate: displacement_rate,
                            LogFields.RMSError: error
                        })

                        output_images.append(deepcopy(estimated_image))
        
        self.calculated_output_images = output_images
        return self.data_log, output_images

    # ASSUMES ONLY ONE IMAGE!!!
    def plot(self, plot_name=None):
        if self.calculated_output_images is None:
            print("Can't plot because experiment didn't run")
            return

        # Calculated sinogram and noised sinograms for comparison:
        sinograms_fig, axes = plt.subplots(1, len(self._snr_list), figsize=(20, 20 * len(self._snr_list)))
        for i, ax in enumerate(axes):
            ax.set_title("SNR: {}".format(self._snr_list[i]))
            ax.imshow(self._noisy_sinograms[i], cmap="gray")
        if plot_name is not None:
            plt.savefig(plot_name + '_noised_sinograms.jpg')

        # Graphs of RMSError by projection downsampling rate (number of thetas) with 2 SNRs
        error_graph_fig = plt.figure(constrained_layout=True)
        rms_error_matrix = np.array(self.data_log._data[LogFields.RMSError]).reshape((len(self._snr_list), len(self._theta_rates)))
        for i, rms_error_vector in enumerate(rms_error_matrix):
            #normalized_rms_error_vector = rms_error_vector / rms_error_vector[0]
            plt.plot(self._theta_rates, rms_error_vector, ['r', 'g', 'b'][i], label="SNR={}".format(self._snr_list[i]))
        plt.xlabel('Proj rate', fontsize=18)
        plt.ylabel('Normalized RMS error', fontsize=16)
        plt.legend()
        plt.xlim(xmin=0)
        plt.ylim(ymin=0)
        if plot_name is not None:
            plt.savefig(plot_name + '_error_graph.jpg')

        # Graphs of RMSError by projection number with 2 SNRs
        error_graph2_fig = plt.figure(constrained_layout=True)
        # rms_error_matrix = np.array(self.data_log._data[LogFields.RMSError]).reshape((len(self._snr_list), len(self._theta_rates)))
        for i, rms_error_vector in enumerate(rms_error_matrix):
            plt.plot(len(self._thetas) / np.array(self._theta_rates), rms_error_vector, ['r', 'g', 'b'][i], label="SNR={}".format(self._snr_list[i]))
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

        reconst_matrix = np.array(self.calculated_output_images).reshape((len(self._snr_list), 
                                                                          len(self._theta_rates),
                                                                          *self._true_images[0].shape))
        mid = len(self._theta_rates) // 2
        l_rate_without_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[0], self._snr_list[0]))
        l_rate_without_snr_ax.imshow(reconst_matrix[0, 0], cmap="gray")
        m_rate_without_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[mid], self._snr_list[0]))
        m_rate_without_snr_ax.imshow(reconst_matrix[0, mid], cmap="gray")
        h_rate_without_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[-1], self._snr_list[0]))
        h_rate_without_snr_ax.imshow(reconst_matrix[0, -1], cmap="gray")

        l_rate_with_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[0], self._snr_list[1]))
        l_rate_with_snr_ax.imshow(reconst_matrix[1, 0], cmap="gray")
        m_rate_with_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[mid], self._snr_list[1]))
        m_rate_with_snr_ax.imshow(reconst_matrix[1, mid], cmap="gray")
        h_rate_with_snr_ax.set_title("ProjRate={}\nSNR={}".format(self._theta_rates[-1], self._snr_list[1]))
        h_rate_with_snr_ax.imshow(reconst_matrix[1, -1], cmap="gray")

        image_reconst_cmp_fig.tight_layout()
        if plot_name is not None:
            plt.savefig(plot_name + '_image_comp.jpg')
        plt.show()