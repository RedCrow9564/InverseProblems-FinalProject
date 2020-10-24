# -*- coding: utf-8 -*-
"""
fbp_experiment.py - The module which performs experiments of the FBP algorithm
=========================================================================================

This module contanis the class which performs the FBP algorithm, for various kinds of the
filter. For instance, "ramp" filter or "Hamming" filter. 
"""
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy
from Infrastructure.enums import LogFields, SolverName
from Infrastructure.utils import ex, List, Scalar, Vector, ThreeDMatrix, DataLog
from Experiments.base_experiment import BaseExperiment, add_noise_by_snr, error_in_circle_pixels
from Solvers import get_solver


class FilteredBackprojectionExperiment(BaseExperiment):
    """
    A class which is responsible for performing filtered-backprojection experiment.
    """

    @ex.capture(prefix="fbp_experiment_config")
    def __init__(self, original_images: ThreeDMatrix, data_type: str, projections_number: int, 
                 fbp_filters_list: Vector, snr_list: Vector, _seed: int):
        """
        Constructor for Filtered-Backprojection experiments objects.

        Args:
            original_images(ThreeDMatrix): A collection of 2D images where original_images[i] is the i-th image.
            
        """
        log_fields: List[str] = [
            LogFields.FilterName, LogFields.ProjectionsNumber, 
            LogFields.DataType, LogFields.SNR, LogFields.RMSError,
            LogFields.SolverName]
        super(FilteredBackprojectionExperiment, self).__init__(
            original_images, data_type, _seed, log_fields)
        self._thetas: Vector = np.linspace(0., 180., projections_number, endpoint=False)
        self._filters_list: Vector = fbp_filters_list
        self._snr_list: Vector = snr_list

    def run(self) -> DataLog:
        """
        A method for performing the filtered-backprojection experiment.

        Returns:
            A DataLog object, containing all sampled results of this experiment.
            
        """
        sinograms, _ = BaseExperiment.radon_transform_all_images(
            self._true_images, self._thetas)
        solver = get_solver(SolverName.FBP)
        self._calculated_output_images = dict()

        # Plotting every SNR value on a different figure.
        for snr in self._snr_list:
            noisy_sinograms: ThreeDMatrix = add_noise_by_snr(
                sinograms, snr=snr, random_generator=self._rng)
            min_error_per_solver = dict()

            # Iterating over all requested filters for the Filtered-Backprojection algorithm.
            for filter_name in self._filters_list:
                # Perform FBP on every sinogram.
                min_error_per_solver[filter_name] = (None, None, np.inf)
                estimated_images: ThreeDMatrix = np.empty_like(self._true_images)
                for i, sinogram in enumerate(noisy_sinograms):
                    estimated_images[i] = solver(sinogram, self._thetas, filter_name)
                    
                # Calc error and place all results in the log object.
                error: Scalar = error_in_circle_pixels(self._true_images, estimated_images)
                self.data_log.append_dict({
                    LogFields.FilterName: filter_name,
                    LogFields.ProjectionsNumber: len(self._thetas),
                    LogFields.DataType: self._data_type,
                    LogFields.SNR: snr,
                    LogFields.RMSError: error,
                    LogFields.SolverName: SolverName.FBP
                })

                if min_error_per_solver[filter_name][2] > error:
                    min_error_per_solver[filter_name] = (estimated_images.copy(), error)

            self._calculated_output_images[snr] = deepcopy(min_error_per_solver)
        
        return self.data_log, self._calculated_output_images

    @ex.capture
    def plot(self, plot_name=None):
        """
        A method for plotting the graphs we are interested in, for this specific experiment.
        """
        plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif"
        })

        # Plotting every SNR value on a different figure.
        first_snr: bool = True
        for snr in self._calculated_output_images.keys():
            fig = plt.figure(constrained_layout=True)
            plt.suptitle("\\textit{Results for SNR = " + str(snr) + ', }' + str(len(self._thetas)) +
                         "\\textit{ projections}", fontsize=16)
            gs = None
            if first_snr:
                gs = fig.add_gridspec(2, 2 + len(self._filters_list))
                true_image_ax = fig.add_subplot(gs[:, :2])

                true_image_ax.set_title("True image", fontsize=16)
                true_image_ax.imshow(self._true_images[0], cmap="gray")
            else:
                gs = fig.add_gridspec(2, len(self._filters_list))

            # Each figure contains all results for all filters.
            for index, (filter_name, (estimated_images, error)) in \
                    enumerate(self._calculated_output_images[snr].items()):
                if first_snr:
                    estimation_ax = fig.add_subplot(gs[0, 2 + index])
                else:
                    estimation_ax = fig.add_subplot(gs[0, index])
                title = r'\textit{' + filter_name.capitalize() + ' Filter}'
                estimation_ax.set_title(title, fontsize=12)
                estimation_ax.imshow(estimated_images[0], cmap="gray")

                if first_snr:
                    error_ax = fig.add_subplot(gs[1, 2 + index])
                else:
                    error_ax = fig.add_subplot(gs[1, index])
                error_ax.set_title(r'\textit{RMS Error ' + str(error)[:6] + '}', fontsize=12)
                error_ax.imshow(self._true_images[0] - estimated_images[0], cmap="gray")

            first_snr = False

        plt.show()
