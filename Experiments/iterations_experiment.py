# -*- coding: utf-8 -*-
"""
iterations_experiment.py - The module which performs experiments of iterative algorithms.
=========================================================================================

This module contains the class which performs iterative algorithms and tests their performance
for each iteration.
"""
import numpy as np
from matplotlib import pyplot as plt
from itertools import product
from copy import deepcopy
from Infrastructure.enums import LogFields, SolverName
from Infrastructure.utils import ex, Scalar, Vector, ThreeDMatrix, \
    DataLog, List
from Experiments.base_experiment import BaseExperiment, add_noise_by_snr, error_in_circle_pixels
from Solvers import get_solver


class IterationsExperiment(BaseExperiment):
    """
    A class which is responsible for performing iterations experiment.
    """

    @ex.capture(prefix="iterations_experiment_config")
    def __init__(self, original_images: ThreeDMatrix, data_type: str,
                 projections_number: int, compared_algorithms: Vector,
                 snr_list: Vector, alphas_list: Vector, max_iterations: int,
                 _seed: int):
        """
        Constructor for iterations experiments objects.

        Args:
            original_images(ThreeDMatrix): A collection of 2D images where original_images[i] is the i-th image.

        """
        log_fields: List[str] = [
            LogFields.SolverName, LogFields.ProjectionsNumber,
            LogFields.DataType, LogFields.Iterations, LogFields.SNR,
            LogFields.RegularizarionParameter, LogFields.RMSError]
        super(IterationsExperiment, self).__init__(
            original_images, data_type, _seed, log_fields)
        self._thetas: Vector = np.linspace(0., 180., projections_number, endpoint=False)
        self._solvers_list: List[str] = compared_algorithms
        self._snr_list: Vector = snr_list
        self._alphas_list: Scalar = alphas_list
        self._max_iterations: int = max_iterations

    def run(self) -> DataLog:
        """
        A method for performing the iterations-experiment.

        Returns:
            A DataLog object, containing all sampled results of this experiment.

        """
        self._calculated_output_images = dict()

        image_shape = self._true_images.shape[1:]
        sinograms, R = BaseExperiment.radon_transform_all_images(
            self._true_images, self._thetas)

        # Iterating over all requested SNR values. 
        for snr in self._snr_list:

            # Creating the noisy sinograms.
            noisy_sinograms: ThreeDMatrix = add_noise_by_snr(
                sinograms, snr=snr, random_generator=self._rng)

            is_sart_run: bool = False
            min_error_per_solver = dict()
            for solver_name in self._solvers_list:
                min_error_per_solver[solver_name] = (None, None, -1, np.inf)

            # Iterating over all regularization parameters and all requested solvers.
            for solver_name, alpha in product(self._solvers_list, self._alphas_list):
                solver = get_solver(solver_name)

                # For SART, the algorithm is run only once (since it does not depend on regularization).
                is_sart: bool = solver_name == SolverName.SART
                if is_sart and is_sart_run:
                    continue

                # Initializing the estimated images to all zeroes images.
                estimated_images: ThreeDMatrix = np.zeros_like(
                    self._true_images)

                # Performing all iterations.
                for i in range(self._max_iterations):

                    # Perform the Inverse Radon Transform on every sinogram.
                    for j, sinogram in enumerate(noisy_sinograms):
                        if is_sart:
                            estimated_images[j] = solver(
                                sinogram, self._thetas, estimated_images[j])
                        else:
                            estimated_images[j] = solver(
                                sinogram, alpha, image_shape, R,
                                estimated_images[j])

                    # Calc error and place all results in the log object.
                    error = error_in_circle_pixels(self._true_images, estimated_images)
                    alpha_str: str = "None" if is_sart else alpha
                    self.data_log.append_dict({
                        LogFields.SolverName: solver_name,
                        LogFields.ProjectionsNumber: len(self._thetas),
                        LogFields.DataType: self._data_type,
                        LogFields.Iterations: i + 1,
                        LogFields.SNR: snr,
                        LogFields.RMSError: error,
                        LogFields.RegularizarionParameter: alpha_str
                    })

                    # Keeping (for plotting) the alpha and estimated image,
                    # for which the reconstruction error is minimal.
                    if min_error_per_solver[solver_name][3] > error:
                        min_error_per_solver[solver_name] = (estimated_images.copy(), alpha_str, i + 1, error)

                if is_sart:
                    is_sart_run = True

            self._calculated_output_images[snr] = deepcopy(min_error_per_solver)

        return self.data_log, self._calculated_output_images

    def plot(self, plot_name=None):
        """
        A method for plotting the graphs we are interested in, for this specific experiment.
        """
        df = self.data_log.to_pandas()
        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif"
            })

        # Plotting every SNR value on a different figure.
        groupby_first_columns = [LogFields.SNR]
        groupby_second_columns = [LogFields.SolverName, LogFields.RegularizarionParameter]
        for name, group in df.groupby(groupby_first_columns):
            print(name)
            fig = plt.figure()
            ax = fig.gca()

            # Each figure contains all results for all solvers and all regularization parameters.
            for name2, inner_group in group.groupby(groupby_second_columns):
                print(name2)
                graph_label = r'\textit{' + name2[0] + ', } '
                if name2[1] != 'None':
                    graph_label += r'$\alpha$ = ' + str(name2[1])
                inner_group.plot(x=LogFields.Iterations, y=LogFields.RMSError,
                                 ax=ax, label=graph_label)
                plt.xlabel('\\textit{' + LogFields.Iterations + '}', fontsize=12)
                plt.ylabel('\\textit{' + LogFields.RMSError + '}', fontsize=12)
                title = 'Solvers RMS Error for SNR = ' + str(name) + ", " + str(len(self._thetas)) + " projections"
                title = '\\textit{' + title + '}'
                plt.title(title, fontsize=16)

        # Plotting the reconstructed images. Every SNR value on a different figure.
        for snr in self._calculated_output_images.keys():
            fig = plt.figure(constrained_layout=True)
            plt.suptitle("\\textit{Results for SNR = " + str(snr) + ', }' + str(len(self._thetas)) +
                         "\\textit{ projections}", fontsize=16)
            gs = fig.add_gridspec(2, 2 + len(self._solvers_list))
            true_image_ax = fig.add_subplot(gs[:, :2])

            true_image_ax.set_title("True image", fontsize=16)
            true_image_ax.imshow(self._true_images[0], cmap="gray")

            # Each figure contains all results for all solvers. Only the best regularization parameters are shown here.
            for index, (solver_name, (estimated_images, alpha, iterations, error)) in \
                    enumerate(self._calculated_output_images[snr].items()):
                estimation_ax = fig.add_subplot(gs[0, 2 + index])
                title = r'\textit{' + solver_name + '}'
                estimation_ax.set_title(title, fontsize=16)
                estimation_ax.imshow(estimated_images[0], cmap="gray")

                error_ax = fig.add_subplot(gs[1, 2 + index])
                title = r'\textit{' + str(iterations) + ' iteration'
                if iterations > 1:
                    title += 's'
                title += '}'

                if alpha != "None":
                    title += r'\textit{, }$\alpha$ = \textit{' + str(alpha) + '}'
                error_ax.set_title(title, fontsize=16)
                error_ax.set_xlabel(r'\textit{RMS Error ' + str(error)[:6] + '}', fontsize=16)
                error_ax.imshow(self._true_images[0] - estimated_images[0], cmap="gray")

        plt.show()
