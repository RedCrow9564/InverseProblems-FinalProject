import numpy as np
from Experiments.base_experiment import BaseExperiment
from Infrastructure.enums import LogFields
from copy import deepcopy
from Infrastructure.enums import LogFields, SolverName
from Infrastructure.utils import ex, Scalar, Vector, Matrix, ThreeDMatrix, DataLog
from Solvers import get_solver


class SampleRateExperiment(BaseExperiment):
    """
    Experiments what happens when the sampling rate is change. 
    Two values are taken into consideration:
    1. The angle sampling rate
    2. The displacement sampling rate
    Given a sinogram with axes projection degree and projection displacement,
    it just mean taking some of the information (taking coarser grid)
    """
    def __init__(self, original_images, data_type, snr_list, theta_num_list,
                 beam_num_list, _seed):
        log_fields = [LogFields.SolverName, LogFields.ProjectionsNumber,
                      LogFields.DataType, LogFields.Iterations, LogFields.SNR,
                      LogFields.RMSError]
        # Initializes self._true_images, self._data_type,
        # self._save_estimated_images, self.data_log, self._rng
        super(SampleRateExperiment, self).__init__(
            original_images, data_type, _seed, log_fields)
        self._thetas_list = map(lambda x: np.linspace(0., 180., x, endpoint=False),
                                theta_num_list)
        # self._solvers_list = compared_algorithms
        self._snr_list = snr_list
        self._alpha = alpha
        self._max_iterations = max_iterations

    def run(self) -> DataLog:
        output_images = list()
        sinograms: ThreeDMatrix = BaseExperiment.radon_transform_all_images(
            self._true_images, self._thetas, method="Scikit-Image")
        solver = get_solver(SolverName.FBP)

        estimated_images = list()

        for sinogram in sinograms:
            for displacement_rate in self._displacement_rates:
                for theta_rate in self._theta_rates:
                    downsampled_sinogram = sinogram[::displacement_rate, ::theta_rate]
                    thetas = self._thetas = self._thetas[::theta_rate]
            estimated_image: Matrix = solver(downsampled_sinogram, theta=self._thetas)
            estimated_images.append(estimated_image)

        estimated_images: ThreeDMatrix = np.array(estimated_images)

        

        # Perform Radon Transform on every image.
                # estimated_images = list()
                # for sinogram in noisy_sinograms:
                #     estimated_image: Matrix = solver(sinogram, theta=self._thetas, filter_name=filter_name)
                #     estimated_images.append(estimated_image)
                # estimated_images: ThreeDMatrix = np.array(estimated_images)

        # Calc error and place all results in the log object.
        error: Scalar = error_in_circle_pixels(self._true_images, estimated_images)
        self.data_log.append_dict({
            # LogFields.ProjectionsNumber: len(self._thetas),
            LogFields.DataType: self._data_type,
            # LogFields.SNR: snr,
            LogFields.RMSError: error,
            LogFields.SolverName: SolverName.FBP
        })
        output_images.append(deepcopy(estimated_images))
        
        return self.data_log, output_images
                

        # # THEIRS

        # for snr in self._snr_list:
        #     noisy_sinograms: ThreeDMatrix = add_noise_by_snr(
        #         sinograms, snr=snr, random_generator=self._rng)

        #     for filter_name in self._filters_list:
        #         # Perform Radon Transform on every image.
        #         estimated_images = list()
        #         for sinogram in noisy_sinograms:
        #             estimated_image: Matrix = solver(sinogram, theta=self._thetas, filter_name=filter_name)
        #             estimated_images.append(estimated_image)
        #         estimated_images: ThreeDMatrix = np.array(estimated_images)

        #         # Calc error and place all results in the log object.
        #         error: Scalar = error_in_circle_pixels(self._true_images, estimated_images)
        #         self.data_log.append_dict({
        #             LogFields.FilterName: filter_name,
        #             #LogFields.ProjectionsNumber: len(self._thetas),
        #             LogFields.DataType: self._data_type,
        #             LogFields.SNR: snr,
        #             LogFields.RMSError: error,
        #             LogFields.SolverName: SolverName.FBP
        #         })
        #         output_images.append(deepcopy(estimated_images))
        
        # return self.data_log, output_images
