import numpy as np
from copy import deepcopy
from Infrastructure.enums import LogFields, SolverName
from Infrastructure.utils import ex, List, Scalar, Vector, ThreeDMatrix, DataLog
from Experiments.base_experiment import BaseExperiment, add_noise_by_snr, error_in_circle_pixels
from Solvers import get_solver


class FilteredBackprojectionExperiment(BaseExperiment):
    @ex.capture(prefix="fbp_experiment_config")
    def __init__(self, original_images: ThreeDMatrix, data_type: str, projections_number: int, 
                 fbp_filters_list: Vector, snr_list: Vector, _seed: int):
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
        output_images = list()
        sinograms: ThreeDMatrix = BaseExperiment.radon_transform_all_images(
            self._true_images, self._thetas, method="Scikit-Image")
        solver = get_solver(SolverName.FBP)

        for snr in self._snr_list:
            noisy_sinograms: ThreeDMatrix = add_noise_by_snr(
                sinograms, snr=snr, random_generator=self._rng)

            for filter_name in self._filters_list:
                # Perform FBP on every sinogram.
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
                output_images.append(deepcopy(estimated_images))
        
        return self.data_log, output_images
