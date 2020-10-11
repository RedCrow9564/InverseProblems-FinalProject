import numpy as np
from copy import deepcopy
from Infrastructure.enums import LogFields, SolverName
from Infrastructure.utils import ex, Scalar, Vector, ThreeDMatrix, \
    DataLog, List
from Experiments.base_experiment import BaseExperiment, add_noise_by_snr, error_in_circle_pixels
from Solvers import get_solver


class IterationsExperiment(BaseExperiment):
    @ex.capture(prefix="iterations_experiment_config")
    def __init__(self, original_images: ThreeDMatrix, data_type: str, 
                 projections_number: int, compared_algorithms: Vector, 
                 snr_list: Vector, alpha: Scalar, max_iterations:int, 
                 _seed: int):
        log_fields: List[str] = [
            LogFields.SolverName, LogFields.ProjectionsNumber,
            LogFields.DataType, LogFields.Iterations, LogFields.SNR,
            LogFields.RegularizarionParameter, LogFields.RMSError]
        super(IterationsExperiment, self).__init__(
            original_images, data_type, _seed, log_fields)
        self._thetas: Vector = np.linspace(0., 180., projections_number, endpoint=False)
        self._solvers_list: List[str] = compared_algorithms
        self._snr_list: Vector = snr_list
        self._alpha: Scalar = alpha
        self._max_iterations: int = max_iterations

    def run(self) -> DataLog:
        output_images = list()
        image_shape = self._true_images.shape[1:]
        sinograms, R = BaseExperiment.radon_transform_all_images(
            self._true_images, self._thetas)
        
        for snr in self._snr_list:
            noisy_sinograms: ThreeDMatrix = add_noise_by_snr(
                sinograms, snr=snr, random_generator=self._rng)
            
            for solver_name in self._solvers_list:
                solver = get_solver(solver_name)
                is_sart: bool = solver_name == SolverName.SART
                estimated_images: ThreeDMatrix = np.zeros_like(
                    self._true_images)
                
                for i in range(self._max_iterations):
                    
                    # Perform the Inverse Radon Transform on every sinogram.
                    for j, sinogram in enumerate(noisy_sinograms):
                        if is_sart:
                            estimated_images[j] = solver(
                                sinogram, self._thetas, estimated_images[j])
                        else:
                            estimated_images[j] = solver(
                                sinogram, self._thetas, self._alpha, 
                                image_shape, R, estimated_images[j])
                
    
                    # Calc error and place all results in the log object.
                    error = error_in_circle_pixels(self._true_images, estimated_images)
                    self.data_log.append_dict({
                        LogFields.SolverName: solver_name,
                        LogFields.ProjectionsNumber: len(self._thetas),
                        LogFields.DataType: self._data_type,
                        LogFields.Iterations: i + 1,
                        LogFields.SNR: snr,
                        LogFields.RMSError: error,
                        LogFields.RegularizarionParameter: "None" if is_sart else self._alpha
                    })
                    output_images.append(deepcopy(estimated_images))
        
        return self.data_log, output_images