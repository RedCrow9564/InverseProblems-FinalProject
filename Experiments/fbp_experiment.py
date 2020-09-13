import numpy as np
from skimage.transform import iradon
from Infrastructure.enums import LogFields
from Infrastructure.utils import ex, Scalar, Vector, Matrix, ThreeDMatrix, DataLog
from Experiments.base_experiment import BaseExperiment


class FilteredBackprojectionExperiment(BaseExperiment):
    @ex.capture(prefix="fbp_experiment_config")
    def __init__(self, original_images: ThreeDMatrix, data_type: str, projections_number: float, 
                 fbp_filters_list: Vector):
        super(FilteredBackprojectionExperiment, self).__init__(original_images, data_type)
        self._thetas: Vector = np.linspace(0., 180., projections_number, endpoint=False)
        self._filters_list: Vector = fbp_filters_list

    def run(self) -> DataLog:
        sinograms: ThreeDMatrix = FilteredBackprojectionExperiment.radon_transform_all_images(
            self._true_images, self._thetas)

        for filter_name in self._filters_list:
            # Perform Radon Transform on every image.
            estimated_images = list()
            for sinogram in sinograms:
                estimated_image: Matrix = iradon(sinogram, theta=self._thetas, filter_name=filter_name)
                estimated_images.append(estimated_image)
            estimated_images: ThreeDMatrix = np.array(estimated_images)

            # Calc error and place all results in the log object.
            error: float = FilteredBackprojectionExperiment.estimate_error(self._true_images, estimated_images)
            self.data_log.append_dict({
                LogFields.FilterName: filter_name,
                LogFields.ProjectionsNumber: len(self._thetas),
                LogFields.DataType: self._data_type,
                LogFields.RMSError: error
            })
        
        return self.data_log  
