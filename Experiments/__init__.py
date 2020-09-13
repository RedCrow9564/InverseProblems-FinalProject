from Experiments.base_experiment import BaseExperiment
from Experiments.sample_rate_experiment import SampleRateExperiment
from Experiments.fbp_experiment import FilteredBackprojectionExperiment
from Experiments.iterations_experiment import IterationsExperiment
from Infrastructure.utils import ThreeDMatrix
from Infrastructure.enums import ExperimentType



class ExperimentBuilder:
    @staticmethod
    def create_experiment(experiment_type: str, true_images: ThreeDMatrix, data_type: str) -> BaseExperiment:
        if experiment_type not in _type_to_constructor:
            raise ValueError("Object type {0} is NOT supported".format(experiment_type))
        else:
            return _type_to_constructor[experiment_type](true_images, data_type)
        

    @staticmethod
    def _create_sample_rate_experiment(true_images: ThreeDMatrix, data_type: str) -> SampleRateExperiment:
        pass

    @staticmethod
    def _create_FBP_experiment(true_images: ThreeDMatrix, data_type: str) -> FilteredBackprojectionExperiment:
        return FilteredBackprojectionExperiment(true_images, data_type)

    @staticmethod
    def _create_iterations_experiment(true_images: ThreeDMatrix, data_type: str) -> IterationsExperiment:
        pass


_type_to_constructor = {
    ExperimentType.SampleRateExperiment: ExperimentBuilder._create_sample_rate_experiment,
    ExperimentType.FBPExperiment: ExperimentBuilder._create_FBP_experiment,
    ExperimentType.IterationsExperiment: ExperimentBuilder._create_iterations_experiment
}

