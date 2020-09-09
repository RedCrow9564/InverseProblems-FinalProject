from Experiments.base_experiment import BaseExperiment
from Experiments.sample_rate_experiment import SampleRateExperiment
from Experiments.fbp_experiment import FilteredBackprojectionExperiment
from Experiments.iterations_experiment import IterationsExperiment


class ExperimentBuilder:
    @staticmethod
    def create_experiment(experiment_type) -> BaseExperiment:
        pass

    @staticmethod
    def _create_sample_rate_experiment() -> SampleRateExperiment:
        pass

    @staticmethod
    def _create_FBP_experiment() -> FilteredBackprojectionExperiment:
        pass

    @staticmethod
    def _create_iterations_experiment() -> IterationsExperiment:
        pass


