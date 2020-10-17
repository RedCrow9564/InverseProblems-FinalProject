
# -*- coding: utf-8 -*-
"""
Experiments\\__init__.py - Factory of experiment type objects.
=================================================================

This module creates a specific experiment object. The experiment is chosen 
according to a given experiment-type name. 
For example:

    ExperimentBuilder.create_experiment(ExperimentType.IterationsExperiment) -
        Returns the iterations-experiment class constructor.

"""
from Experiments.base_experiment import BaseExperiment
from Experiments.sample_rate_experiment import SampleRateExperiment
from Experiments.fbp_experiment import FilteredBackprojectionExperiment
from Experiments.iterations_experiment import IterationsExperiment
from Infrastructure.utils import ThreeDMatrix
from Infrastructure.enums import ExperimentType


class ExperimentBuilder:
    """
    A static class which is responsible for the generation of all experiment objects, via the
    create_experiment method.
    """
    @staticmethod
    def create_experiment(experiment_type: str, true_images: ThreeDMatrix, data_type: str) -> BaseExperiment:
        """
        This function generates the specific experiment, whose name is given as input.

        Args:
            experiment_type(str): A name of a specific experiment type, to be created.
            true_images(ThreeDMatrix): A database of images.
            data_type(str): The name of the given database. A DBType variable name.

        Returns:
            An experiment object.
        """ 
        if experiment_type not in _type_to_constructor:
            raise ValueError("Object type {0} is NOT supported".format(experiment_type))
        else:
            return _type_to_constructor[experiment_type](true_images, data_type)
        
    @staticmethod
    def create_sample_rate_experiment(true_images: ThreeDMatrix, data_type: str) -> SampleRateExperiment:
        """
        This function generates a SampleRateExperiment object, with the give images as input.

        Args:
            true_images(ThreeDMatrix): A database of images.
            data_type(str): The name of the given database. A DBType variable name.

        Returns:
            An SampleRateExperiment object.
        """ 
        return SampleRateExperiment(true_images, data_type)

    @staticmethod
    def create_FBP_experiment(true_images: ThreeDMatrix, data_type: str) -> FilteredBackprojectionExperiment:
        """
        This function generates a FilteredBackprojectionExperiment object, with the give images as input.

        Args:
            true_images(ThreeDMatrix): A database of images.
            data_type(str): The name of the given database. A DBType variable name.

        Returns:
            An FilteredBackprojectionExperiment object.
        """ 
        return FilteredBackprojectionExperiment(true_images, data_type)

    @staticmethod
    def create_iterations_experiment(true_images: ThreeDMatrix, data_type: str) -> IterationsExperiment:
        """
        This function generates a IterationsExperiment object, with the give images as input.

        Args:
            true_images(ThreeDMatrix): A database of images.
            data_type(str): The name of the given database. A DBType variable name.

        Returns:
            An IterationsExperiment object.
        """ 
        return IterationsExperiment(true_images, data_type)


# Defining a mapping between experiment type names to the constructors of these experiment classes.
_type_to_constructor = {
    ExperimentType.SampleRateExperiment: ExperimentBuilder.create_sample_rate_experiment,
    ExperimentType.FBPExperiment: ExperimentBuilder.create_FBP_experiment,
    ExperimentType.IterationsExperiment: ExperimentBuilder.create_iterations_experiment
}

