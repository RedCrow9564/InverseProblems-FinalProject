#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
main.py - The main module of the project
========================================

This module contains the config for the experiment in the "config" function.
Running this module invokes the :func:`main` function, which then performs the experiment and saves its results
to the configured results folder. Example for running an experiment: ``python main.py``

"""
from Infrastructure.utils import ex, DataLog, Dict, Union, List, Scalar, Vector, Matrix, ThreeDMatrix
from Infrastructure.enums import LogFields, DBType, ExperimentType, SolverName
from data_generation import fetch_data
from Experiments import ExperimentBuilder
from matplotlib import pyplot as plt
import numpy as np


@ex.config
def config():
    """ Config section

    This function contains all possible configuration for all experiments. Full details on each configuration values
    can be found in :mod:`enums.py`.
    """


    experiment_name: str = "Test after merge"
    database_name: str = DBType.SheppLogan
    experiment_type: str = ExperimentType.SampleRateExperiment

    # Artificial noise config
    noise_config: Dict = {
        "noise_type": "Gaussian",  # Maybe this should be another enum?
        "signal_noise_ratio": 0.0
    }
    
    # General config for experiments
    sample_rate_experiment_config: Dict = {
        "projections_number": 160,
        "snr_list": [1000, 0.0001],
        "reconstruction_algorithm": SolverName.L1Regularization, 
        "theta_rates": [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 80, 160],
        # "theta_rates": [1, 2, 4, 5, 8, 10],
        # "theta_rates": [16, 20, 32, 40, 80],
        "displacement_rates": [1]
    }
    fbp_experiment_config: Dict = {
        "fbp_filters_list": ["ramp"],
        "projections_number": 600,
        "snr_list": [0.0]
    }
    iterations_experiment_config: Dict = {
        "max_iterations": 5,
        "snr_list": [0.01],
        "projections_number": 160,
        "alpha": 10,
        "compared_algorithms": [SolverName.SART,
                                SolverName.L1Regularization,
                                SolverName.TVRegularization,
                                SolverName.L2Regularization]
    }

    # Paths config (relative paths, not absolute paths)
    results_path: str = r'Results'
    resources_path: str = r'resources'
    shepp_logan_scaling_factors: List[float] = [1, 0.2]
    covid19_ct_scans_config: Dict[str, str] = {
        "db_path": r'COVID-19 CT scans',
        "database_file_name": r'COVID19_CT_scans.csv'
    }
    ct_medical_images_kaggle_config: Dict[str, str] = {
        "db_path": r'CT Medical Images Kaggle DB',
        "database_file_name": r'overview.csv'
    }


@ex.automain
def main(database_name: str, experiment_type: str, experiment_name: str, results_path: str) -> None:
    """ The main function of this project

    This functions performs the desired experiment according to the given configuration.
    Then it saves all the results to a csv file in the results folder (given in the configuration).
    """

    data: ThreeDMatrix = fetch_data(database_name, 2)
    # Create an experiment object, and then perform the experiment.
    print("Before creating experiment")
    experiment = ExperimentBuilder.create_experiment(experiment_type, data, database_name)

    print("Before running experiment")
    
    experiment_log, output_images  = experiment.run()
    print("Experiment done running. {} image reconstructions created from {} images recieved as data".format(
          len(output_images), len(data)))

    print("Before plotting experiment")
    experiment.plot('{}\\{}'.format(results_path, experiment_name))
    
    experiment_log.save_log(log_file_name=experiment_name, results_folder_path=results_path)
