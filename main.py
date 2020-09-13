
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
from Infrastructure.enums import LogFields, DBType, ExperimentType
from data_generation import fetch_data
from Experiments import ExperimentBuilder


@ex.config
def config():
    """ Config section

    This function contains all possible configuration for all experiments. Full details on each configuration values
    can be found in :mod:`enums.py`.
    """


    experiment_name: str = "Test Code"
    database_name: str = DBType.SheppLogan
    experiment_type: str = ExperimentType.FBPExperiment

    # Artificial noise config
    noise_config: Dict = {
        "noise_type": "Gaussian",  # Maybe this should be another enum?
        "signal_noise_ratio": 0.0
    }
    
    # General config for experiments
    sample_rate_experiment_config: Dict = {
        "ratios_list": [0.5, 1, 1.5, 2, 2.5],
        "fbp_filters_list": ["ramp", "hamming"]
    }
    fbp_experiment_config: Dict = {
        "fbp_filters_list": ["ramp", "hamming"],
        "projections_number": 160
    }
    iterations_experiment_config: Dict = {
        "iterations_list": [1, 2, 3]
    }

    # Paths config (relative paths, not absolute paths)
    results_path: str = r'Results'
    resources_path: str = r'resources'
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

    data: ThreeDMatrix = fetch_data(database_name, 3)
    # Create an experiment object, and then perform the experiment.
    experiment = ExperimentBuilder.create_experiment(experiment_type, data, database_name)
    print(experiment_type)
    
    experiment_log: DataLog = experiment.run()
    print(experiment_log._data[LogFields.RMSError])
    experiment_log.save_log(log_file_name=experiment_name, results_folder_path=results_path)
