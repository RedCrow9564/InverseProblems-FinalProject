#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
main.py - The main module of the project
========================================

This module contains the config for the experiment in the "config" function.
Running this module invokes the :func:`main` function, which then performs the experiment and saves its results
to the configured results folder. Example for running an experiment: ``python main.py``

"""
from Infrastructure.utils import ex, Dict, List, ThreeDMatrix
from Infrastructure.enums import DBType, ExperimentType, SolverName, FBPFilter
from data_generation import fetch_data
from Experiments import ExperimentBuilder


@ex.config
def config():
    """ Config section

    This function contains all possible configuration for all experiments. Full details on each configuration values
    can be found in :mod:`enums.py`.
    """

    _seed: int = 1995  # Random seed.
    experiment_name: str = "TEST"  # A name for the results csv file. It should be unique for every experiment.
    database_name: str = DBType.SheppLogan  # The used database.
    experiment_type: str = ExperimentType.SampleRateExperiment  # The type of experiment for running.
    
    # General config for sample-rate experiments
    sample_rate_experiment_config: Dict = {
        "projections_number": 160,
        "snr_list": [0.0],
        "reconstruction_algorithm": SolverName.TVRegularization, 
        "theta_rates": [1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 80, 160],
        # "theta_rates": [1, 2, 4, 5, 8, 10],
        # "theta_rates": [16, 20, 32, 40, 80],
        "displacement_rates": [1],
        "alpha": 0.01,
        "iterations_number": 1
    }

    # General config for Filtered-Backprojection experiments
    fbp_experiment_config: Dict = {
        "fbp_filters_list": [FBPFilter.Ramp, FBPFilter.Hamming, FBPFilter.SheppLogan],
        "projections_number": 128,  # Number of projections used for Radon-Transform.
        "snr_list": [0.0, 1e-2]  # List of SNR to use. SNR of np.inf or 0 are both interpreted as having no noise at all
    }

    # General config for Iterations experiments
    iterations_experiment_config: Dict = {
        "max_iterations": 5,  # Iterations for each iterative algorithm.
        "snr_list": [0.0],  # List of SNR to use. SNR of np.inf or 0 are both interpreted as having no noise at all.
        "projections_number": 128,  # Number of projections used for Radon-Transform.
        "alphas_list": [1e-2],  # List of regularization terms for each regularization algorithm.
        "compared_algorithms": [SolverName.SART]
    }

    # Paths config (relative paths, not absolute paths)
    results_path: str = r'Results'  # Path for saving csv outputs.
    resources_path: str = r'resources'  # Path from which local databases are loaded.
    covid19_ct_scans_config: Dict[str, str] = {  # DB config for the COVID-19 images DB.
        "db_path": r'COVID-19 CT scans',
        "database_file_name": r'COVID19_CT_scans.csv'
    }
    ct_medical_images_kaggle_config: Dict[str, str] = {  # DB config for the chest CT images DB from Kaggle.com.
        "db_path": r'CT Medical Images Kaggle DB',
        "database_file_name": r'overview.csv'
    }

    # List of scaling factors to use when generating Shepp-Logan images.
    shepp_logan_scaling_factors: List[float] = [1, 0.4]


@ex.automain
def main(database_name: str, experiment_type: str, experiment_name: str, results_path: str) -> None:
    """ The main function of this project

    This functions performs the desired experiment according to the given configuration.
    Then it saves all the results to a csv file in the results folder (given in the configuration).
    """

    # Loading the requested database.
    data: ThreeDMatrix = fetch_data(database_name, 1)

    # Create an experiment object, and then perform the experiment.
    print("Before creating experiment")
    experiment = ExperimentBuilder.create_experiment(experiment_type, data, database_name)

    print("Before running experiment")
    
    experiment_log, output_images = experiment.run()
    print("Experiment done running. {} image reconstructions created from {} images received as data".format(
          len(output_images), len(data)))

    # Plotting the graphics for this specific experiment type and results.
    print("Before plotting experiment")
    experiment.plot('{}\\{}'.format(results_path, experiment_name))
    
    # Saving the experiment output to a CSV file.
    experiment_log.save_log(log_file_name=experiment_name, results_folder_path=results_path)
