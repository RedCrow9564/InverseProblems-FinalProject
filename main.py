#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
main.py - The main module of the project
========================================

This module contains the config for the experiment in the "config" function.
Running this module invokes the :func:`main` function, which then performs the experiment and saves its results
to the configured results folder. Example for running an experiment: ``python main.py``

"""
import numpy as np
from itertools import product
from Infrastructure.utils import ex, DataLog, Dict, Union, List, Scalar, Vector, Matrix, ThreeDMatrix
from Infrastructure.enums import LogFields, DBType, ExperimentType
from data_generation import fetch_data
from Experiments import ExperimentBuilder

import matplotlib.pyplot as plt


# TODO: Fix
def error_in_circle_pixels(img1: Matrix, img2: Matrix) -> float:
    indices_mask: Matrix = np.zeros_like(img1)
    middle_index: int = int(img1.shape[0] / 2)
    for i, j in product(range(img1.shape[0]), repeat=2):
        if (middle_index - i)  ** 2 + (middle_index - j) ** 2 < middle_index ** 2:
            indices_mask[i, j] = 1
    
    error: float = np.sqrt(np.mean((img1[indices_mask == 1] - img2[indices_mask == 1]) ** 2))
    return error


@ex.config
def config():
    """ Config section

    This function contains all possible configuration for all experiments. Full details on each configuration values
    can be found in :mod:`enums.py`.
    """


    experiment_name: str = "Test Code"
    database_name: str = DBType.SheppLogan
    experiment_type: str = ExperimentType.IterationsExperiment
    results_path: str = r'C:\Users\Elad Eatah\Google Drive\MScStudies\Courses\Topics in Inverse Problems\FinalProject\Results'
    resources_path: str = r'C:\Users\Elad Eatah\Google Drive\MScStudies\Courses\Topics in Inverse Problems\FinalProject\resources'
    covid19_ct_scans_config: Dict[str, str] = {
        "db_path": r'COVID-19 CT scans',
        "database_file_name": r'COVID19_CT_scans.csv'
    }
    ct_medical_images_kaggle_config: Dict[str, str] = {
        "db_path": r'CT Medical Images Kaggle DB',
        "database_file_name": r'overview.csv'
    }


@ex.automain
def main(database_name: str, experiment_type: str) -> None:
    """ The main function of this project

    This functions performs the desired experiment according to the given configuration.
    The function runs the random_svd and random_id for every combination of data_size, approximation rank and increment
    given in the config and saves all the results to a csv file in the results folder (given in the configuration).
    """

    data: ThreeDMatrix = fetch_data(database_name, 3)
    for image in data:
        print(image.shape)
        # TODO: Perform Radon Transform no each image. Then, create an experiment object and perform it.

    experiment = ExperimentBuilder.create_experiment(experiment_type)
    print(experiment_type)
