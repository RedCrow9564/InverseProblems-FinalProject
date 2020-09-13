# -*- coding: utf-8 -*-
"""
enums.py - All enums section
============================

This module contains all possible enums of this project. Most of them are used by the configuration section in
:mod:`main`. An example for using enum:
::
    DistributionType.Uniform

"""

from Infrastructure.utils import BaseEnum


class LogFields(BaseEnum):
    """
    The enum class of fields within experiments logs. Possible values:

    * ``LogFields.DataSize``

    * ``LogFields.DataType``

    * ``LogFields.RMSError``
    """
    DataType: str = "Data type"
    FilterName: str = "Filter Name"
    ProjectionsNumber: str = "Projections Number"
    RMSError: str = "RMS Error"


class ExperimentType(BaseEnum):
    SampleRateExperiment: str = "Sample-Rate Experiment"
    FBPExperiment: str = "Filtered-Backprojection Experiment"
    IterationsExperiment: str = "Iterations Experiment"


class DBType(BaseEnum):
    Random: str = "Random"
    SheppLogan: str = "Shepp-Logan Phantom"
    COVID19_CT_Scans: str = "COVID19 CT Scans"
    CT_Medical_Images: str = "CT Medical Images Kaggle DB"
