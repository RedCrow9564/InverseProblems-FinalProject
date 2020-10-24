# -*- coding: utf-8 -*-
"""
enums.py - All enums section
============================

This module contains all possible enums of this project. Most of them are used by the configuration section in
:mod:`main`. An example for using enum:
::
    LogFields.DataType

"""

from Infrastructure.utils import BaseEnum


class LogFields(BaseEnum):
    """
    The enum class of fields within experiments logs. Possible values:

    * ``LogFields.DataType``
    
    * ``LogFields.ImageIndex``
    
    * ``LogFields.SolverName``
    
    * ``LogFields.Iterations``

    * ``LogFields.FilterName``
    
    * ``LogFields.ProjectionsNumber``

    * ``LogFields.SNR``

    * ``LogFields.RMSError``

    * ``LogFields.SinogramError``
    
    * ``LogFields.ThetaRate``
    
    * ``LogFields.DisplacementRate``
        
    * ``LogFields.RegularizarionParameter``
    """
    DataType: str = "Data type"
    ImageIndex: str = "Image index"
    SolverName: str = "Solver name"
    Iterations: str = "Iterations"
    FilterName: str = "Filter Name"
    ProjectionsNumber: str = "Projections Number"
    SNR: str = "SNR"
    RMSError: str = "RMS Error"
    SinogramError: str = "Sinogram Error"
    ThetaRate: str = "Theta Rate"
    DisplacementRate: str = "Displacement Rate"
    RegularizarionParameter: str = "Regularization Parameter"


class ExperimentType(BaseEnum):
    """
    The enum class of experiment types. Possible values:

    * ``ExperimentType.SampleRateExperiment``
    
    * ``ExperimentType.FBPExperiment``
    
    * ``ExperimentType.IterationsExperiment``
    """
    SampleRateExperiment: str = "Sample-Rate Experiment"
    FBPExperiment: str = "Filtered-Backprojection Experiment"
    IterationsExperiment: str = "Iterations Experiment"


class DBType(BaseEnum):
    """
    The enum class of database types. Possible values:

    * ``DBType.SheppLogan``
    
    * ``DBType.COVID19_CT_Scans``
    
    * ``DBType.CT_Medical_Images``
    """
    SheppLogan: str = "Shepp-Logan Phantom"
    COVID19_CT_Scans: str = "COVID19 CT Scans"
    CT_Medical_Images: str = "CT Medical Images Kaggle DB"


class SolverName(BaseEnum):
    """
    The enum class of algoritm types. Possible values:

    * ``SolverName.FBP``
    
    * ``SolverName.SART``

    * ``SolverName.TruncatedSVD``
    
    * ``SolverName.L1Regularization``
    
    * ``SolverName.L2Regularization``

    * ``SolverName.TVRegularization``
    """
    FBP: str = "Filtered-Backprojection"
    SART: str = "SART"
    TruncatedSVD: str = "Truncated SVD"
    L1Regularization: str = "L1 Regularization"
    L2Regularization: str = "L2 Regularization"
    TVRegularization: str = "TV Regularization"


class FBPFilter(BaseEnum):
    """
    The enum class of algoritm types. Possible values:

    * ``FBPFilter.Ramp``
    
    * ``FBPFilter.Hamming``

    * ``FBPFilter.SheppLogan``
    """
    Ramp: str = "ramp"
    Hamming: str = "hamming"
    SheppLogan: str = "shepp-logan"
