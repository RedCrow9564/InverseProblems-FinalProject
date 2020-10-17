
# -*- coding: utf-8 -*-
"""
Solvers\\__init__.py - Factory of Inverse Radon-Transform methods.
=================================================================

This module creates a solver for the Inverse Radon-Transform problem.
The solver is chosen according to a given method name. 
For example:

    get_solver(SolverName.FBP) - Creates the Filtered-Backprojection solver.

"""
from Solvers.fbp_and_sart import filtered_back_projection, sart
from Solvers.regularization_methods import l1_regularization, \
    l2_regularization, total_variation_regularization, TSVD

from Infrastructure.utils import Callable, Dict
from Infrastructure.enums import SolverName


# Defining a mapping between solvers names to the functions of these solvers.
_name_to_solver: Dict = {
        SolverName.FBP: filtered_back_projection,
        SolverName.SART: sart,
        SolverName.TruncatedSVD: TSVD,
        SolverName.L1Regularization: l1_regularization,
        SolverName.TVRegularization: total_variation_regularization,
        SolverName.L2Regularization: l2_regularization
}


def get_solver(solver_name: str) -> Callable:   
    """
    This function generates the specific solver, whose name is given as input.

    Args:
        solver_name(str): A name of a specific solver, to be created.

    Returns:
        A function of the requested solver.
    """ 
    return _name_to_solver[solver_name]
