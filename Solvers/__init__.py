from Solvers.fbp_and_sart import filtered_back_projection, sart
from Solvers.regularization_methods import l1_regularization, \
    l2_regularization, total_variation_regularization

from Infrastructure.utils import Callable, Dict
from Infrastructure.enums import SolverName



_name_to_solver: Dict = {
        SolverName.FBP: filtered_back_projection,
        SolverName.SART: sart,
        SolverName.L1Regularization: l1_regularization,
        SolverName.TVRegularization: total_variation_regularization,
        SolverName.L2Regularization: l2_regularization
}


def get_solver(solver_name: str) -> Callable:    
    return _name_to_solver[solver_name]
