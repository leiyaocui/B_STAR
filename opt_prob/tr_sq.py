import cvxpy as cp
import numpy as np
from loguru import logger

from opt_term import Term
from utils.logger import pformat_table

from .problem import ProbManager, ProbResult, ProbStatus


class TR_SP:
    """
    Trust Region Sequential Programming (TR-SP) algorithm.
    """

    def __init__(
        self,
        max_iter: int = 40,
        improve_ratio_threshold: float = 0.2,
        min_approx_improve: float = 1e-4,
        min_approx_improve_frac: float = 1e-3,
        trust_shrink_ratio: float = 0.1,
        trust_expand_ratio: float = 1.5,
        max_merit_coeff_iter: int = 5,
        merit_coeff_increase_ratio: float = 10.0,
    ):
        """
        Initialize the TR-SP algorithm settings.

        Parameters
        ----------
        max_iter : int, optional
            The maximum number of iterations
        improve_ratio_threshold : float, optional
            The minimum ratio of actual to predicted improvement required to accept a step
        min_approx_improve : float, optional
            The minimum improvement required to keep the current trust region size
        min_approx_improve_frac : float, optional
            The minimum improvement ratio required to keep the current trust region size
        trust_shrink_ratio : float, optional
            The ratio to shrink the trust region when the actual improvement is less than the minimum required
        trust_expand_ratio : float, optional
            The ratio to expand the trust region when the actual improvement is greater than the minimum required
        max_merit_coeff_iter : int, optional
            The maximum number of merit function coefficient increases
        merit_coeff_increase_ratio : float, optional
            The ratio to increase the merit function coefficient
        """
        self.improve_ratio_threshold = improve_ratio_threshold
        self.min_approx_improve = min_approx_improve
        self.min_approx_improve_frac = min_approx_improve_frac
        self.max_iter = int(max_iter)
        self.trust_shrink_ratio = trust_shrink_ratio
        self.trust_expand_ratio = trust_expand_ratio
        self.max_merit_coeff_iter = int(max_merit_coeff_iter)
        self.merit_coeff_increase_ratio = merit_coeff_increase_ratio

        self.trust_region_size: np.ndarray = None
        self.min_trust_region_size: np.ndarray = None

        self.variable_bounds: tuple[np.ndarray, np.ndarray] = []
        self.prob_manager = ProbManager()

    def set_variable(
        self,
        variable: np.ndarray,
        lower_bound: np.ndarray = None,
        upper_bound: np.ndarray = None,
    ):
        if lower_bound is not None:
            assert lower_bound.shape == variable.shape
        else:
            lower_bound = np.full(variable.shape, -np.inf)
        if upper_bound is not None:
            assert upper_bound.shape == variable.shape
        else:
            upper_bound = np.full(variable.shape, np.inf)
        self.variable_bounds = (lower_bound, upper_bound)

        self.prob_manager.set_variable(variable)
        self.prob_manager.set_variable_bounds(lower_bound, upper_bound)

    def set_trust_region_size(
        self, trust_region_size: np.ndarray, min_trust_region_size: np.ndarray = None
    ):
        assert trust_region_size.shape == self.variable_bounds[0].shape
        if min_trust_region_size is not None:
            assert min_trust_region_size.shape == trust_region_size.shape
        else:
            min_trust_region_size = np.full(trust_region_size.shape, 1e-4)
        self.trust_region_size = trust_region_size
        self.min_trust_region_size = min_trust_region_size

    def add_term(self, term: Term, merit_coeff: float = 1.0):
        self.prob_manager.add_term(term, merit_coeff)

    def solve(
        self,
        solver: str = "COPT",
        canon_backend: str = "CPP",
        solver_verbose: bool = False,
        print_variable_dof: int = 3,
        ret_all_steps: bool = False,
        **kwargs,
    ) -> ProbResult:
        """
        Solve the optimization problem.

        Parameters
        ----------
        solver : str, optional
            The optimization solver to use
        canon_backend : str, optional
            The canonicalization backend to use
        solver_verbose : bool, optional
            Whether to info the solver information
        print_variable_dof : int, optional
            The number of degrees of freedom to print for the variable
        ret_all_steps : bool, optional
            Whether to return all steps of the variable
        kwargs : dict
            Additional keyword arguments for the solver
        """
        result = self._solve(
            solver, canon_backend, solver_verbose, ret_all_steps, **kwargs
        )
        merit, result.cost_values, result.constraint_violations = (
            self.prob_manager.eval(result.variable)
        )

        logger.opt(lazy=True).success(
            "\n========== Final Results ==========\n"
            "Status: {}\n"
            "Variables:\n{}\n"
            "Merit:\n{}\n"
            "Cost Values:\n{}\n"
            "Constraint Violations:\n{}",
            lambda: result.status.value,
            lambda: result.variable[:, :print_variable_dof],
            lambda: merit,
            lambda: pformat_table(result.cost_values),
            lambda: pformat_table(result.constraint_violations),
        )
        return result

    @logger.catch(reraise=True)
    def _solve(
        self,
        solver: str,
        canon_backend: str,
        solver_verbose: bool,
        ret_all_steps: bool,
        **kwargs,
    ):
        result = ProbResult()
        result.variable = self.prob_manager.variable_value
        merit_prev, result.cost_values, result.constraint_violations = (
            self.prob_manager.eval(result.variable)
        )
        merit_prev_clipped = (
            np.sign(merit_prev) * np.clip(np.abs(merit_prev), 1e-10, None)
            if merit_prev != 0
            else 1e-10
        )
        if ret_all_steps:
            result.variable_all_steps.append(result.variable)

        num_merit_coeff_iter = 0
        num_iter = 0
        partially_converged = False
        trust_region_size = self.trust_region_size
        while num_iter < self.max_iter:
            self.prob_manager.set_variable(result.variable)
            self.prob_manager.construct()

            while np.any(trust_region_size > self.min_trust_region_size):
                partially_converged = False
                self.prob_manager.set_variable(result.variable)

                trust_regions = [
                    np.maximum(
                        self.variable_bounds[0],
                        self.prob_manager.variable_value - trust_region_size,
                    ),
                    np.minimum(
                        self.variable_bounds[1],
                        self.prob_manager.variable_value + trust_region_size,
                    ),
                ]
                self.prob_manager.set_variable_bounds(
                    trust_regions[0], trust_regions[1]
                )

                self.prob_manager.solve(
                    solver=solver,
                    canon_backend=canon_backend,
                    solver_verbose=solver_verbose,
                    **kwargs,
                )
                if self.prob_manager.problem_status != cp.OPTIMAL:
                    logger.error(
                        "Optimization failed:", self.prob_manager.problem_status
                    )
                    result.status = ProbStatus.SolverFailed
                    return result

                variable_curr = self.prob_manager.variable_value
                merit_curr, cost_values_curr, constraint_violations_curr = (
                    self.prob_manager.eval(variable_curr)
                )
                (
                    approx_merit_curr,
                    approx_cost_values_curr,
                    approx_constraint_violations_curr,
                ) = self.prob_manager.eval(variable_curr, eval_approx=True)

                actual_improve = merit_prev - merit_curr
                approx_improve = merit_prev - approx_merit_curr
                approx_improve_clipped = (
                    np.sign(approx_improve)
                    * np.clip(np.abs(approx_improve), 1e-10, None)
                    if approx_improve != 0
                    else 1e-10
                )
                merit_improve_ratio = actual_improve / approx_improve_clipped

                logger.opt(lazy=True).info(
                    "\n========== Iteration ==========\n"
                    "Number of Merit Coefficient Increase: {}\n"
                    "Number of Iteration: {}\n"
                    # "========== Previous Actual ==========\n"
                    # "Cost Values:\n{}\n"
                    # "Constraint Violations:\n{}\n"
                    # "========== Current Actual ==========\n"
                    # "Cost Values:\n{}\n"
                    # "Constraint Violations:\n{}\n"
                    # "========== Curret Approx ==========\n"
                    # "Cost Values:\n{}\n"
                    # "Constraint Violations:\n{}\n"
                    # "========== Trust Region ==========\n"
                    # "{}\n"
                    "========== Merit ==========\n"
                    "Previous Actual: {}\n"
                    "Current Actual: {}\n"
                    "Current Approx: {}\n"
                    "========== Improvement ==========\n"
                    "Actual Improvement: {}\n"
                    "Approximate Improvement: {}\n"
                    "Approximate Improvement Ratio: {}\n"
                    "Improvement Ratio: {}",
                    lambda: num_merit_coeff_iter,
                    lambda: num_iter,
                    # lambda: pformat_table(result.cost_values),
                    # lambda: pformat_table(result.constraint_violations),
                    # lambda: pformat_table(cost_values_curr),
                    # lambda: pformat_table(constraint_violations_curr),
                    # lambda: pformat_table(approx_cost_values_curr),
                    # lambda: pformat_table(approx_constraint_violations_curr),
                    # lambda: trust_region_size[0],
                    lambda: merit_prev,
                    lambda: merit_curr,
                    lambda: approx_merit_curr,
                    lambda: actual_improve,
                    lambda: approx_improve,
                    lambda: approx_improve / merit_prev_clipped,
                    lambda: merit_improve_ratio,
                )

                if approx_improve < -1e-10:
                    logger.warning("Convexification is probably wrong")
                    logger.debug("Shrink trust region")
                    trust_region_size = np.maximum(
                        trust_region_size * self.trust_shrink_ratio,
                        self.min_trust_region_size,
                    )
                    continue
                if (approx_improve < self.min_approx_improve) or (
                    approx_improve / merit_prev_clipped < self.min_approx_improve_frac
                ):
                    logger.info(
                        "Convergence reached partially, because improvement (ratio) is too small"
                    )
                    partially_converged = True
                    break

                if merit_improve_ratio < self.improve_ratio_threshold:
                    logger.debug("Shrink trust region")
                    trust_region_size = np.maximum(
                        trust_region_size * self.trust_shrink_ratio,
                        self.min_trust_region_size,
                    )
                    continue
                else:
                    logger.opt(lazy=True).debug(
                        "Accept this step, update variable length: {}",
                        lambda: np.linalg.norm(variable_curr - result.variable),
                    )
                    merit_prev = merit_curr
                    merit_prev_clipped = (
                        np.sign(merit_prev) * np.clip(np.abs(merit_prev), 1e-10, None)
                        if merit_prev != 0
                        else 1e-10
                    )
                    result.variable = variable_curr
                    result.cost_values = cost_values_curr
                    result.constraint_violations = constraint_violations_curr
                    if ret_all_steps:
                        result.variable_all_steps.append(result.variable)

                    if self.trust_expand_ratio > 1.0:
                        logger.debug("Expand trust region")
                        trust_region_size *= self.trust_expand_ratio
                    else:
                        logger.debug("Reset trust region")
                        trust_region_size = self.trust_region_size

                    break
            else:
                logger.info(
                    "Convergence reached partially, because trust region is too small"
                )
                partially_converged = True

            if (len(result.constraint_violations) == 0) or (
                np.all([it["is_satisfied"] for it in result.constraint_violations])
            ):
                if (len(result.cost_values) == 0) or np.all(
                    [it["is_satisfied"] for it in result.cost_values]
                ):
                    logger.info("Convergence reached, and constraints satisfied")
                    result.status = ProbStatus.Converged
                    return result
                else:
                    logger.info("Constraints satisfied, but unconverged")
                    result.status = ProbStatus.Unconverged

                    if partially_converged or num_iter == self.max_iter - 1:
                        logger.error(
                            "Stop the optimization, because there is no improvement"
                        )
                        return result
                    else:
                        logger.debug("Move to the next iteration")
                        num_iter += 1
                        continue
            else:
                logger.info("Constraints not satisfied")
                result.status = ProbStatus.Infeasible

                if partially_converged or num_iter == self.max_iter - 1:
                    if num_merit_coeff_iter < self.max_merit_coeff_iter:
                        logger.warning(
                            "Increase merit function coefficient, and restore trust region"
                        )

                        for it in result.constraint_violations:
                            if not it["is_satisfied"]:
                                self.prob_manager.scale_merit_coeff(
                                    it["idx"], self.merit_coeff_increase_ratio
                                )
                        trust_region_size = self.trust_region_size
                        merit_prev, _, _ = self.prob_manager.eval(result.variable)
                        merit_prev_clipped = (
                            np.sign(merit_prev)
                            * np.clip(np.abs(merit_prev), 1e-10, None)
                            if merit_prev != 0
                            else 1e-10
                        )

                        num_iter = 0
                        num_merit_coeff_iter += 1
                        continue
                    else:
                        logger.error(
                            "Max merit coefficient iterations reached, but not satisfied constraints"
                        )
                        return result
                else:
                    logger.debug("Move to the next iteration")
                    num_iter += 1
                    continue
