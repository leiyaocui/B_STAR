from dataclasses import dataclass, field
from enum import Enum

import cvxpy as cp
import numpy as np

from opt_term import Term, TermType


class ProbStatus(Enum):
    Unconverged = "Unconverged"
    Converged = "Converged"
    Infeasible = "Infeasible"
    SolverFailed = "Solver failed"


@dataclass
class ProbResult:
    status: ProbStatus = field(default_factory=lambda: ProbStatus.Unconverged)
    variable: np.ndarray = field(default_factory=lambda: None)
    cost_values: list[dict] = field(default_factory=list)
    constraint_violations: list[dict] = field(default_factory=list)
    variable_all_steps: list[np.ndarray] = field(default_factory=list)


class ProbManager:
    def __init__(self):
        self.variable: cp.Variable = None
        self.variable_bounds: tuple[cp.Parameter, cp.Parameter] = None
        self.terms: list[Term] = []
        self.merit_coeffs: list[cp.Parameter] = []
        self.problem: cp.Problem = None
        self.problem_status: str = None

    def __len__(self):
        return len(self.terms)

    @property
    def variable_value(self):
        return self.variable.value

    def set_variable(self, variable: np.ndarray):
        if self.variable is None:
            self.variable = cp.Variable(variable.shape)
        self.variable.value = variable

    def set_variable_bounds(self, lower_bound: np.ndarray, upper_bound: np.ndarray):
        assert np.all(lower_bound <= upper_bound)
        if self.variable_bounds is None:
            assert lower_bound.shape == self.variable.shape
            assert upper_bound.shape == self.variable.shape
            self.variable_bounds = (
                cp.Parameter(lower_bound.shape),
                cp.Parameter(lower_bound.shape),
            )
        self.variable_bounds[0].value = lower_bound
        self.variable_bounds[1].value = upper_bound

    def add_term(self, term: Term, merit_coeff: float = 1.0):
        self.terms.append(term)
        self.merit_coeffs.append(cp.Parameter(nonneg=True, value=merit_coeff))

    def scale_merit_coeff(self, idx_term: int, scale: float):
        self.merit_coeffs[idx_term].value = self.merit_coeffs[idx_term].value * scale

    def construct(self) -> tuple[cp.Expression, list[cp.Expression]]:
        total_cost = 0.0
        total_constraints = [
            self.variable_bounds[0] <= self.variable,
            self.variable <= self.variable_bounds[1],
        ]
        for i, term in enumerate(self.terms):
            cost, constraint = term.apply(self.variable)
            total_cost += cost * self.merit_coeffs[i]
            total_constraints.extend(constraint)

        self.problem = cp.Problem(cp.Minimize(total_cost), total_constraints)

    def solve(
        self, solver: str, canon_backend: str, solver_verbose: bool = False, **kwargs
    ):
        try:
            self.problem.solve(
                solver=solver,
                canon_backend=canon_backend,
                solver_verbose=solver_verbose,
                **kwargs,
            )
            self.problem_status = self.problem.status
            if (
                not np.all(self.variable_bounds[0].value - 1e-4 <= self.variable_value)
            ) or (
                not np.all(self.variable_value <= self.variable_bounds[1].value + 1e-4)
            ):
                self.problem_status = cp.SOLVER_ERROR
        except cp.SolverError:
            self.problem_status = cp.SOLVER_ERROR

    def eval(
        self, variable: np.ndarray, eval_approx: bool = False
    ) -> tuple[float, list[dict], list[dict]]:
        if eval_approx:
            variable_prev = self.variable.value
            self.variable.value = variable

        merit = 0.0
        cost_values: list[dict] = []
        constraint_violations: list[dict] = []
        for i, term in enumerate(self.terms):
            if term.type in [TermType.COST_SQUARE, TermType.COST_ABS]:
                value, is_satisfied = term.eval(variable, eval_approx)
                cost_values.append(
                    {
                        "idx": i,
                        "name": term.name,
                        "value": value,
                        "type": term.type.name,
                        "merit_coeff": self.merit_coeffs[i].value,
                        "is_satisfied": is_satisfied,
                    }
                )
            elif term.type in [TermType.CONSTRAINT_EQ, TermType.CONSTRAINT_INEQ]:
                value, is_satisfied = term.eval(variable, eval_approx)
                constraint_violations.append(
                    {
                        "idx": i,
                        "name": term.name,
                        "value": value,
                        "type": term.type.name,
                        "merit_coeff": self.merit_coeffs[i].value,
                        "is_satisfied": is_satisfied,
                    }
                )
            merit += value * self.merit_coeffs[i].value

        if eval_approx:
            self.variable.value = variable_prev

        return merit, cost_values, constraint_violations
