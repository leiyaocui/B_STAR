from abc import ABC, abstractmethod
from enum import Enum

import cvxpy as cp
import numpy as np


class TermType(Enum):
    COST_SQUARE = "squared cost"
    COST_ABS = "absolute cost"
    CONSTRAINT_EQ = "equality constraint"
    CONSTRAINT_INEQ = "inequality constraint"


class Term(ABC):
    def __init__(self, type: TermType, name: str, threshold: float):
        self.type = type
        self.name = name
        self.threshold = threshold
        self.expr: cp.Expression = None
        self.slack_pos: cp.Expression = None
        self.slack_neg: cp.Expression = None

    def __str__(self):
        return f"[{self.type.value.upper()}] {self.name}"

    @abstractmethod
    def _approx(self, variable: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def _construct_approx_expr(self, variable: cp.Expression) -> cp.Expression:
        raise NotImplementedError

    def _construct(
        self, expr: cp.Expression
    ) -> tuple[cp.Expression, list[cp.Expression]]:
        assert expr.ndim <= 1

        cost = 0.0
        constraints = []
        if self.type == TermType.COST_SQUARE:
            expr = cp.square(expr)
            cost = cp.sum(expr) / expr.shape[0]
        elif self.type == TermType.COST_ABS:
            self.slack_pos = cp.Variable(expr.shape, nonneg=True)
            self.slack_pos.value = np.abs(expr.value)

            cost = cp.sum(self.slack_pos) / self.slack_pos.shape[0]
            constraints.append(expr <= self.slack_pos)
            constraints.append(-expr <= self.slack_pos)
        elif self.type == TermType.CONSTRAINT_EQ:
            self.slack_pos, self.slack_neg = (
                cp.Variable(expr.shape, nonneg=True),
                cp.Variable(expr.shape, nonneg=True),
            )
            self.slack_pos.value = np.maximum(expr.value, 0)
            self.slack_neg.value = np.maximum(-expr.value, 0)

            cost = (
                cp.sum(self.slack_pos) / self.slack_pos.shape[0]
                + cp.sum(self.slack_neg) / self.slack_neg.shape[0]
            )
            constraints.append(self.slack_pos - self.slack_neg == expr)
        elif self.type == TermType.CONSTRAINT_INEQ:
            self.slack_pos = cp.Variable(expr.shape, nonneg=True)
            self.slack_pos.value = np.maximum(expr.value, 0)

            cost = cp.sum(self.slack_pos) / self.slack_pos.shape[0]
            constraints.append(expr <= self.slack_pos)
        else:
            raise ValueError(f"Invalid term type: {self.type}")
        return cost, constraints

    def apply(
        self, variable: cp.Expression
    ) -> tuple[cp.Expression, list[cp.Expression]]:
        self._approx(variable.value)
        self.expr = self._construct_approx_expr(variable)
        cost, constraints = self._construct(self.expr)
        return cost, constraints

    @abstractmethod
    def _eval(self, variable: np.ndarray) -> float | np.ndarray:
        raise NotImplementedError

    def eval(self, variable: np.ndarray = None, eval_approx: bool = False) -> float:
        if eval_approx:
            value = self.expr.value
        else:
            value = self._eval(variable)

        if self.type == TermType.COST_SQUARE:
            value = np.power(value, 2)
        elif self.type == TermType.COST_ABS:
            value = np.abs(value)
        elif self.type == TermType.CONSTRAINT_EQ:
            value = np.abs(value)
        elif self.type == TermType.CONSTRAINT_INEQ:
            value = np.maximum(0, value)

        is_satisfied = np.all(value <= self.threshold)
        value = value.sum() / value.shape[0]
        return value, is_satisfied
