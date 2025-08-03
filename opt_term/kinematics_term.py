import cvxpy as cp
import numpy as np
import pinocchio as pin

from .term import Term, TermType


class TrajSmoothTerm(Term):
    def __init__(
        self,
        k: int = 1,
        offset: int = 0,
        name: str = "joint space trajectory smoothness",
        threshold: float = 0.2,
    ):
        super().__init__(TermType.COST_ABS, name, threshold)
        self.k = k
        self.offset = offset

    def _approx(self, traj_q: np.ndarray):
        return

    def _construct_approx_expr(self, traj_q: cp.Expression):
        expr = cp.diff(traj_q[:, self.offset :], k=self.k, axis=0)
        return expr.flatten("C")

    def _eval(self, traj_q: np.ndarray):
        value = np.diff(traj_q[:, self.offset :], n=self.k, axis=0)
        return value.flatten()


class BaseFixationTerm(Term):
    def __init__(
        self,
        rad2meter: float = 0.1,
        name: str = "joint space base fixation",
        threshold: float = 1e-4,
    ):
        super().__init__(TermType.CONSTRAINT_EQ, name, threshold)
        self.rad2meter = rad2meter

    def _approx(self, traj_q: np.ndarray):
        return

    def _construct_approx_expr(self, traj_q: cp.Expression):
        expr = traj_q[:, :3] - cp.mean(traj_q[:, :3], axis=0, keepdims=True)
        expr = cp.hstack([expr[:, :2], expr[:, [2]] * self.rad2meter])
        return expr.flatten("C")

    def _eval(self, traj_q: np.ndarray):
        value = traj_q[:, :3] - np.mean(traj_q[:, :3], axis=0, keepdims=True)
        value[:, 2] *= self.rad2meter
        return value.flatten()


class EEPoseTerm(Term):
    def __init__(
        self,
        model: pin.Model,
        ee_frame_id: int,
        timestep_list: list[int],
        target_pose_list: list[np.ndarray],
        delta_q: float = 1e-6,
        rad2meter: float = 0.1,
        name: str = "operational space end effector pose",
        threshold: float = 1e-4,
    ):
        super().__init__(TermType.CONSTRAINT_EQ, name, threshold)
        self.model = model
        self.data = self.model.createData()
        self.ee_frame_id = ee_frame_id
        self.timestep_list = timestep_list
        self.delta_q = delta_q
        self.rad2meter = rad2meter
        self.target_pose_inv_list = [
            pin.SE3(target_pose).inverse() for target_pose in target_pose_list
        ]

    def _calc_pose_err(
        self, q: np.ndarray, target_pose_inv: pin.SE3, with_jacobian: bool = False
    ):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose_err = target_pose_inv.act(self.data.oMf[self.ee_frame_id])
        pose_err = pin.log6(pose_err).np

        if with_jacobian:
            J = np.zeros((6, self.model.nq))
            for i in range(self.model.nq):
                q_perturbed = q.copy()
                q_perturbed[i] += self.delta_q

                pin.forwardKinematics(self.model, self.data, q_perturbed)
                pin.updateFramePlacements(self.model, self.data)
                pose_err_perturbed = target_pose_inv.act(
                    self.data.oMf[self.ee_frame_id]
                )

                J[:, i] = (pin.log6(pose_err_perturbed).np - pose_err) / self.delta_q
            return pose_err, J
        return pose_err, None

    def _calc_pose_err_list(self, traj_q: np.ndarray, with_jacobian: bool = False):
        pose_err_list = []
        J_list = []
        for q, target_pose_inv in zip(traj_q, self.target_pose_inv_list):
            ret = self._calc_pose_err(q, target_pose_inv, with_jacobian)
            pose_err_list.append(ret[0])
            if with_jacobian:
                J_list.append(ret[1])
        if with_jacobian:
            return pose_err_list, J_list
        else:
            return pose_err_list

    def _approx(self, traj_q: np.ndarray):
        q0_list = traj_q[self.timestep_list]
        pose_err0_list, J_list = self._calc_pose_err_list(q0_list, with_jacobian=True)
        self.q0_list = q0_list
        self.pose_err0_list = pose_err0_list
        self.J_list = J_list

    def _construct_approx_expr(self, traj_q: cp.Expression):
        q_list = traj_q[self.timestep_list]
        expr_list = []
        for q, q0, pose_err0, J in zip(
            q_list, self.q0_list, self.pose_err0_list, self.J_list
        ):
            expr = J @ (q - q0) + pose_err0
            expr_list.append(expr)
        expr = cp.vstack(expr_list)
        expr = cp.hstack([expr[:, :3], expr[:, 3:] * self.rad2meter])
        return expr.flatten("C")

    def _eval(self, traj_q: np.ndarray):
        q_list = traj_q[self.timestep_list]
        pose_err_list = self._calc_pose_err_list(q_list, with_jacobian=False)
        value = np.array(pose_err_list)
        value[:, 3:] *= self.rad2meter
        return value.flatten()
