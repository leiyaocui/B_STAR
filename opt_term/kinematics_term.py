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
        self.target_pose_list = [
            pin.SE3(target_pose) for target_pose in target_pose_list
        ]

    def _calc_twist_err(
        self, q: np.ndarray, target_pose: pin.SE3, with_jacobian: bool = False
    ):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose_err = target_pose.actInv(self.data.oMf[self.ee_frame_id])
        twist_err = pin.log6(pose_err).np

        if with_jacobian:
            J = pin.Jlog6(pose_err)
            if not np.isfinite(J).all():  # Instable case of zero rotation error
                J = np.eye(6)
                J[:3, 3:] = 0.5 * pin.skew(twist_err[:3])
            J = J @ pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_frame_id, pin.LOCAL
            )

            return twist_err, J
        return twist_err, None

    def _calc_twist_err_list(self, traj_q: np.ndarray, with_jacobian: bool = False):
        twist_err_list = []
        J_list = []
        for q, target_pose in zip(traj_q, self.target_pose_list):
            ret = self._calc_twist_err(q, target_pose, with_jacobian)
            twist_err_list.append(ret[0])
            if with_jacobian:
                J_list.append(ret[1])
        if with_jacobian:
            return twist_err_list, J_list
        else:
            return twist_err_list

    def _approx(self, traj_q: np.ndarray):
        q0_list = traj_q[self.timestep_list]
        twist_err0_list, J_list = self._calc_twist_err_list(q0_list, with_jacobian=True)
        self.q0_list = q0_list
        self.twist_err0_list = twist_err0_list
        self.J_list = J_list

    def _construct_approx_expr(self, traj_q: cp.Expression):
        q_list = traj_q[self.timestep_list]
        expr_list = []
        for q, q0, twist_err0, J in zip(
            q_list, self.q0_list, self.twist_err0_list, self.J_list
        ):
            expr = J @ (q - q0) + twist_err0
            expr_list.append(expr)
        expr = cp.vstack(expr_list)
        expr = cp.hstack([expr[:, :3], expr[:, 3:] * self.rad2meter])
        return expr.flatten("C")

    def _eval(self, traj_q: np.ndarray):
        q_list = traj_q[self.timestep_list]
        twist_err_list = self._calc_twist_err_list(q_list, with_jacobian=False)
        value = np.array(twist_err_list)
        value[:, 3:] *= self.rad2meter
        return value.flatten()


class ManipulabilityTerm(Term):
    def __init__(
        self,
        model: pin.Model,
        ee_frame_id: int,
        min_manipulability: float = 0.05,
        name: str = "manipulability",
        threshold: float = 1e-4,
    ):
        super().__init__(TermType.CONSTRAINT_INEQ, name, threshold)
        self.model = model
        self.data = self.model.createData()
        self.ee_frame_id = ee_frame_id
        self.min_manipulability = min_manipulability

    def _calc_kinematic_hessian(self, J: np.ndarray) -> np.ndarray:
        n = J.shape[1]
        v = J[:3, :]  # (3, n)
        omega = J[3:, :]  # (3, n)

        omega_j = omega.T[:, None, :]  # (n, 1, 3)
        v_i = v.T[None, :, :]  # (1, n, 3)
        omega_i = omega.T[None, :, :]  # (1, n, 3)

        H_linear = np.cross(omega_j, v_i)  # (n, n, 3)
        H_angular = np.cross(omega_j, omega_i)  # (n, n, 3)

        # Zero out lower triangle (j > i) - we only compute for j <= i
        mask_upper = np.triu(np.ones((n, n), dtype=bool))
        H_linear = H_linear * mask_upper[:, :, None]
        H_angular = H_angular * mask_upper[:, :, None]

        # Apply symmetry for linear part: H_linear[i, j, :] = H_linear[j, i, :]
        i_lower, j_lower = np.tril_indices(n, k=-1)
        H_linear[i_lower, j_lower, :] = H_linear[j_lower, i_lower, :]

        H = np.zeros((n, 6, n))
        H[:, :3, :] = H_linear.transpose(0, 2, 1)
        H[:, 3:, :] = H_angular.transpose(0, 2, 1)

        return H

    def _calc_manip(self, q: np.ndarray, with_jacobian: bool = False):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        J = pin.computeFrameJacobian(
            self.model, self.data, q, self.ee_frame_id, pin.LOCAL
        )

        JJT = J @ J.T
        manip = np.sqrt(np.linalg.det(JJT))

        if not with_jacobian:
            return manip, None

        H = self._calc_kinematic_hessian(J)
        JJTinv = np.linalg.pinv(JJT)

        n = J.shape[1]
        Jm = np.zeros((1, n))
        for i in range(n):
            JH = J @ H[i].T
            Jm[0, i] = manip * JH.flatten("F").T @ JJTinv.flatten("F")

        return manip, Jm

    def _calc_manip_list(self, traj_q: np.ndarray, with_jacobian: bool = False):
        manip_list = []
        Jm_list = []
        for q in traj_q:
            ret = self._calc_manip(q, with_jacobian)
            manip_list.append(ret[0])
            if with_jacobian:
                Jm_list.append(ret[1])
        if with_jacobian:
            return manip_list, Jm_list
        else:
            return manip_list

    def _approx(self, traj_q: np.ndarray):
        q0_list = traj_q
        manip0_list, Jm_list = self._calc_manip_list(q0_list, with_jacobian=True)
        self.q0_list = q0_list
        self.manip0_list = manip0_list
        self.Jm_list = Jm_list

    def _construct_approx_expr(self, traj_q: cp.Expression):
        q_list = traj_q
        expr_list = []
        for q, q0, manip0, Jm in zip(
            q_list, self.q0_list, self.manip0_list, self.Jm_list
        ):
            expr_list.append(self.min_manipulability - manip0 - Jm @ (q - q0))
        return cp.vstack(expr_list).flatten("C")

    def _eval(self, traj_q: np.ndarray):
        q_list = traj_q
        manip_list = self._calc_manip_list(q_list, with_jacobian=False)
        value = self.min_manipulability - np.array(manip_list)
        return value.flatten()
