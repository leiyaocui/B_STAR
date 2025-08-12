import cvxpy as cp
import numpy as np
import pinocchio as pin

from .term import Term, TermType


class DiscreteCollisionTerm(Term):
    def __init__(
        self,
        model: pin.Model,
        collision_model: pin.GeometryModel,
        sd_check: float,
        sd_safe: float,
        name: str = "discrete-time collision avoidance",
        threshold: float = 1e-4,
    ):
        super().__init__(TermType.CONSTRAINT_INEQ, name, threshold)
        self.model = model
        self.data = self.model.createData()
        self.collision_model = collision_model
        self.collision_data = self.collision_model.createData()
        self.sd_check = sd_check
        self.sd_safe = sd_safe
        assert self.sd_check >= self.sd_safe

    def _calc_collisions(self, q: np.ndarray):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(
            self.model, self.data, self.collision_model, self.collision_data
        )
        pin.computeJointJacobians(self.model, self.data)
        pin.computeDistances(self.collision_model, self.collision_data)

        collisions = []
        for collision_pair, distance_result in zip(
            self.collision_model.collisionPairs,
            self.collision_data.distanceResults,
        ):
            sd = distance_result.min_distance
            if sd > self.sd_check:
                continue

            pointA = distance_result.getNearestPoint1().copy()  # in world frame
            pointB = distance_result.getNearestPoint2().copy()  # in world frame
            normal = -distance_result.normal.copy()
            if np.linalg.norm(normal) < 1e-6:
                continue

            joint_id_A = self.collision_model.geometryObjects[
                collision_pair.first
            ].parentJoint
            joint_id_B = self.collision_model.geometryObjects[
                collision_pair.second
            ].parentJoint

            J_pointA2q = pin.getFrameJacobian(
                self.model,
                self.data,
                joint_id_A,
                pin.SE3(np.eye(3), self.data.oMi[joint_id_A].actInv(pointA)),
                pin.LOCAL_WORLD_ALIGNED,
            )[:3].copy()
            J_pointB2q = pin.getFrameJacobian(
                self.model,
                self.data,
                joint_id_B,
                pin.SE3(np.eye(3), self.data.oMi[joint_id_B].actInv(pointB)),
                pin.LOCAL_WORLD_ALIGNED,
            )[:3].copy()
            J = normal.reshape(1, -1) @ (J_pointA2q - J_pointB2q)
            collisions.append([sd, J, q])
        return collisions

    def _calc_collisions_list(self, traj_q: np.ndarray):
        collisions_list = []
        for q in traj_q:
            collisions = self._calc_collisions(q)
            collisions_list.append(collisions)
        return collisions_list

    def _approx(self, traj_q: np.ndarray):
        self.collisions_list: list[list[tuple[float, np.ndarray, np.ndarray]]] = (
            self._calc_collisions_list(traj_q)
        )

    def _construct_approx_expr(self, traj_q: cp.Expression):
        expr = []
        for q, collisions in zip(traj_q, self.collisions_list):
            for collision in collisions:
                sd0, J, q0 = collision
                if sd0 is None:
                    continue
                sd = sd0 + J @ (q - q0)
                expr.append(self.sd_safe - sd)
        if len(expr) == 0:
            return cp.Constant(np.zeros(1))
        expr = cp.vstack(expr)
        return expr.flatten("C")

    def _eval(self, traj_q: np.ndarray):
        collisions_list = self._calc_collisions_list(traj_q)
        value = []
        for collisions in collisions_list:
            for collision in collisions:
                value.append(self.sd_safe - collision[0])
        if len(value) == 0:
            return np.zeros(1)
        value = np.array(value)
        return value.flatten()
