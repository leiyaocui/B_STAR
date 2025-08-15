import coal
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

    def _calc_collisions(self, q: np.ndarray, only_sd: bool = False):
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

            if only_sd:
                collisions.append([sd])
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

    def _calc_collisions_list(self, traj_q: np.ndarray, only_sd: bool = False):
        collisions_list = []
        for q in traj_q:
            collisions = self._calc_collisions(q, only_sd=only_sd)
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
                sd = sd0 + J @ (q - q0)
                expr.append(self.sd_safe - sd)
        if len(expr) == 0:
            return cp.Constant(np.zeros(1))
        expr = cp.vstack(expr)
        return expr.flatten("C")

    def _eval(self, traj_q: np.ndarray):
        collisions_list = self._calc_collisions_list(traj_q, only_sd=True)
        value = []
        for collisions in collisions_list:
            for collision in collisions:
                value.append(self.sd_safe - collision[0])
        if len(value) == 0:
            return np.zeros(1)
        value = np.array(value)
        return value.flatten()


class ContinuousCollisionTerm(Term):
    def __init__(
        self,
        model: pin.Model,
        collision_model: pin.GeometryModel,
        sd_check: float,
        sd_safe: float,
        sd_eq_tol: float = 0.1,
        name: str = "continuous-time collision avoidance",
        threshold: float = 1e-4,
    ):
        super().__init__(TermType.CONSTRAINT_INEQ, name, threshold)
        self.model = model
        self.data0 = self.model.createData()
        self.data1 = self.model.createData()
        self.collision_model = collision_model
        self.collision_data0 = self.collision_model.createData()
        self.collision_data1 = self.collision_model.createData()
        self.sd_check = sd_check
        self.sd_safe = sd_safe
        self.sd_eq_tol = sd_eq_tol
        assert self.sd_check >= self.sd_safe

    def _calc_swept_volume_collisions(
        self, q0: np.ndarray, q1: np.ndarray, only_sd: bool = False
    ):
        pin.forwardKinematics(self.model, self.data0, q0)
        pin.updateFramePlacements(self.model, self.data0)
        pin.updateGeometryPlacements(
            self.model, self.data0, self.collision_model, self.collision_data0
        )
        pin.computeJointJacobians(self.model, self.data0)

        pin.forwardKinematics(self.model, self.data1, q1)
        pin.updateFramePlacements(self.model, self.data1)
        pin.updateGeometryPlacements(
            self.model, self.data1, self.collision_model, self.collision_data1
        )
        pin.computeJointJacobians(self.model, self.data1)

        collisions = []
        for collision_pair in self.collision_model.collisionPairs:
            # TODO: assume the first geometry is the static obstacle
            # self.collision_model.geometryObjects[collision_pair.first].geometry.points()

            vertices = self.collision_model.geometryObjects[
                collision_pair.second
            ].geometry.points()  # (N, 3)

            tf0 = self.collision_data0.oMg[collision_pair.second].homogeneous  # (4, 4)
            tf1 = self.collision_data1.oMg[collision_pair.second].homogeneous  # (4, 4)

            vertices0 = vertices @ tf0[:3, :3].T + tf0[:3, 3].reshape(1, 3)
            vertices1 = vertices @ tf1[:3, :3].T + tf1[:3, 3].reshape(1, 3)

            vertices_all = coal.StdVec_Vec3s()
            vertices_all.extend(vertices0)
            vertices_all.extend(vertices1)

            hull = coal.Convex.convexHull(vertices_all, True, "Qt")

            distance_req = coal.DistanceRequest()
            distance_res = coal.DistanceResult()
            distance_res.clear()

            coal.distance(
                hull,
                coal.Transform3s.Identity(),
                self.collision_model.geometryObjects[collision_pair.first].geometry,
                self.collision_model.geometryObjects[collision_pair.first].placement,
                distance_req,
                distance_res,
            )

            sd = distance_res.min_distance
            if sd > self.sd_check:
                continue

            if only_sd:
                collisions.append([sd])
                continue

            pointA = distance_res.getNearestPoint1().copy()  # (3,) in world frame
            pointB = distance_res.getNearestPoint2().copy()  # (3,) in world frame
            normal = (
                -distance_res.normal.copy()
            )  # (3,) pointing from A to B in world frame
            if np.linalg.norm(normal) < 1e-6:
                continue

            pointA0 = vertices0[
                np.argmax(np.sum(vertices0 * -normal.reshape(1, -1), axis=-1))
            ]  # (3,) in world frame
            pointA1 = vertices1[
                np.argmax(np.sum(vertices1 * -normal.reshape(1, -1), axis=-1))
            ]  # (3,) in world frame
            sd0 = np.sum((pointA0 - pointB) * normal)
            sd1 = np.sum((pointA1 - pointB) * normal)

            joint_id_A = self.collision_model.geometryObjects[
                collision_pair.second
            ].parentJoint

            if sd1 - sd0 > self.sd_eq_tol:
                J_pointA0_q = pin.getFrameJacobian(
                    self.model,
                    self.data0,
                    joint_id_A,
                    pin.SE3(np.eye(3), self.data0.oMi[joint_id_A].actInv(pointA0)),
                    pin.LOCAL_WORLD_ALIGNED,
                )[:3].copy()
                J0 = normal.reshape(1, -1) @ J_pointA0_q
                J1 = None
            elif sd0 - sd1 > self.sd_eq_tol:
                J_pointA1_q = pin.getFrameJacobian(
                    self.model,
                    self.data1,
                    joint_id_A,
                    pin.SE3(np.eye(3), self.data1.oMi[joint_id_A].actInv(pointA1)),
                    pin.LOCAL_WORLD_ALIGNED,
                )[:3].copy()
                J0 = None
                J1 = normal.reshape(1, -1) @ J_pointA1_q
            else:
                length_A0_A = np.linalg.norm(pointA0 - pointA, ord=2)
                length_A1_A = np.linalg.norm(pointA1 - pointA, ord=2)

                if length_A0_A + length_A1_A < 1e-3:
                    alpha = 0.5
                else:
                    alpha = length_A1_A / (length_A0_A + length_A1_A)

                J_pointA0_q = pin.getFrameJacobian(
                    self.model,
                    self.data0,
                    joint_id_A,
                    pin.SE3(np.eye(3), self.data0.oMi[joint_id_A].actInv(pointA0)),
                    pin.LOCAL_WORLD_ALIGNED,
                )[:3].copy()
                J_pointA1_q = pin.getFrameJacobian(
                    self.model,
                    self.data1,
                    joint_id_A,
                    pin.SE3(np.eye(3), self.data1.oMi[joint_id_A].actInv(pointA1)),
                    pin.LOCAL_WORLD_ALIGNED,
                )[:3].copy()
                J0 = alpha * normal.reshape(1, -1) @ J_pointA0_q
                J1 = (1 - alpha) * normal.reshape(1, -1) @ J_pointA1_q

            collisions.append([sd, J0, q0, J1, q1])
        return collisions

    def _calc_swept_volume_collisions_list(
        self, traj_q: np.ndarray, only_sd: bool = False
    ):
        collisions_list = []
        for i in range(traj_q.shape[0] - 1):
            collisions = self._calc_swept_volume_collisions(
                traj_q[i], traj_q[i + 1], only_sd=only_sd
            )
            collisions_list.append(collisions)
        return collisions_list

    def _approx(self, traj_q: np.ndarray):
        self.collisions_list: list[list[tuple[float, np.ndarray, np.ndarray]]] = (
            self._calc_swept_volume_collisions_list(traj_q)
        )

    def _construct_approx_expr(self, traj_q: cp.Expression):
        expr = []
        for i in range(traj_q.shape[0] - 1):
            collisions = self.collisions_list[i]
            for collision in collisions:
                sd0, J0, q0, J1, q1 = collision
                sd = sd0
                if J0 is not None:
                    sd += J0 @ (traj_q[i] - q0)
                if J1 is not None:
                    sd += J1 @ (traj_q[i + 1] - q1)
                expr.append(self.sd_safe - sd)
        if len(expr) == 0:
            return cp.Constant(np.zeros(1))
        expr = cp.vstack(expr)
        return expr.flatten("C")

    def _eval(self, traj_q: np.ndarray):
        collisions_list = self._calc_swept_volume_collisions_list(traj_q, only_sd=True)
        value = []
        for collisions in collisions_list:
            for collision in collisions:
                value.append(self.sd_safe - collision[0])
        if len(value) == 0:
            return np.zeros(1)
        value = np.array(value)
        return value.flatten()
