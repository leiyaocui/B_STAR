import xml.etree.ElementTree as ET
from pathlib import Path

import hppfcl
import numpy as np
import pinocchio as pin
import yaml


def compute_ik(
    model: pin.Model,
    frame_id: int,
    T_ee2world: pin.SE3,
    rng_gen: np.random.Generator,
    q_init: np.ndarray = None,
    eps: float = 1e-4,
    iter_max: int = 1000,
    dt: float = 1e-1,
    damp: float = 1e-12,
):
    if q_init is None:
        q_init = rng_gen.uniform(
            model.lowerPositionLimit, model.upperPositionLimit, size=model.nq
        )

    data = model.createData()
    q = q_init.copy()
    success = False
    iter = 0
    while iter < iter_max:
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        T_link2world = data.oMf[frame_id]
        T_ee2link = T_link2world.actInv(T_ee2world)
        err = pin.log6(T_ee2link).np

        if np.linalg.norm(err) < eps:
            success = True
            break

        J = pin.computeFrameJacobian(model, data, q, frame_id)
        J = -np.dot(pin.Jlog6(T_ee2link.inverse()), J)
        v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(model, q, v * dt)

        iter += 1

    if not np.all(model.lowerPositionLimit + eps <= q) or not np.all(
        q <= model.upperPositionLimit - eps
    ):
        success = False

    if success:
        return q
    else:
        return None


def compute_traj_ik(
    model: pin.Model,
    frame_id: int,
    T_ee2world_list: pin.SE3,
    rng_gen: np.random.Generator,
    eps: float = 1e-4,
    iter_max: int = 1000,
    dt: float = 1e-1,
    damp: float = 1e-12,
):
    traj_q = []

    idx = 0
    while idx < len(T_ee2world_list):
        q = compute_ik(
            model,
            frame_id,
            T_ee2world_list[idx],
            rng_gen=rng_gen,
            q_init=traj_q[-1] if len(traj_q) > 0 else None,
            eps=eps,
            iter_max=iter_max,
            dt=dt,
            damp=damp,
        )
        if q is not None:
            traj_q.append(q)
            idx += 1
        else:
            traj_q = []
            idx = 0

    return traj_q


def convert_curobo_config_to_srdf(config: dict, with_ground_plane: bool = True):
    root = ET.Element("robot", name="cuRobo")

    robot_cfg = config["robot_cfg"]["kinematics"]

    disabled_collisions = robot_cfg["self_collision_ignore"]
    for link1, ignored_links in disabled_collisions.items():
        for link2 in ignored_links:
            ET.SubElement(
                root,
                "disable_collisions",
                link1=link1,
                link2=link2,
                reason="from cuRobo",
            )

    if with_ground_plane:
        ET.SubElement(
            root,
            "disable_collisions",
            link1=robot_cfg["root_link"],
            link2=robot_cfg["base_link"],
            reason="ignore the collision between ground plane (mount on root link) and base link",
        )

    srdf_str = ET.tostring(root, encoding="unicode")
    return "<?xml version='1.0'?>" + srdf_str


def load_robot(config_path: str, asset_path: str, with_ground_plane: bool = True):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    robot_cfg = config["robot_cfg"]["kinematics"]
    asset_path = Path(asset_path)
    urdf_path = asset_path / robot_cfg["urdf_path"]
    package_dir = asset_path / robot_cfg["asset_root_path"]
    srdf_str = convert_curobo_config_to_srdf(config, with_ground_plane)

    model = pin.buildModelFromUrdf(str(urdf_path))
    ee_frame_id = model.getBodyId(robot_cfg["ee_link"])

    collision_model = pin.buildGeomFromUrdf(
        model, urdf_path, pin.GeometryType.COLLISION, package_dirs=[str(package_dir)]
    )
    for geom_obj in collision_model.geometryObjects:
        geom_obj.geometry.buildConvexRepresentation(True)
        geom_obj.geometry = geom_obj.geometry.convex

    if with_ground_plane:
        ground_plane = pin.GeometryObject(
            "ground_plane", 0, 1, pin.SE3().Identity(), hppfcl.Plane(0.0, 0.0, 1.0, 0.0)
        )
        ground_plane_id = collision_model.addGeometryObject(ground_plane)
        for i in range(collision_model.ngeoms):
            if i != ground_plane_id:
                collision_model.addCollisionPair(pin.CollisionPair(ground_plane_id, i))

    collision_model.addAllCollisionPairs()
    pin.removeCollisionPairsFromXML(model, collision_model, srdf_str, False)

    return model, collision_model, ee_frame_id
