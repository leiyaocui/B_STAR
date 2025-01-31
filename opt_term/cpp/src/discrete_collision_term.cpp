#include "discrete_collision_term.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/spatial/se3.hpp>

void expose_discrete_collision_term(const std::string &parent_module_name) {
  bp::class_<DiscreteCollisionTerm>(
      "DiscreteCollisionTerm",
      bp::init<const pin::Model &, const pin::GeometryModel &, double, double,
               int>())
      .def("calc_collisions", &DiscreteCollisionTerm::calc_collisions,
           bp::arg("q"))
      .def("calc_collisions_list", &DiscreteCollisionTerm::calc_collisions_list,
           bp::arg("q_list"));
}

bp::list DiscreteCollisionTerm::calc_collisions(const Eigen::VectorXd &q) {
  const auto &collision_pairs = collision_model_.collisionPairs;

  const std::size_t num_collisions = collision_pairs.size();

  pin::forwardKinematics(model_, data_, q);
  pin::updateFramePlacements(model_, data_);
  pin::updateGeometryPlacements(model_, data_, collision_model_,
                                collision_data_);
  pin::computeJointJacobians(model_, data_);

  std::vector<double> sd_list(num_collisions,
                              std::numeric_limits<double>::infinity());
  std::vector<Eigen::VectorXd> J_list(num_collisions);

#pragma omp parallel for schedule(dynamic)
  for (std::size_t i = 0; i < num_collisions; ++i) {
    const std::size_t thread_id = omp_get_thread_num();

    const auto &distance_result =
        pin::computeDistance(collision_model_, collision_data_, i);

    double sd = distance_result.min_distance;
    if (sd > sd_check_) {
      continue;
    }

    const Eigen::Vector3d &pointA = distance_result.nearest_points[0];
    const Eigen::Vector3d &pointB = distance_result.nearest_points[1];
    Eigen::Vector3d normal = pointA - pointB;
    double normal_length = normal.norm();
    if (normal_length < 1e-6) {
      continue;
    }
    normal /= normal_length;

    pin::JointIndex joint_id_A =
        collision_model_.geometryObjects[collision_pairs[i].first].parentJoint;
    pin::JointIndex joint_id_B =
        collision_model_.geometryObjects[collision_pairs[i].second].parentJoint;

    const Eigen::MatrixXd &J_pointA2q =
        pin::getFrameJacobian(model_, data_, joint_id_A,
                              pin::SE3(Eigen::Matrix3d::Identity(),
                                       data_.oMi[joint_id_A].actInv(pointA)),
                              pin::LOCAL_WORLD_ALIGNED);
    const Eigen::MatrixXd &J_pointB2q =
        pin::getFrameJacobian(model_, data_, joint_id_B,
                              pin::SE3(Eigen::Matrix3d::Identity(),
                                       data_.oMi[joint_id_B].actInv(pointB)),
                              pin::LOCAL_WORLD_ALIGNED);

    Eigen::VectorXd J =
        normal.transpose() * (J_pointA2q.topRows(3) - J_pointB2q.topRows(3));

    sd_list[i] = sd;
    J_list[i] = J;
  }

  bp::list py_collision_list;

  for (std::size_t i = 0; i < num_collisions; ++i) {
    if (sd_list[i] <= sd_check_) {
      bp::list collision;
      collision.append(sd_list[i]);
      collision.append(J_list[i]);
      collision.append(q);
      py_collision_list.append(collision);
    }
  }

  return py_collision_list;
}

bp::list
DiscreteCollisionTerm::calc_collisions_list(const Eigen::MatrixXd &q_list) {
  const std::size_t num_configs = q_list.rows();

  bp::list py_collision_list;

  for (std::size_t i = 0; i < num_configs; ++i) {
    const Eigen::VectorXd &q = q_list.row(i);

    const auto &result = calc_collisions(q);
    const std::size_t num_collisions = bp::len(result);
    for (std::size_t j = 0; j < num_collisions; ++j) {
      py_collision_list.append(result[j]);
    }
  }

  return py_collision_list;
}
