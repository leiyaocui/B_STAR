#include "ee_pose_term.hpp"

#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

void expose_ee_pose_term(const std::string &parent_module_name) {
  bp::class_<EEPoseTerm>(
      "EEPoseTerm",
      bp::init<const pin::Model &, int, double, const bp::list &, int>())
      .def("calc_pose_err", &EEPoseTerm::calc_pose_err,
           (bp::arg("q"), bp::arg("target_pose_inv"),
            bp::arg("with_jacobian") = false))
      .def("calc_pose_err_list", &EEPoseTerm::calc_pose_err_list,
           (bp::arg("q_list"), bp::arg("with_jacobian") = false));
}

bp::tuple EEPoseTerm::calc_pose_err(const Eigen::VectorXd &q,
                                    const pin::SE3 &target_pose_inv,
                                    bool with_jacobian) {
  const auto &models = pool_.getModels();
  auto &datas = pool_.getDatas();

  pin::forwardKinematics(models[0], datas[0], q);
  pin::updateFramePlacements(models[0], datas[0]);

  pin::SE3 pose_err = target_pose_inv.act(datas[0].oMf[ee_frame_id_]);
  Eigen::VectorXd pose_err_vec = pin::log6(pose_err).toVector();

  if (with_jacobian) {
    const std::size_t num_dof = models[0].nq;

    Eigen::MatrixXd J(6, num_dof);
    J.setZero();

#pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < num_dof; ++i) {
      const std::size_t thread_id = omp_get_thread_num();

      const pin::Model &model = models[thread_id];
      pin::Data &data = datas[thread_id];

      Eigen::VectorXd q_perturbed = q;
      q_perturbed[i] += delta_q_;

      pin::forwardKinematics(model, data, q_perturbed);
      pin::updateFramePlacements(model, data);

      pin::SE3 pose_err_perturbed = target_pose_inv.act(data.oMf[ee_frame_id_]);

      J.col(i) =
          (pin::log6(pose_err_perturbed).toVector() - pose_err_vec) / delta_q_;
    }

    return bp::make_tuple(pose_err_vec, J);
  } else {
    return bp::make_tuple(pose_err_vec, bp::object());
  }
}

bp::tuple EEPoseTerm::calc_pose_err_list(const Eigen::MatrixXd &q_list,
                                         bool with_jacobian) {
  const std::size_t num_configs = q_list.rows();
  const std::size_t dof = q_list.cols();

  const auto &models = pool_.getModels();
  auto &datas = pool_.getDatas();

  std::vector<Eigen::VectorXd> pose_err_list(num_configs,
                                             Eigen::VectorXd::Zero(6));
  std::vector<Eigen::MatrixXd> J_list(num_configs,
                                      Eigen::MatrixXd::Zero(6, dof));

#pragma omp parallel for schedule(static)
  for (std::size_t i = 0; i < num_configs; ++i) {
    const std::size_t thread_id = omp_get_thread_num();

    const pin::Model &model = models[thread_id];
    pin::Data &data = datas[thread_id];
    const pin::SE3 &target_pose_inv = target_pose_inv_list_[i];

    const Eigen::VectorXd &q = q_list.row(i);

    pin::forwardKinematics(model, data, q);
    pin::updateFramePlacements(model, data);

    pin::SE3 pose_err = target_pose_inv.act(data.oMf[ee_frame_id_]);
    pose_err_list[i] = pin::log6(pose_err).toVector();
  }

  if (with_jacobian) {
#pragma omp parallel for schedule(static)
    for (std::size_t idx = 0; idx < num_configs * dof; ++idx) {
      const std::size_t thread_id = omp_get_thread_num();

      const pin::Model &model = models[thread_id];
      pin::Data &data = datas[thread_id];

      const std::size_t i = idx / dof;
      const std::size_t j = idx % dof;

      const pin::SE3 &target_pose_inv = target_pose_inv_list_[i];
      Eigen::VectorXd q_perturbed = q_list.row(i);
      q_perturbed(j) += delta_q_;

      pin::forwardKinematics(model, data, q_perturbed);
      pin::updateFramePlacements(model, data);

      pin::SE3 pose_err_perturbed = target_pose_inv.act(data.oMf[ee_frame_id_]);

      J_list[i].col(j) =
          (pin::log6(pose_err_perturbed).toVector() - pose_err_list[i]) /
          delta_q_;
    }
  }

  bp::list py_pose_err_list;
  bp::list py_J_list;

  for (std::size_t i = 0; i < num_configs; ++i) {
    py_pose_err_list.append(pose_err_list[i]);
    if (with_jacobian) {
      py_J_list.append(J_list[i]);
    }
  }

  return with_jacobian ? bp::make_tuple(py_pose_err_list, py_J_list)
                       : bp::make_tuple(py_pose_err_list, bp::object());
}
