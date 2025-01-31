#include <boost/python.hpp>
#include <omp.h>

#include <Eigen/Core>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/pool/model.hpp>
#include <pinocchio/spatial/se3.hpp>

namespace bp = boost::python;
namespace pin = pinocchio;

void expose_ee_pose_term(const std::string &parent_module_name);

class EEPoseTerm {
public:
  EEPoseTerm(const pin::Model &model, int ee_frame_id, double delta_q,
             const bp::list &target_pose_inv_list_bp, int num_threads)
      : pool_(model, num_threads), ee_frame_id_(ee_frame_id), delta_q_(delta_q),
        num_threads_(num_threads) {
    omp_set_num_threads(num_threads_);

    const std::size_t num_target_poses = bp::len(target_pose_inv_list_bp);
    target_pose_inv_list_.resize(num_target_poses);
    for (std::size_t i = 0; i < num_target_poses; ++i) {
      target_pose_inv_list_[i] =
          bp::extract<pin::SE3>(target_pose_inv_list_bp[i]);
    }
  }

  bp::tuple calc_pose_err(const Eigen::VectorXd &q,
                          const pin::SE3 &target_pose_inv, bool with_jacobian);
  bp::tuple calc_pose_err_list(const Eigen::MatrixXd &q_list,
                               bool with_jacobian = false);

private:
  pin::ModelPool pool_;
  int ee_frame_id_;
  double delta_q_;
  std::vector<pin::SE3> target_pose_inv_list_;
  int num_threads_;
};
