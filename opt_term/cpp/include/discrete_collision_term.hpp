#include <boost/python.hpp>
#include <omp.h>

#include <Eigen/Core>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/multibody/model.hpp>

namespace bp = boost::python;
namespace pin = pinocchio;

void expose_discrete_collision_term(const std::string &parent_module_name);

class DiscreteCollisionTerm {
public:
  DiscreteCollisionTerm(const pin::Model &model,
                        const pin::GeometryModel &collision_model,
                        double sd_check, double sd_safe, int num_threads)
      : model_(model), data_(model), collision_model_(collision_model),
        collision_data_(collision_model_), sd_check_(sd_check),
        sd_safe_(sd_safe), num_threads_(num_threads) {
    if (sd_check < sd_safe) {
      throw std::invalid_argument("sd_check must be greater than sd_safe");
    }

    omp_set_num_threads(num_threads_);
  }

  bp::list calc_collisions(const Eigen::VectorXd &q);
  bp::list calc_collisions_list(const Eigen::MatrixXd &q_list);

private:
  const pin::Model &model_;
  pin::Data data_;
  const pin::GeometryModel &collision_model_;
  pin::GeometryData collision_data_;
  double sd_check_;
  double sd_safe_;
  int num_threads_;
};
