#include "discrete_collision_term.hpp"
#include "ee_pose_term.hpp"

#include <boost/python.hpp>

#include <eigenpy/eigenpy.hpp>

BOOST_PYTHON_MODULE(opt_term_cpp) {
  eigenpy::enableEigenPy();

  std::string parent_module_name = "opt_term_cpp";

  expose_ee_pose_term(parent_module_name);
  expose_discrete_collision_term(parent_module_name);
}
