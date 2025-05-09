cmake_minimum_required(VERSION 3.14)

project(opt_term_cpp)

set(CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -O3 -fopenmp")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Eigen3 REQUIRED)
find_package(pinocchio REQUIRED)

add_library(${PROJECT_NAME} SHARED
    ${CMAKE_SOURCE_DIR}/src/opt_term.cpp
    ${CMAKE_SOURCE_DIR}/src/ee_pose_term.cpp
    ${CMAKE_SOURCE_DIR}/src/discrete_collision_term.cpp
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    -DPINOCCHIO_WITH_URDFDOM
    -DPINOCCHIO_WITH_SDFORMAT
    -DPINOCCHIO_WITH_HPP_FCL
    -DPINOCCHIO_WITH_EXTRA_SUPPORT
    -DPINOCCHIO_ENABLE_TEMPLATE_INSTANTIATION
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIRS}
    ${EIGENPY_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIRS}
    ${COAL_INCLUDE_DIRS}
    ${PYTHON_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${PINOCCHIO_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
    ${EIGEN3_LIBRARIES}
    ${EIGENPY_LIBRARIES}
    ${NUMPY_LIBRARIES}
    ${COAL_LIBRARIES}
    ${PYTHON_LIBRARIES}
    ${Boost_LIBRARIES}
    ${Boost_PYTHON_LIBRARY}
    ${PINOCCHIO_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX ""
    SUFFIX ".so"
)