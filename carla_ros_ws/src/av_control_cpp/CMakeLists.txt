cmake_minimum_required(VERSION 3.8)
project(av_control_cpp)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(osqp_vendor REQUIRED)
find_package(osqp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(carla_msgs REQUIRED)
find_package(derived_object_msgs REQUIRED)

# ROS-free algorithm library (what the gtests cover)
add_library(mpc_core src/mpc.cpp src/safety.cpp)
target_include_directories(mpc_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(mpc_core PUBLIC Eigen3::Eigen osqp::osqp)

add_executable(mpc_controller_node src/mpc_controller_node.cpp)
target_link_libraries(mpc_controller_node mpc_core)
ament_target_dependencies(mpc_controller_node
  rclcpp nav_msgs geometry_msgs std_msgs carla_msgs derived_object_msgs)

install(TARGETS mpc_core mpc_controller_node
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY include/ DESTINATION include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_mpc test/test_mpc.cpp)
  target_link_libraries(test_mpc mpc_core)
  ament_add_gtest(test_safety test/test_safety.cpp)
  target_link_libraries(test_safety mpc_core)
endif()

ament_package()
