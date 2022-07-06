cmake_minimum_required(VERSION 3.8)
project(rmf_camera)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(common_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rmf_obstacle_ros2 REQUIRED)
find_package(pluginlib REQUIRED)


#=============================================================

add_library(rmf_human_detector SHARED src/HumanDetector.cpp)

target_include_directories(rmf_human_detector
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    ${rmf_obstacle_ros2_INCLUDE_DIRS}
    ${pluginlib_INCLUDE_DIRS}
    ${rclcpp_INCLUDE_DIRS}
)

target_link_libraries(rmf_human_detector
  PUBLIC
    rmf_obstacle_ros2::rmf_obstacle_ros2
    pluginlib::pluginlib
    rclcpp::rclcpp
)

target_compile_features(rmf_human_detector INTERFACE cxx_std_17)
target_compile_definitions(rmf_human_detector PUBLIC "PLUGINLIB__DISABLE_BOOST_FUNCTIONS")

pluginlib_export_plugin_description_file(rmf_obstacle_ros2 plugins.xml)

ament_export_libraries(
  rmf_human_detector
)
# ament_export_targets(
#   export_${PROJECT_NAME}
# )

ament_export_dependencies(rclcpp rmf_obstacle_ros2 pluginlib)

install(
  TARGETS rmf_human_detector
  EXPORT rmf_human_detector
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

#=============================================================
add_executable(YoloDetector src/YoloDetector.cpp)


target_include_directories(YoloDetector PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(YoloDetector PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

ament_target_dependencies(
  YoloDetector
  rclcpp
  sensor_msgs
  image_transport
  cv_bridge
  common_interfaces
  OpenCV
  rmf_obstacle_ros2
)

ament_environment_hooks("${CMAKE_CURRENT_SOURCE_DIR}/env-hooks/${PROJECT_NAME}.dsv.in")

install(TARGETS YoloDetector
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY
  launch models assets worlds
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
