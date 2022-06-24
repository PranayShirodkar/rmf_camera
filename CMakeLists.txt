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
find_package(cv_bridge REQUIRED)
find_package(common_interfaces REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED IMPORTED_TARGET opencv4)

add_executable(image_proc src/image_proc.cpp)
target_include_directories(image_proc PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(image_proc PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

target_compile_options(image_proc PUBLIC ${OPENCV4_CFLAGS_OTHER})
target_link_libraries(image_proc ${OPENCV4_LIBRARIES})
target_include_directories(image_proc PUBLIC ${OPENCV4_INCLUDE_DIRS})

ament_target_dependencies(
  image_proc
  "rclcpp"
  "sensor_msgs"
  "image_transport"
  "cv_bridge"
  "common_interfaces"
)

install(TARGETS image_proc
  DESTINATION lib/${PROJECT_NAME})

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
