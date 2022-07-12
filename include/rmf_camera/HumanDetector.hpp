#ifndef HUMANDETECTOR_HPP
#define HUMANDETECTOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <rmf_obstacle_ros2/Detector.hpp>
#include <rmf_obstacle_msgs/msg/obstacles.hpp>

namespace rmf_human_detector {
//==============================================================================
class HumanDetector : public rmf_obstacle_ros2::Detector
{
public:
  using Obstacles = rmf_obstacle_ros2::Detector::Obstacles;
  using DetectorCallback = rmf_obstacle_ros2::Detector::DetectorCallback;

  /// Documentation inherited
  void initialize(
    const rclcpp::Node& node,
    DetectorCallback cb) final;

  /// Documentation inherited
  std::string name() const final;

private:
  DetectorCallback _cb;
  std::string _name = "rmf_human_detector";

};

} // namespace rmf_human_detector

#endif
