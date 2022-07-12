#include <std_msgs/msg/string.hpp>

#include <rmf_camera/HumanDetector.hpp>

namespace rmf_human_detector {

//==============================================================================
void HumanDetector::initialize(
  const rclcpp::Node& node,
  DetectorCallback cb)
{
  _cb = std::move(cb);
  RCLCPP_INFO(node.get_logger(), "Publishing: '%s'", "initialize start");

  // detect all camera topics being published from gazebo world

  // subscribe to camera topics
  

  // for each camera subscription
    // initialize YoloDetector object, call imageCallback, return vector of obstacles
    // convert obstacles from camera coordinates to world coordinates

  // add all obstacles in rmf_obstacles::msg:Obstacles msg

  // init obstacles_msg
  auto obstacles_msg = rmf_obstacle_msgs::msg::Obstacles();
  int num_obstacles = 4;
  int id = 12;

  // prepare obstacle_msg objects and add to obstacles_msg
  obstacles_msg.obstacles.reserve(num_obstacles);
  for (int i = 0; i < num_obstacles; i++) {
    auto obstacle_msg = rmf_obstacle_msgs::msg::Obstacle();
    // auto obstacle = rmf_obstacle_msgs::build<rmf_obstacle_msgs::msg::Obstacle>();
    obstacle_msg.id = id++;
    obstacle_msg.header.frame_id = "some stuff";
    // obstacle.header.stamp = node.get_clock().now().to_msg();

    obstacles_msg.obstacles.push_back(obstacle_msg);
  }

  //publish obstacles_msg
  _cb(obstacles_msg);

}

//==============================================================================
std::string HumanDetector::name() const
{
  return _name;
}


} // rmf_human_detector

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  rmf_human_detector::HumanDetector,
  rmf_obstacle_ros2::Detector)