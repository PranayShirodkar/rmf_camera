#include <std_msgs/msg/string.hpp>

#include <rmf_camera/HumanDetector.hpp>
#include <rmf_camera/YoloDetector.hpp>

namespace rmf_human_detector {

//==============================================================================
void HumanDetector::initialize(
  const rclcpp::Node& node,
  DetectorCallback cb)
{
  RCLCPP_INFO(node.get_logger(), "Publishing: '%s'", "initialize start");
  _cb = std::move(cb);
  _node = std::make_shared<rclcpp::Node>("human_detector_node");
  _thread = std::thread(
    [n = _node]()
    {
      while(rclcpp::ok())
        rclcpp::spin_some(n);
    }
  );

  YoloDetector yoloDetector = YoloDetector();
  _sub = _node->create_subscription<sensor_msgs::msg::Image>(
  "camera/image_rect", 10,
  std::bind(&YoloDetector::imageCallback, yoloDetector, std::placeholders::_1));

  // init obstacles_msg
  auto obstacles_msg = rmf_obstacle_msgs::msg::Obstacles();
  int num_obstacles = 4;
  int id = 12;

  // prepare obstacle_msg objects and add to obstacles_msg
  obstacles_msg.obstacles.reserve(num_obstacles);
  for (int i = 0; i < num_obstacles; i++) {
    auto obstacle = rmf_obstacle_msgs::msg::Obstacle();
    // auto obstacle = rmf_obstacle_msgs::build<rmf_obstacle_msgs::msg::Obstacle>();
    obstacle.id = id++;
    obstacle.header.frame_id = "some stuff";
    obstacle.header.stamp = _node->get_clock()->now();

    obstacles_msg.obstacles.push_back(obstacle);
  }

  // publish obstacles_msg
  _cb(obstacles_msg);
  RCLCPP_INFO(node.get_logger(), "Publishing: '%s'", "initialize end");
}

//==============================================================================
std::string HumanDetector::name() const
{
  return _name;
}

HumanDetector::~HumanDetector()
{
  if (_thread.joinable())
    _thread.join();
}


} // rmf_human_detector

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  rmf_human_detector::HumanDetector,
  rmf_obstacle_ros2::Detector)