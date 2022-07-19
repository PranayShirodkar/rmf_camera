#include <std_msgs/msg/string.hpp>

#include <rmf_camera/HumanDetector.hpp>

namespace rmf_human_detector {

//==============================================================================
void HumanDetector::initialize(
  const rclcpp::Node& node,
  DetectorCallback cb)
{
  RCLCPP_INFO(node.get_logger(), "Publishing: '%s'", "initialize start");
  _data->_cb = std::move(cb);

  _data->_sub = _data->_node->create_subscription<sensor_msgs::msg::Image>(
  "camera1/image_rect",
  10,
  [data = _data](const sensor_msgs::msg::Image::ConstSharedPtr &msg)
  {
    // perform detections
    auto rmf_obstacles_msg = data->_yoloDetector->imageCallback(msg);

    // convert from camera coordinates to world coordinates
    // populate other fields like time stamp, etc


    // publish rmf_obstacles_msg
    data->_cb(rmf_obstacles_msg);
  });


  RCLCPP_INFO(_data->_node->get_logger(), "Publishing: '%s'", "initialize end");
}

//==============================================================================
std::string HumanDetector::name() const
{
  return _data->_name;
}

HumanDetector::HumanDetector() : _data(std::make_shared<Data>())
{
  // provide camera config in the argument, pass to _data->_yoloDetector
}

HumanDetector::~HumanDetector()
{
  if (_data->_thread.joinable())
    _data->_thread.join();
}


} // rmf_human_detector

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(
  rmf_human_detector::HumanDetector,
  rmf_obstacle_ros2::Detector)