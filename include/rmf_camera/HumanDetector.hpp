#ifndef HUMANDETECTOR_HPP
#define HUMANDETECTOR_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rmf_obstacle_msgs/msg/obstacles.hpp>
#include <rmf_camera/YoloDetector.hpp>

class HumanDetector : public rclcpp::Node
{
public:
  using Obstacles = rmf_obstacle_msgs::msg::Obstacles;
  HumanDetector();
  ~HumanDetector();

private:
  YoloDetector::Config get_config();

  struct Data
  {
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _sub;
    rclcpp::Publisher<Obstacles>::SharedPtr _pub;
    std::shared_ptr<YoloDetector> _yoloDetector;
  };
  std::shared_ptr<Data> _data;
};

#endif // HUMANDETECTOR_HPP
