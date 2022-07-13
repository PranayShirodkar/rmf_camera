#ifndef HUMANDETECTOR_HPP
#define HUMANDETECTOR_HPP

#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rmf_obstacle_ros2/Detector.hpp>
#include <rmf_obstacle_msgs/msg/obstacles.hpp>
#include <rmf_camera/YoloDetector.hpp>

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

  HumanDetector();
  ~HumanDetector();

private:
  struct Data
  {
    DetectorCallback _cb;
    std::string _name = "rmf_human_detector";
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _sub;
    rclcpp::Node::SharedPtr _node;
    std::thread _thread;
    std::shared_ptr<YoloDetector> _yoloDetector;

    Data()
    {
      _node = std::make_shared<rclcpp::Node>("human_detector_node");
      _thread = std::thread(
        [n = _node]()
        {
          while(rclcpp::ok())
            rclcpp::spin_some(n);
        }
      );
      _yoloDetector = std::make_shared<YoloDetector>();
    }
  };
  std::shared_ptr<Data> _data;

};

} // namespace rmf_human_detector

#endif // HUMANDETECTOR_HPP
