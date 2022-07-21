#ifndef HUMANDETECTOR_HPP
#define HUMANDETECTOR_HPP

#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <rmf_obstacle_msgs/msg/obstacles.hpp>
#include <rmf_obstacle_ros2/ObstacleManager.hpp>
#include <rmf_camera/YoloDetector.hpp>

class HumanDetector
{
public:
  using Obstacles = rmf_obstacle_msgs::msg::Obstacles;
  HumanDetector();
  ~HumanDetector();

private:
  struct Data
  {
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _sub;
    std::shared_ptr<rmf_obstacle_ros2::ObstacleManager> _obstacleManager;
    rclcpp::Node::SharedPtr _node;
    std::thread _thread;
    std::shared_ptr<YoloDetector> _yoloDetector;

    Data()
    {
      _obstacleManager = rmf_obstacle_ros2::ObstacleManager::make("human_detector");
      _node = _obstacleManager->node();
      _thread = std::thread(
        []()
        {
          while(rclcpp::ok()) {}
        }
      );
    }
  };
  std::shared_ptr<Data> _data;
};

#endif // HUMANDETECTOR_HPP
