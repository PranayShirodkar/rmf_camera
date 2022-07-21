#include <rmf_camera/HumanDetector.hpp>

void HumanDetector::initialize()
{
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
    data->_obstacleManager->process(rmf_obstacles_msg);
  });
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

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  HumanDetector humanDetector = HumanDetector();
  humanDetector.initialize();
  return 0;
}
