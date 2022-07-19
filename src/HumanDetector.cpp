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

int main(int argc, char **argv)
{
  std::cout << "test" << std::endl;
  return 0;
  // rclcpp::init(argc, argv);
  // std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("image_listener");

  // cv::namedWindow(OPENCV_WINDOW, cv::WINDOW_AUTOSIZE);

  // net = readNet("/home/osrc/dev_ws/src/rmf_camera/src/yolov5s.onnx");
  // ifstream ifs("/home/osrc/dev_ws/src/rmf_camera/src/coco.names");
  // string line;
  // while (getline(ifs, line))
  // {
  //     class_list.push_back(line);
  // }

  // rclcpp::QoS video_qos(10);
  // video_qos.keep_last(10);
  // video_qos.best_effort();
  // video_qos.durability_volatile();
  // auto sub = node->create_subscription<sensor_msgs::msg::Image>(
  //             "depth_camera/image_raw", video_qos, imageCallback);

  // rclcpp::spin(node);
  // cv::destroyWindow(OPENCV_WINDOW);
  // rclcpp::shutdown();
  return 0;
}