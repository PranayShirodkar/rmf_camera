#include <rmf_camera/HumanDetector.hpp>

HumanDetector::HumanDetector() : _data(std::make_shared<Data>())
{
  const std::string camera_topic = _data->_node->declare_parameter(
    "camera_topic", "/camera1/image_rect");
  const float camera_afov = _data->_node->declare_parameter(
    "camera_afov", 2.0);
  const float camera_pose_z = _data->_node->declare_parameter(
    "camera_pose_z", 2.5);
  const float camera_pose_p = _data->_node->declare_parameter(
    "camera_pose_p", 0.3);
  const float score_threshold = _data->_node->declare_parameter(
    "score_threshold", 0.45);
  const float nms_threshold = _data->_node->declare_parameter(
    "nms_threshold", 0.45);
  const float confidence_threshold = _data->_node->declare_parameter(
    "confidence_threshold", 0.25);
  const bool visualize = _data->_node->declare_parameter(
    "visualize", true);

  YoloDetector::Config config = {
                camera_topic,
                camera_afov,
                camera_pose_z,
                camera_pose_p,
                score_threshold,
                nms_threshold,
                confidence_threshold,
                visualize
            };
  _data->_yoloDetector = std::make_shared<YoloDetector>(config);

  _data->_sub = _data->_node->create_subscription<sensor_msgs::msg::Image>(
  camera_topic,
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

HumanDetector::~HumanDetector()
{
  if (_data->_thread.joinable())
    _data->_thread.join();
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  HumanDetector humanDetector = HumanDetector();
  return 0;
}
