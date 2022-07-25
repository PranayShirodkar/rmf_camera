#include <rclcpp/wait_for_message.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_msgs/msg/tf_message.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <rmf_camera/HumanDetector.hpp>

HumanDetector::HumanDetector() : Node("human_detector"), _data(std::make_shared<Data>())
{
  YoloDetector::Config config = get_config();

  _data->_yoloDetector = std::make_shared<YoloDetector>(config);

  _data->_pub = this->create_publisher<Obstacles>(
  "/rmf_obstacles",
  rclcpp::QoS(10).reliable()
  );

  const std::string camera_image_topic = config.camera_name + "/image_rect";
  _data->_sub = this->create_subscription<sensor_msgs::msg::Image>(
  camera_image_topic,
  10,
  [data = _data](const sensor_msgs::msg::Image::ConstSharedPtr &msg)
  {
    // perform detections
    auto rmf_obstacles_msg = data->_yoloDetector->imageCallback(msg);

    // convert from camera coordinates to world coordinates
    // populate other fields like time stamp, etc

    // publish rmf_obstacles_msg
    data->_pub->publish(rmf_obstacles_msg);
  });
}

YoloDetector::Config HumanDetector::get_config()
{
  // get ros2 params
  const std::string camera_name = this->declare_parameter(
    "camera_name", "/camera1");
  const float score_threshold = this->declare_parameter(
    "score_threshold", 0.45);
  const float nms_threshold = this->declare_parameter(
    "nms_threshold", 0.45);
  const float confidence_threshold = this->declare_parameter(
    "confidence_threshold", 0.25);
  const bool visualize = this->declare_parameter(
    "visualize", true);

  // get one camera_info and one pose msg
  const std::string camera_info_topic = camera_name + "/camera_info";
  const std::string camera_pose_topic = camera_name + "/pose";

  std::shared_ptr<rclcpp::Node> temp_node = std::make_shared<rclcpp::Node>("wait_for_msg_node");
  sensor_msgs::msg::CameraInfo camera_info;
  rclcpp::wait_for_message(camera_info, temp_node, camera_info_topic);
  tf2_msgs::msg::TFMessage camera_pose;
  rclcpp::wait_for_message(camera_pose, temp_node, camera_pose_topic);

  // calculate camera pitch
  tf2::Quaternion q(
    camera_pose.transforms[1].transform.rotation.x,
    camera_pose.transforms[1].transform.rotation.y,
    camera_pose.transforms[1].transform.rotation.z,
    camera_pose.transforms[1].transform.rotation.w);
  tf2::Matrix3x3 m(q);
  double roll, pitch, yaw;
  m.getRPY(roll, pitch, yaw);

  // calculate camera fov
  float f_x = camera_info.p[0];
  float fov_x = 2 * atan2( camera_info.width, (2*f_x) );

  // return config
  YoloDetector::Config config = {
                camera_name,
                fov_x,
                static_cast<float>(camera_pose.transforms[1].transform.translation.z),
                static_cast<float>(pitch),
                score_threshold,
                nms_threshold,
                confidence_threshold,
                visualize
  };
  return config;
}

HumanDetector::~HumanDetector()
{
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  std::cout << "Starting HumanDetector node" << std::endl;
  rclcpp::spin(std::make_shared<HumanDetector>());
  rclcpp::shutdown();
  return 0;
}
