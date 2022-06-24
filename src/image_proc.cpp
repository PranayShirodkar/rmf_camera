#include <memory>
#include <cstdio>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
// #include <sensor_msgs/image_encodings.h>
// #include <std_msgs/msg/string.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/qos.hpp>

#include <memory>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>

// OpenCV Window Name
const std::string OPENCV_WINDOW = "Image window";

void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
  cv_bridge::CvImagePtr cv_ptr;

  cv_ptr = cv_bridge::toCvCopy(msg,msg->encoding);
  if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
  // cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));
  cv::imshow(OPENCV_WINDOW, cv_ptr->image);
  cv::waitKey(3);

  // printf("Image received \tSize: %dx%d - Timestamp: %u.%u sec - Encoding: %s\n",
  //               msg->width, msg->height,
  //               msg->header.stamp.sec,msg->header.stamp.nanosec, msg->encoding.c_str());
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("image_listener");

  cv::namedWindow(OPENCV_WINDOW, cv::WINDOW_AUTOSIZE);

  rclcpp::QoS video_qos(10);
  video_qos.keep_last(10);
  video_qos.best_effort();
  video_qos.durability_volatile();
  auto sub = node->create_subscription<sensor_msgs::msg::Image>(
              "depth_camera/image_raw", video_qos, imageCallback);

  rclcpp::spin(node);
  cv::destroyWindow(OPENCV_WINDOW);
  rclcpp::shutdown();
  return 0;
}
