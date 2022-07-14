#ifndef YOLODETECTOR_HPP
#define YOLODETECTOR_HPP

#include <string>

// ROS includes
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

// Project includes
#include <rmf_obstacle_msgs/msg/obstacles.hpp>

class YoloDetector
{
public:
    using Obstacles = rmf_obstacle_msgs::msg::Obstacles;

    YoloDetector();
    ~YoloDetector();
    Obstacles imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

private:

    // Constants
    const std::string OPENCV_WINDOW = "Image window";
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;

    // Threshold Configurations
    const float SCORE_THRESHOLD = 0.45;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.25;

    // Camera Configurations
    const float D_config = 5.28; // STANDING_PERSON_X
    const float CAMERA_AFOV = 2;
    const float CAMERA_PITCH = 0.3;
    const float CAMERA_TO_HUMAN = D_config/cos(CAMERA_PITCH); // real world distance from camera to center of human bounding box
    const float W_config = CAMERA_TO_HUMAN*tan(CAMERA_AFOV/2); // width of real world at depth of human bounding box

    // Members
    // image_transport::ImageTransport it;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    std::vector<std::string> class_list_;
    cv::dnn::Net net_;

    // Methods
    cv::Mat format_yolov5(const cv::Mat &source);
    std::vector<cv::Mat> detect(cv::Mat &input_image);
    Obstacles post_process(const cv::Mat &original_image, cv::Mat &image, std::vector<cv::Mat> &detections);
    cv::Point3d img_coord_to_cam_coord(const cv::Point &centroid, const cv::Mat &original_image);
    Obstacles to_rmf_obstacles(const cv::Mat &original_image, const std::vector<int> &final_class_ids, const std::vector<float> &final_confidences, const std::vector<cv::Rect> &final_boxes, const std::vector<cv::Point> &final_centroids);
    void drawing(const cv::Mat &original_image, cv::Mat &image, const std::vector<int> &final_class_ids, const std::vector<float> &final_confidences, const std::vector<cv::Rect> &final_boxes, const std::vector<cv::Point> &final_centroids);
    void draw_label(cv::Mat &input_image, std::string label, int left, int top);

};

#endif // YOLODETECTOR_HPP