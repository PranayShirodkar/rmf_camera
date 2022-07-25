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

    // User Configurations
    struct Config
    {
        // Camera configurations
        std::string camera_name;
        const float camera_afov;
        const float camera_pose_z;
        const float camera_pose_p;
        // YoloDetector configurations
        const float score_threshold;
        const float nms_threshold;
        const float confidence_threshold;
        const bool visualize = true;
    };

    YoloDetector(Config config);
    ~YoloDetector();
    Obstacles imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg);

private:

    // Yolov5s Constants
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;

    // Members
    // image_transport::ImageTransport it;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    std::vector<std::string> class_list_;
    cv::dnn::Net net_;
    Config _config;
    float _d_config;
    float _w_config;

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