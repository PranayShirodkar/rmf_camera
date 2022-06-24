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
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class ImageProc : public rclcpp::Node {
private:
    const std::string OPENCV_WINDOW = "Image window";
//    image_transport::ImageTransport it;
    image_transport::Subscriber sub_;

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        printf("Image received \tSize: %dx%d - Timestamp: %u.%u sec - Encoding: %s\n",
                msg->width, msg->height,
                msg->header.stamp.sec,msg->header.stamp.nanosec, msg->encoding.c_str());
        if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
            cv::circle(cv_ptr->image, cv::Point(50, 50), 50, CV_RGB(255,0,0));

        cv::imshow(OPENCV_WINDOW, cv_ptr->image);
        cv::waitKey(3);
    }

public:
    ImageProc() : Node("image_converter") {
        cv::namedWindow(OPENCV_WINDOW, cv::WINDOW_AUTOSIZE);

        rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
        sub_ = image_transport::create_subscription(this, "camera/image_raw",
                std::bind(&ImageProc::imageCallback, this, std::placeholders::_1), "raw", custom_qos);
    }

    ~ImageProc()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageProc>());
    rclcpp::shutdown();
    return 0;
}