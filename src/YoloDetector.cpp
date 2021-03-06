#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <memory>

// ROS includes
// #include <sensor_msgs/image_encodings.h>
// #include <image_transport/image_transport.hpp>
// #include <rclcpp/qos.hpp>

// Project includes
#include "YoloDetector.hpp"
#include "BoundingBox3D.hpp"

// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
const Scalar BLACK = Scalar(0,0,0);
const Scalar BLUE = Scalar(255, 178, 50);
const Scalar YELLOW = Scalar(0, 255, 255);
const Scalar RED = Scalar(0,0,255);

Mat YoloDetector::format_yolov5(const Mat &source) {
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    // int xpad = 0;
    // int ypad = 0;
    // if (col > row) {
    //     xpad = 0;
    //     ypad = (col - row) / 2;
    // }
    // else {
    //     xpad = (row - col) / 2;
    //     ypad = 0;
    // }

    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

vector<Mat> YoloDetector::detect(Mat &input_image)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net_.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());

    return outputs;
}

void YoloDetector::post_process(const Mat &original_image, Mat &image, vector<Mat> &detections)
{
    // Initialize vectors to hold respective outputs while unwrapping     detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    vector<Point> centroids; // image coordinates, Point is int
    // Resizing factor.
    float x_factor = image.cols / INPUT_WIDTH;
    float y_factor = image.rows / INPUT_HEIGHT;
    float *data = (float *)detections[0].data;
    const int dimensions = 5 + class_list_.size();
    // 25200 for default size 640.
    const int rows = 25200;
    // Iterate through 25200 detections.
    for (int i = 0; i < rows; ++i)
    {
        float confidence = data[4];
        // Discard bad detections and continue.
        if (confidence >= CONFIDENCE_THRESHOLD)
        {
            float * classes_scores = data + 5;
            // Create a 1x85 Mat and store class scores of 80 classes.
            Mat scores(1, class_list_.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            // class_id.x == 0 corresponds to objects labelled "person"
            if (max_class_score > SCORE_THRESHOLD && class_id.x == 0)
            {
                // Store class ID and confidence in the pre-defined respective vectors.
                confidences.push_back(confidence);
                class_ids.push_back(class_id.x);
                // Center.
                float px = data[0];
                float py = data[1];
                // Box dimension.
                float w = data[2];
                float h = data[3];
                // Bounding box coordinates.
                int left = int((px - 0.5 * w) * x_factor);
                int top = int((py - 0.5 * h) * y_factor);
                int width = int(w * x_factor);
                int height = int(h * y_factor);
                // Store good detections in the boxes vector.
                boxes.push_back(Rect(left, top, width, height));
                centroids.push_back(Point(px * x_factor, py * y_factor));
            }
        }
        // Jump to the next row.
        data += dimensions;
    }

    vector<Point2d> obstacles; // camera coordinates, Point2d is double
    vector<int> final_class_ids;

    // Perform Non-Maximum Suppression and draw predictions.
    vector<int> indices;
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);
    for (size_t i = 0; i < indices.size(); i++)
    {
        int idx = indices[i];
        Rect box = boxes[idx];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);
        // Draw centroid.
        circle(image, centroids[idx], 2, CV_RGB(255,0,0), -1);
        printf("Pixel: (%d, %d)\n", centroids[idx].x, centroids[idx].y);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_list_[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(image, label, left, top);

        // save camera coordinates.
        Point2d obstacle = img_coord_to_cam_coord(centroids[idx], original_image);
        obstacles.push_back(obstacle);
        final_class_ids.push_back(class_ids[idx]);
    }

    // publish rmf_obstacle
    auto message = std_msgs::msg::String();
    for (size_t i = 0; i < obstacles.size(); i++) {
        string class_label = class_list_[final_class_ids[i]];
        BoundingBox3D bb = {
            class_label,  // classification
            static_cast<int>(i),  // id
            Point3d(obstacles[i].x, obstacles[i].y, 0.0),  // position
            Vec3d(1.0, 1.0, 2.0),  // size
        };
        message.data = bb.toString();
        RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        publisher_->publish(message);
    }

    // Put efficiency information.
    // The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes).
    vector<double> layersTimes;
    double freq = getTickFrequency() / 1000;
    double t = net_.getPerfProfile(layersTimes) / freq;
    string label = format("Inference time : %.2f ms", t);
    putText(image, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);

    // Slicing to crop the image
    image = image(Range(0,original_image.rows),Range(0,original_image.cols));

    // Draw image center
    circle(image, Point(original_image.cols/2, original_image.rows/2), 2, CV_RGB(255,255,0), -1);
}

Point2d YoloDetector::img_coord_to_cam_coord(const Point &centroid, const Mat &original_image) {
    // img coordinates are pixel x & y position in the frame, px, py
    // camera coordinates are in bird's eye view (top down), with origin at camera, cx, cy
    // cx is +ve toward front of camera, cy is +ve toward left of camera
    const float WIDTH_PER_PIXEL_M = W_config*2/original_image.cols;
    const float DEPTH_PER_PIXEL_M = D_config*2/original_image.rows;
    float px = centroid.x;
    float py = centroid.y;
    float factor =  py/(original_image.rows/2);
    // float pitch_factor = (25/CAMERA_PITCH);
    float depth_per_pixel_m_dynamic = DEPTH_PER_PIXEL_M;
    if (factor > 1) {
        depth_per_pixel_m_dynamic = DEPTH_PER_PIXEL_M*factor;
    }
    else {
        depth_per_pixel_m_dynamic = DEPTH_PER_PIXEL_M*(((1 - factor)*100));
        // depth_per_pixel_m_dynamic = DEPTH_PER_PIXEL_M*(((1 - factor)*pitch_factor));
    }
    float cx = D_config + (depth_per_pixel_m_dynamic * ((original_image.rows/2) - py));
    if (cx < 0) cx = 0.1;

    float width_per_pixel_m_dynamic = WIDTH_PER_PIXEL_M * cx / D_config;
    float cy = width_per_pixel_m_dynamic * ((original_image.cols/2) - px);
    return Point2d(cx, cy);
}

void YoloDetector::draw_label(Mat& input_image, string label, int left, int top)
{
    // Display the label at the top of the bounding box.
    int baseLine;
    Size label_size = getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS, &baseLine);
    top = max(top, label_size.height);
    // Top left corner.
    Point tlc = Point(left, top);
    // Bottom right corner.
    Point brc = Point(left + label_size.width, top + label_size.height + baseLine);
    // Draw white rectangle.
    rectangle(input_image, tlc, brc, BLACK, FILLED);
    // Put the label on the black rectangle.
    putText(input_image, label, Point(left, top + label_size.height), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS);
}

void YoloDetector::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    // printf("Image received \tSize: %dx%d - Timestamp: %u.%u sec - Encoding: %s\n",
    //                 msg->width, msg->height,
    //                 msg->header.stamp.sec,msg->header.stamp.nanosec, msg->encoding.c_str());
    cv_bridge::CvImagePtr cv_ptr;
    cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    Mat original_image = cv_ptr->image;
    // Mat original_image = imread("/home/osrc/Pictures/Screenshots/empty_world/test.png");
    Mat image = format_yolov5(original_image);
    vector<Mat> detections = detect(image);
    post_process(original_image, image, detections);
    imshow(OPENCV_WINDOW, image);
    // imwrite("/home/osrc/Pictures/Screenshots/empty_world/1.jpg", image);
    waitKey(3);
}

YoloDetector::YoloDetector() : Node("YoloDetector") {
    cv::namedWindow(OPENCV_WINDOW, cv::WINDOW_AUTOSIZE);

    // rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
            // sub_ = image_transport::create_subscription(this, "camera/image_raw",
            // std::bind(&YoloDetector::imageCallback, this, std::placeholders::_1), "raw", custom_qos);
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
            "camera/image_rect", 10, std::bind(&YoloDetector::imageCallback, this, std::placeholders::_1));
    publisher_ = this->create_publisher<std_msgs::msg::String>("topic_out", 10);
    auto pwd = string(filesystem::current_path());
    auto model_filepath = pwd + "/install/rmf_camera/share/rmf_camera/assets/yolov5s.onnx";
    net_ = readNet(model_filepath);
    auto labels_filepath = pwd + "/install/rmf_camera/share/rmf_camera/assets/coco.names";
    ifstream ifs(labels_filepath);
    string line;
    while (getline(ifs, line))
    {
        class_list_.push_back(line);
    }
}

YoloDetector::~YoloDetector()
{
    cv::destroyWindow(OPENCV_WINDOW);
}

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::cout << "Starting YoloDetector node" << std::endl;
    rclcpp::spin(std::make_shared<YoloDetector>());
    rclcpp::shutdown();
    return 0;
}