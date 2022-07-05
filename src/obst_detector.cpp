#include <memory>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <filesystem>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
// #include <sensor_msgs/image_encodings.h>
#include <std_msgs/msg/string.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/qos.hpp>

#include <memory>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>


// DNN
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

// Constants
// const float RAD_TO_DEGREE = 180/3.14159;
// const float DEGREE_TO_RAD = 3.14159/180;


class ObstDetector : public rclcpp::Node {
private:

    const std::string OPENCV_WINDOW = "Image window";

    // Constants
    const float INPUT_WIDTH = 640.0;
    const float INPUT_HEIGHT = 640.0;
    const float SCORE_THRESHOLD = 0.45;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.25;

    // Configurations
    const float D_config = 5.28; // STANDING_PERSON_X
    const float CAMERA_AFOV = 2;
    const float CAMERA_PITCH = 0.3;
    const float CAMERA_TO_HUMAN = D_config/cos(CAMERA_PITCH); // real world distance from camera to center of human bounding box
    const float W_config = CAMERA_TO_HUMAN*tan(CAMERA_AFOV/2); // width of real world at depth of human bounding box


    // Members
    // image_transport::ImageTransport it;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    vector<string> class_list_;
    Net net_;

    Mat format_yolov5(const Mat &source) {
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

    vector<Mat> detect(Mat &input_image)
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

    void post_process(const Mat &original_image, Mat &image, vector<Mat> &detections)
    {
        // Initialize vectors to hold respective outputs while unwrapping     detections.
        vector<int> class_ids;
        vector<float> confidences;
        vector<Rect> boxes;
        vector<Point> centroids;
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
                if (max_class_score > SCORE_THRESHOLD && class_id.x == 0)
                {
                    // Store class ID and confidence in the pre-defined respective vectors.
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);
                    // Center.
                    float cx = data[0];
                    float cy = data[1];
                    // Box dimension.
                    float w = data[2];
                    float h = data[3];
                    // Bounding box coordinates.
                    int left = int((cx - 0.5 * w) * x_factor);
                    int top = int((cy - 0.5 * h) * y_factor);
                    int width = int(w * x_factor);
                    int height = int(h * y_factor);
                    // Store good detections in the boxes vector.
                    boxes.push_back(Rect(left, top, width, height));
                    centroids.push_back(Point(cx * x_factor, cy * y_factor));
                }
            }
            // Jump to the next row.
            data += dimensions;
        }

        // 3D
        vector<Point3d> positions;
        const float WIDTH_PER_PIXEL_M = W_config*2/original_image.cols;
        const float DEPTH_PER_PIXEL_M = D_config*2/original_image.rows;

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

            // 3D
            float cx = centroids[idx].x;
            float cy = centroids[idx].y;
            float factor =  cy/(original_image.rows/2);
            // float pitch_factor = (25/CAMERA_PITCH);
            float depth_per_pixel_m_dynamic = DEPTH_PER_PIXEL_M;
            if (factor > 1) {
                depth_per_pixel_m_dynamic = DEPTH_PER_PIXEL_M*factor;
            }
            else {
                depth_per_pixel_m_dynamic = DEPTH_PER_PIXEL_M*(((1 - factor)*100));
                // depth_per_pixel_m_dynamic = DEPTH_PER_PIXEL_M*(((1 - factor)*pitch_factor));
            }
            float px = D_config + (depth_per_pixel_m_dynamic * ((original_image.rows/2) - cy));
            if (px < 0) px = 0.1;

            float width_per_pixel_m_dynamic = WIDTH_PER_PIXEL_M * px / D_config;
            float py = width_per_pixel_m_dynamic * ((original_image.cols/2) - cx);
            float pz = 0.0;
            positions.push_back(Point3d(px, py, pz));
            printf("3D  P: (%.3f, %.3f, %.3f)\n", px, py, pz);

            // publish rmf_obstacle
            auto message = std_msgs::msg::String();
            for (size_t i = 0; i < positions.size(); i++) {
                //class_list_[class_ids[idx]] = "person"
                message.data = class_list_[class_ids[idx]] + " " + std::to_string(class_ids[idx])
                + " " + std::to_string(positions[i].x) + " " + std::to_string(positions[i].y) + " " + std::to_string(positions[i].z);
                RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
                publisher_->publish(message);
            }

            // Get the label for the class name and its confidence.
            string label = format("%.2f", confidences[idx]);
            label = class_list_[class_ids[idx]] + ":" + label;
            // Draw class labels.
            draw_label(image, label, left, top);
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

    void draw_label(Mat& input_image, string label, int left, int top)
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

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
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

public:
    ObstDetector() : Node("ObstDetector") {
        cv::namedWindow(OPENCV_WINDOW, cv::WINDOW_AUTOSIZE);

        // rmw_qos_profile_t custom_qos = rmw_qos_profile_default;
                // sub_ = image_transport::create_subscription(this, "camera/image_raw",
                // std::bind(&ObstDetector::imageCallback, this, std::placeholders::_1), "raw", custom_qos);
        subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
                "camera/image_rect", 10, std::bind(&ObstDetector::imageCallback, this, std::placeholders::_1));
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

    ~ObstDetector()
    {
        cv::destroyWindow(OPENCV_WINDOW);
    }
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::cout << "Starting ObstDetector node" << std::endl;
    rclcpp::spin(std::make_shared<ObstDetector>());
    rclcpp::shutdown();
    return 0;
}