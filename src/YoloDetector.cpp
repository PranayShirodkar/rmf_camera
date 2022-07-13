#include <memory>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <filesystem>
#include <memory>

// OpenCV includes
#include <cv_bridge/cv_bridge.h>

// Project includes
#include <rmf_camera/YoloDetector.hpp>
#include <rmf_camera/BoundingBox3D.hpp>
#include <rmf_obstacle_msgs/msg/obstacles.hpp>

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

YoloDetector::YoloDetector() {
    cv::namedWindow(OPENCV_WINDOW, cv::WINDOW_AUTOSIZE);

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

YoloDetector::Obstacles YoloDetector::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
    // printf("Image received \tSize: %dx%d - Timestamp: %u.%u sec - Encoding: %s\n",
    //                 msg->width, msg->height,
    //                 msg->header.stamp.sec,msg->header.stamp.nanosec, msg->encoding.c_str());

    // bridge from ROS image type to OpenCV image type
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
    Mat original_image = cv_ptr->image;
    // Mat original_image = imread("/home/osrc/Pictures/Screenshots/empty_world/test.png");

    // format image, forward propagate and post process
    Mat image = format_yolov5(original_image);
    vector<Mat> detections = detect(image);
    Obstacles rmf_obstacles = post_process(original_image, image, detections);

    for (auto &obstacle : rmf_obstacles.obstacles)
    {
      obstacle.header = msg->header;
    }

    // display
    imshow(OPENCV_WINDOW, image);
    // imwrite("/home/osrc/Pictures/Screenshots/empty_world/1.jpg", image);
    waitKey(3);
    return rmf_obstacles;
}

Mat YoloDetector::format_yolov5(const Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);

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

YoloDetector::Obstacles YoloDetector::post_process(const Mat &original_image, Mat &image, vector<Mat> &detections)
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

    // Perform Non-Maximum Suppression.
    vector<int> indices; // will contain indices of final bounding boxes
    NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

    // use indices vector to get the final vectors
    vector<int> final_class_ids;
    vector<float> final_confidences;
    vector<Rect> final_boxes;
    vector<Point> final_centroids; // image coordinates, (int, int)
    vector<Point2d> final_obstacles; // camera coordinates, (double, double)
    for (auto i : indices)
    {
        final_class_ids.push_back(class_ids[i]);
        final_confidences.push_back(confidences[i]);
        final_boxes.push_back(boxes[i]);
        final_centroids.push_back(centroids[i]);
        Point2d obstacle = img_coord_to_cam_coord(centroids[i], original_image);
        final_obstacles.push_back(obstacle);
    }

    // draw to image
    drawing(original_image, image, final_class_ids, final_confidences, final_boxes, final_centroids);

    // generate rmf_obstacles
    auto obstacles_msg = cam_coord_to_rmf_obstacle(final_class_ids, final_confidences, final_obstacles);

    return obstacles_msg;
}

Point2d YoloDetector::img_coord_to_cam_coord(const Point &centroid, const Mat &original_image)
{
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

YoloDetector::Obstacles YoloDetector::cam_coord_to_rmf_obstacle(const vector<int> &final_class_ids, const vector<float> &final_confidences, const vector<Point2d> &final_obstacles)
{
    auto rmf_obstacles_msg = Obstacles();

    // prepare obstacle_msg objects and add to rmf_obstacles_msg
    rmf_obstacles_msg.obstacles.reserve(final_obstacles.size());
    for (size_t i = 0; i < final_obstacles.size(); i++) {
        auto obstacle = rmf_obstacle_msgs::msg::Obstacle();
        // auto obstacle2 = rmf_obstacle_msgs::build<rmf_obstacle_msgs::msg::Obstacle>()
        // .header()
        // .id()
        // .source()
        // .level_name()
        // .classification()
        // .bbox()
        // .data_resolution()
        // .data()
        // .lifetime()
        // .action();
        obstacle.id = final_class_ids[i];
        obstacle.classification = class_list_[final_class_ids[i]];
        obstacle.bbox.center.position.x = final_obstacles[i].x;
        obstacle.bbox.center.position.y = final_obstacles[i].y;
        obstacle.bbox.center.position.z = 1.0;
        // obstacle.bbox.center.orientation.x =
        // obstacle.bbox.center.orientation.y =
        // obstacle.bbox.center.orientation.z =
        // obstacle.bbox.center.orientation.w =
        obstacle.bbox.size.x = 2.0;
        obstacle.bbox.size.y = 2.0;
        obstacle.bbox.size.z = 2.0;

        rmf_obstacles_msg.obstacles.push_back(obstacle);
    }

    auto message = std_msgs::msg::String();
    for (size_t i = 0; i < final_obstacles.size(); i++) {
        string class_label = class_list_[final_class_ids[i]];
        BoundingBox3D bb = {
            class_label,  // classification
            static_cast<int>(i),  // id
            Point3d(final_obstacles[i].x, final_obstacles[i].y, 0.0),  // position
            Vec3d(1.0, 1.0, 2.0),  // size
        };
        message.data = bb.toString();
        // RCLCPP_INFO(this->get_logger(), "Publishing: '%s'", message.data.c_str());
        // publisher_->publish(message);
    }

    return rmf_obstacles_msg;
}

void YoloDetector::drawing(const Mat &original_image, Mat &image, const vector<int> &final_class_ids, const vector<float> &final_confidences, const vector<Rect> &final_boxes, const vector<Point> &final_centroids)
{
    for (size_t i = 0; i < final_class_ids.size(); i++)
    {
        Rect box = final_boxes[i];
        int left = box.x;
        int top = box.y;
        int width = box.width;
        int height = box.height;
        // Draw bounding box.
        rectangle(image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);
        // Draw centroid.
        circle(image, final_centroids[i], 2, CV_RGB(255,0,0), -1);
        printf("Pixel: (%d, %d)\n", final_centroids[i].x, final_centroids[i].y);

        // Get the label for the class name and its confidence.
        string label = format("%.2f", final_confidences[i]);
        label = class_list_[final_class_ids[i]] + ":" + label;
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

void YoloDetector::draw_label(Mat &input_image, string label, int left, int top)
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

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    std::cout << "Starting YoloDetector node" << std::endl;
    // rclcpp::spin(std::make_shared<YoloDetector>());
    rclcpp::shutdown();
    return 0;
}