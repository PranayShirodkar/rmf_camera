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
#include <opencv2/opencv.hpp>
#include <fstream>

// OpenCV Window Name
const std::string OPENCV_WINDOW = "Image window";

// DNN
// Namespaces.
using namespace cv;
using namespace std;
using namespace cv::dnn;

// Constants.
const float INPUT_WIDTH = 640.0;
const float INPUT_HEIGHT = 640.0;
const float SCORE_THRESHOLD = 0.5;
const float NMS_THRESHOLD = 0.45;
const float CONFIDENCE_THRESHOLD = 0.45;

// Text parameters.
const float FONT_SCALE = 0.7;
const int FONT_FACE = FONT_HERSHEY_SIMPLEX;
const int THICKNESS = 1;

// Colors.
Scalar BLACK = Scalar(0,0,0);
Scalar BLUE = Scalar(255, 178, 50);
Scalar YELLOW = Scalar(0, 255, 255);
Scalar RED = Scalar(0,0,255);

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

vector<Mat> pre_process(Mat &input_image, Net &net)
{
    // Convert to blob.
    Mat blob;
    blobFromImage(input_image, blob, 1./255., Size(INPUT_WIDTH, INPUT_HEIGHT), Scalar(), true, false);

    net.setInput(blob);

    // Forward propagate.
    vector<Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());

    return outputs;
}

Mat post_process(Mat &input_image, vector<Mat> &outputs, const vector<string> &class_name)
{
    // Initialize vectors to hold respective outputs while unwrapping     detections.
    vector<int> class_ids;
    vector<float> confidences;
    vector<Rect> boxes;
    // Resizing factor.
    float x_factor = input_image.cols / INPUT_WIDTH;
    float y_factor = input_image.rows / INPUT_HEIGHT;
    float *data = (float *)outputs[0].data;
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
            Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
            // Perform minMaxLoc and acquire the index of best class  score.
            Point class_id;
            double max_class_score;
            minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
            // Continue if the class score is above the threshold.
            if (max_class_score > SCORE_THRESHOLD)
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
            }
        }
        // Jump to the next row.
        data += 85;
    }

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
        rectangle(input_image, Point(left, top), Point(left + width, top + height), BLUE, 3*THICKNESS);
        // Get the label for the class name and its confidence.
        string label = format("%.2f", confidences[idx]);
        label = class_name[class_ids[idx]] + ":" + label;
        // Draw class labels.
        draw_label(input_image, label, left, top);
    }
    return input_image;
}

vector<string> class_list;
// Load model.
Net net;

void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
  cv_bridge::CvImagePtr cv_ptr;

  cv_ptr = cv_bridge::toCvCopy(msg,msg->encoding);
  // if (cv_ptr->image.rows > 60 && cv_ptr->image.cols > 60)
  //    cv::circle(cv_ptr->image, cv::Point(50, 50), 10, CV_RGB(255,0,0));

  // printf("Image received \tSize: %dx%d - Timestamp: %u.%u sec - Encoding: %s\n",
  //               msg->width, msg->height,
  //               msg->header.stamp.sec,msg->header.stamp.nanosec, msg->encoding.c_str());

  // cv::imshow(OPENCV_WINDOW, cv_ptr->image);
  // cv::waitKey(3);
  
  // Load image.
  Mat frame;
  // frame = imread("/home/osrc/dev_ws/src/rmf_camera/src/groupscene.png");
  frame = cv_ptr->image;

  vector<Mat> detections;     // Process the image.
  detections = pre_process(frame, net);
  Mat frame2 = frame.clone();
  Mat img = post_process(frame2, detections, class_list);
  // Put efficiency information.
  // The function getPerfProfile returns the overall time for     inference(t) and the timings for each of the layers(in layersTimes).
  vector<double> layersTimes;
  double freq = getTickFrequency() / 1000;
  double t = net.getPerfProfile(layersTimes) / freq;
  string label = format("Inference time : %.2f ms", t);
  putText(img, label, Point(20, 40), FONT_FACE, FONT_SCALE, RED);
  imshow(OPENCV_WINDOW, img);
  waitKey(3);
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  std::shared_ptr<rclcpp::Node> node = rclcpp::Node::make_shared("image_listener");

  cv::namedWindow(OPENCV_WINDOW, cv::WINDOW_AUTOSIZE);

  net = readNet("/home/osrc/dev_ws/src/rmf_camera/src/yolov5s.onnx");
  ifstream ifs("/home/osrc/dev_ws/src/rmf_camera/src/coco.names");
  string line;
  while (getline(ifs, line))
  {
      class_list.push_back(line);
  }

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
