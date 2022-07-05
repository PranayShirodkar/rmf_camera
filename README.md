# rmf camera

launch a gazebo world that has a camera model which is publishing a ros topic

set ros topic name in YoloDetector.cpp

subscribe to ros topic:
> ros2 run rmf_camera YoloDetector

## Hard Coded paths

For now, the ros topic name, path to yolov5s.onnx and path to coco.names are hard coded in cpp file.

## LibTorch

### To get LibTorch:
```bash
mkdir rmf_camera/lib
cd rmf_camera/lib
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.11.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-1.11.0+cpu.zip
```

Or get it from https://pytorch.org/get-started/locally/ -> Choose Stable, Linux, LibTorch, C++, CPU, cxx11 ABI

### To remove LibTorch:
Remove the following lines from rmf_camera/CMakeLists.txt:
```cmake
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch)
set(CMAKE_INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(YoloDetector "${TORCH_LIBRARIES}")
set_property(TARGET YoloDetector PROPERTY CXX_STANDARD 14)

  "Torch"
```
Also remove rmf_camera/lib/libtorch

### To get YOLOv5s ONNX format:

ONNX: Open Neural Network Exchange
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
python3 export.py --weights yolov5s.pt --include torchscript onnx
```
will generate yolov5s.pt, yolov5s.torchscript, yolov5s.onnx

### To get coco.names file:

coco.names is a file of all the possible classes/labels for an object.

https://github.com/pjreddie/darknet/blob/master/data/coco.names