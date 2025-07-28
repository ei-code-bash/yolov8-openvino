cmake_minimum_required(VERSION 3.22.1)
project(
    Yolo_detection
    LANGUAGES CXX
    DESCRIPTION "NEXTE Vision Group"
    VERSION 0.1.0

)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui dnn)
add_executable(Yolo_detection src/main.cpp src/yolo_pose.cpp src/utils.cpp)
target_include_directories(Yolo_detection PUBLIC include)
target_compile_features(Yolo_detection PUBLIC cxx_std_17)
target_include_directories(Yolo_detection PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_directories(Yolo_detection PUBLIC ${OpenCV_LIBS})
find_package(OpenVINO REQUIRED)
target_link_libraries(Yolo_detection 
PUBLIC 
opencv_core
opencv_imgproc
opencv_highgui
opencv_dnn
opencv_videoio
openvino::runtime)
