/*** 
 * @Author: ei-code-bash 3080152159@qq.com
 * @Date: 2025-06-22 06:44:20
 * @LastEditors: ei-code-bash 3080152159@qq.com
 * @LastEditTime: 2025-06-29 18:44:00
 * @FilePath: /yolov8_openvino/src/main.cpp
 * @Description: 
 * @
 * @Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
 */


# include"yolo_detection.h"
#include<map>
int main() 
{
    try {
        const std::string model_path = "/home/ei/codes/yolov8_openvino/model/01_openvino_model/01.xml"; // 修改为你的模型路径
        const std::string video_path = "/home/ei/codes/yolov8_openvino/model/test.mp4";
          // 修改为你的视频路径，或用 "0" 打开摄像头
        auto model= ov::Core().read_model(model_path);
        if (model == nullptr) {
            std::cerr << "Error: Model not found at " << model_path << std::endl;
            return -1;
        }
        
        // 1. 实例化检测器
        Yolo_detection detector(model_path, "AUTO", 0.5f, 0.45f);

        // 2. 打开视频源
        cv::VideoCapture cap(video_path);
        // 若要使用摄像头，请取消下面一行的注释
        // cv::VideoCapture cap(0); 
        if (!cap.isOpened()) {
            std::cerr << "Error: Cannot open video source." << std::endl;
            return -1;
        }

        cv::Mat frame;
        while (cap.read(frame)) {
            if (frame.empty()) {
                break;
            }

            // 计时开始
            auto start = std::chrono::high_resolution_clock::now();

            // 3. 执行检测
            std::vector<Detect> detections = detector.detect(frame);

            // 计时结束
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            double fps = 1000.0 / elapsed.count();

            // 4. 在图像上绘制结果
            detector.drawPredictions(frame, detections);
            
            // 显示FPS
            cv::putText(frame, cv::format("FPS: %.2f", fps), cv::Point(10, 30), 
                        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);

            // 5. 显示结果
            cv::imshow("YOLOv8 OpenVINO Detections", frame);

            // 按 'q' 键退出
            if (cv::waitKey(1) == 'q') {
                break;
            }
        }
        
        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
