/*
 * @Author: ei-code-bash 3080152159@qq.com
 * @Date: 2025-07-26 13:48:52
 * @LastEditors: ei-code-bash 3080152159@qq.com
 * @LastEditTime: 2025-07-28 11:29:07
 * @FilePath: /yolopose-openvino/src/main.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <algorithm> // 用于 std::transform
#include <cctype>    // 用于 std::tolower

#include "yolo_pose.h"
#include "utils.h"

#include <iostream>
#include <chrono>
#include <string>     // 需要包含 <string> 以使用 std::stof
#include "yolo_pose.h"
#include "utils.h"



int main(int argc, char* argv[]) {
    // 用法提示保持不变
    if (argc != 3) {
        std::cerr << "用法: " << argv[0] << " <path_to_model.xml> <path_to_video.mp4>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];
    
    // 类别名称
    std::vector<std::string> class_names = {
        "B1", "B2", "B3", "B4", "B7", "R1", "R2", "R3", "R4", "R7", "base", "ignore"
    };

    try {
        // --- 1. 初始化 YoloPose 检测器 (完全不变) ---
        YoloPose detector(model_path, class_names,"CPU",0.1f, 0.5f);

        // --- 2. 打开视频文件或摄像头 ---
        // 如果 video_path 是 "0", "1" 等数字，会尝试打开摄像头
        cv::VideoCapture cap;
        if (video_path.size() == 1 && std::isdigit(video_path[0])) {
            cap.open(std::stoi(video_path));
        } else {
            cap.open(video_path);
        }
        
        if (!cap.isOpened()) {
            std::cerr << "错误: 无法打开视频源 " << video_path << std::endl;
            return -1;
        }
        
        std::cout << "正在处理视频... 按 'q' 键退出。" << std::endl;
        
        cv::Mat frame;
        // --- 3. 逐帧处理视频 ---
        while (cap.read(frame)) {
            if (frame.empty()) {
                std::cout << "视频流结束。" << std::endl;
                break;
            }

            // 记录开始时间
            auto start = std::chrono::high_resolution_clock::now();

            // 运行检测
            std::vector<detection> results = detector.detect(frame);

            // 记录结束时间
            auto end = std::chrono::high_resolution_clock::now();
            
            // 计算耗时和 FPS
            std::chrono::duration<double> diff = end - start;
            double fps = 1.0 / diff.count();

            // 可视化检测结果
            utils::draw_results(frame, results, class_names);

            // 在帧上绘制 FPS 信息
            std::string fps_text = "FPS: " + std::to_string(fps).substr(0, 5);
            cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 2);
            
            // (已移除) 不再需要写入视频文件
            // video_writer.write(frame);
            
            // 实时显示结果帧
            cv::imshow("YOLOv8-Pose C++ Real-time Demo", frame);

            // 按 'q' 键或 ESC 键退出循环
            char key = (char)cv::waitKey(1);
            if (key == 'q' || key == 27) {
                break;
            }
        }

        // --- 4. 释放资源 ---
        std::cout << "处理完成。" << std::endl;
        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "发生异常: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
