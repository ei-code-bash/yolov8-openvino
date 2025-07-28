/*
 * @Author: ei-code-bash 3080152159@qq.com
 * @Date: 2025-07-26 13:47:41
 * @LastEditors: ei-code-bash 3080152159@qq.com
 * @LastEditTime: 2025-07-26 14:30:52
 * @FilePath: /yolopose-openvino/include/utils.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include "yolo_pose.h" // 

namespace utils {
    void draw_results(cv::Mat& image, const std::vector<detection>& results, const std::vector<std::string>& class_names);
}

#endif // UTILS_H
