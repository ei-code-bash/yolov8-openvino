/*
 * @Author: ei-code-bash 3080152159@qq.com
 * @Date: 2025-07-26 13:48:43
 * @LastEditors: ei-code-bash 3080152159@qq.com
 * @LastEditTime: 2025-07-28 12:57:20
 * @FilePath: /yolopose-openvino/src/utils.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include "utils.h"

void utils::draw_results(cv::Mat& image, const std::vector<detection>& results, const std::vector<std::string>& class_names) {
    for (const auto& res : results) {
        // 绘制边界框
        cv::rectangle(image, res.box, cv::Scalar(0, 255, 0), 2);

        // 绘制标签 (类别名 + 置信度)
        std::string label = class_names[res.class_id] + ": " + std::to_string(res.confidence).substr(0, 4);
        int baseLine;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine);
        cv::rectangle(image, cv::Point(res.box.x, res.box.y - label_size.height - baseLine), 
                      cv::Point(res.box.x + label_size.width, res.box.y), cv::Scalar(0, 255, 0), cv::FILLED);
        cv::putText(image, label, cv::Point(res.box.x, res.box.y - baseLine), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);
        
        // 绘制关键点
        for (const auto& kpt : res.keypoints) {
            if (kpt.z > 0.4) 
            { // 只绘制置信度高的关键点
                cv::circle(image, cv::Point(kpt.x, kpt.y-baseLine-label_size.height+80), 4, cv::Scalar(0, 0, 255), -1); // 红色圆点
            }
        }
    }
}
