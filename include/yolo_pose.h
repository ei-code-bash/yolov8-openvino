/*
 * @Author: ei-code-bash 3080152159@qq.com
 * @Date: 2025-07-26 13:47:31
 * @LastEditors: ei-code-bash 3080152159@qq.com
 * @LastEditTime: 2025-07-28 12:41:08
 * @FilePath: /yolopose-openvino/include/yolo_pose.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef YOLO_POSE_H
#define YOLO_POSE_H
#include<string>
#include<vector>
#include<openvino/openvino.hpp>
#include<opencv2/opencv.hpp>
#define DIS_MAX 0
typedef struct 
{
    int class_id;
    float confidence;
    cv::Rect box;
    std::vector<cv::Point3f> keypoints;
}detection;
class YoloPose
{
public:
YoloPose(const std::string& model_path,
        const std::vector<std::string>& class_names,
        const std::string& device = "CPU",
        float conf_threshold = 0.1f,
        float nms_threshold = 0.5f );
~YoloPose();
std::vector<detection>detect(cv::Mat &image);
private:
ov::Core core;
ov::CompiledModel compiled_model;
    
    // 模型属性
ov::Shape input_shape;
std::vector<std::string> class_names;
int num_classes;
int num_keypoints;

    // 阈值参数
float conf_threshold;
float nms_threshold;
//预处理和后处理函数的位置
std::pair<float,cv::Point2f>preprocess(cv::Mat& image,ov::Tensor& input_tensor);
std::vector<detection>postprocess(const ov::Tensor& output_tensor,float scale,const cv::Point2f&pad_xy,const cv::Size& original_shape);

};
#endif