/*** 
 * @Author: ei-code-bash 3080152159@qq.com
 * @Date: 2025-06-22 06:43:51
 * @LastEditors: ei-code-bash 3080152159@qq.com
 * @LastEditTime: 2025-06-29 11:19:51
 * @FilePath: /yolov8_openvino/include/yolo_detection.h
 * @Description: NEXTE VISION Group
 * @
 * @Copyright (c) 2025 by ${git_name_email}, All Rights Reserved. 
 */

#pragma once
#include <openvino/openvino.hpp>
#include<opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <iostream>
struct Detect
{
	cv::Rect box;//边界框
	int class_id;//类别数
	float confidence;//置信度
};
class Yolo_detection
{
public:
std::vector<std::string> class_names = 
	{
		"car",
		"armor1red",
		"armor2red",
		"armor3red",
		"armor4red",
		"armor5red",
		"armor1blue",
		"armor2blue",
		"armor3blue",
		"armor4blue",
		"armor5blue",
		"base",
		"ignore",
		"armor6red",
		"armor6blue"
	};
Yolo_detection(const std::string&model_path,const std::string&device_name,
			   float confThreshold = 0.45, float nmsThreshold = 0.50, 
			   int inpWidth = 1280, int inpHeight = 1280);
~Yolo_detection();
std::vector<Detect> detect( cv::Mat& frame);
// void initmodel(const std::string&model_path);
void drawPredictions(cv::Mat& frame, const std::vector<Detect>& detections);
private:
	float preprocess(const cv::Mat& frame, ov::Tensor& input_tensor);
	std::vector<Detect>postprocess(const ov::Tensor& output_tensor,float scale, const cv::Size& image_shape);
	// void drawPredictions(cv::Mat& frame, const std::vector<Detect>& detections);
	void initModel();
	// ov::Core core;
	ov::CompiledModel compiled_model;
	ov::InferRequest infer_request;
	std::string model_path;
	std::string device_name;
	ov::Tensor input_tensor;
	// ov::InferRequest infer_request;
	// ov::CompiledModel compiled_model;

	float confThreshold;
	float nmsThreshold;
	int inpWidth;
	int inpHeight;
	// std::vector<std::string> class_names = 
	// {
	// 	"car",
	// 	"armor1red",
	// 	"armor2red",
	// 	"armor3red",
	// 	"armor4red",
	// 	"armor5red",
	// 	"armor1blue",
	// 	"armor2blue",
	// 	"armor3blue",
	// 	"armor4blue",
	// 	"armor5blue",
	// 	"base",
	// 	"ignore",
	// 	"armor6red",
	// 	"armor6blue"
	// };
	};