#include "yolo_detection.h"
#include <iostream>
#include <string>
#include <time.h>

std::vector<std::string> class_names = {
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

Yolo_detection::Yolo_detection(const std::string& model_path, const std::string& device_name, 
                              float conf_threshold, float nms_threshold, int inpWidth, int inpHeight)
{
    this->model_path = model_path;
    this->inpWidth = 1280;
    this->inpHeight = 1280;
    this->confThreshold = conf_threshold;
    this->nmsThreshold = nms_threshold;
    this->class_names = class_names;
    this->initModel();
}

void Yolo_detection::initModel()
{
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model(this->model_path);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
 
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::RGB);
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255, 255, 255});
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    this->compiled_model = core.compile_model(model, "AUTO");
    this->infer_request = compiled_model.create_infer_request();
}

std::vector<Detect> Yolo_detection::detect(cv::Mat& frame)
{
    // 1.预处理
    ov::Tensor input_tensor = infer_request.get_input_tensor();
    float scale = preprocess(frame, input_tensor);

    // 2.推理
    infer_request.infer();

    // 3.后处理
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    std::vector<Detect> detections = postprocess(output_tensor, scale, frame.size());

    // 4.绘制预测结果
    drawPredictions(frame, detections);

    return detections;
}

float Yolo_detection::preprocess(const cv::Mat& image, ov::Tensor& input_tensor) 
{
    // Letterbox 缩放：保持宽高比，用灰色填充空白区域
    int original_w = image.cols;
    int original_h = image.rows;
    float scale = std::min(static_cast<float>(inpWidth) / original_w, static_cast<float>(inpHeight) / original_h);
    
    int scaled_w = static_cast<int>(original_w * scale);
    int scaled_h = static_cast<int>(original_h * scale);
    
    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(scaled_w, scaled_h));
    
    cv::Mat padded_image = cv::Mat(inpHeight, inpWidth, CV_8UC3, cv::Scalar(114, 114, 114));
    resized_image.copyTo(padded_image(cv::Rect(0, 0, scaled_w, scaled_h)));

    input_tensor = ov::Tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), padded_image.data);
    
    return scale;
}

Yolo_detection::~Yolo_detection()
{
    // 析构函数可以用于释放资源
}

std::vector<Detect> Yolo_detection::postprocess(const ov::Tensor& output_tensor, float scale, const cv::Size& image_shape)
{

    const float* data = output_tensor.data<const float>();

    cv::Mat output_buffer(output_tensor.get_shape()[2], output_tensor.get_shape()[1], CV_32F, (float*)data);
    //output_buffer = output_buffer.t(); // 转置为 [8400, 19]

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    for (int i = 0; i < output_buffer.rows; i++) {
        cv::Mat classes_scores = output_buffer.row(i).colRange(4, 19);
        
        cv::Point class_id_point;
        double max_conf;
        cv::minMaxLoc(classes_scores, 0, &max_conf, 0, &class_id_point);

        if (max_conf > this->confThreshold) {
            confidences.push_back(static_cast<float>(max_conf));
            class_ids.push_back(class_id_point.x);
            std::cerr<<"Class ID"<<class_id_point.x<<std::endl;

            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);
            
            int left = static_cast<int>((cx - w / 2) / scale);
            int top = static_cast<int>((cy - h / 2) / scale);
            int width = static_cast<int>(w / scale);
            int height = static_cast<int>(h / scale);
            //确保坐标在图像范围之间
            left=std::max(0,left);
            top=std::max(0,top);
            width=std::min(width,image_shape.width -left);
            height=std::min(height,image_shape.height-top);

            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }

    // 执行非极大值抑制 (NMS)
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->confThreshold, this->nmsThreshold, nms_indices);

    std::vector<Detect> detections;
    for (int index : nms_indices) 
    {
        Detect det;
        det.box = boxes[index];
        det.confidence = confidences[index];
        det.class_id = class_ids[index];
        detections.push_back(det);
    }
    
    return detections;
}

void Yolo_detection::drawPredictions(cv::Mat& image, const std::vector<Detect>& detections) 
{
    for (const auto& det : detections) 
    {
        cv::rectangle(image, det.box, cv::Scalar(0, 255, 0), 2);
        
        std::string label = cv::format("%s: %.2f", class_names[det.class_id].c_str(), det.confidence);
        
        int baseline;
        cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.7, 2, &baseline);
        
        cv::rectangle(image, 
                    cv::Point(det.box.x, det.box.y - label_size.height - 10), 
                    cv::Point(det.box.x + label_size.width, det.box.y), 
                    cv::Scalar(0, 255, 0), 
                    cv::FILLED);
        cv::putText(image, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 0), 2);
    }
}