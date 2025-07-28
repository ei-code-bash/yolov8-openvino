#include<iostream>
#include"yolo_pose.h"
YoloPose::YoloPose(const std::string& model_path, const std::vector<std::string>& class_names, const std::string& device, float conf_threshold, float nms_threshold)
    : class_names(class_names), conf_threshold(conf_threshold), nms_threshold(nms_threshold)
        {
            std::cerr<<"------正在输入模型呢------"<<model_path<<std::endl;
            auto model=core.read_model(model_path);
            this->num_classes=12;
            this->num_keypoints=4;
            std::cerr<<"编译模型"<<std::endl;
            this->compiled_model=core.compile_model(model,device);
            this->input_shape=compiled_model.input().get_shape();
        }
std::pair<float,cv::Point2f>YoloPose::preprocess(cv::Mat& image,ov::Tensor& input_tensor)
{
     long input_h = input_shape[2];
    long input_w = input_shape[3];

    // --- Letterbox 缩放和填充 ---
    float scale = std::min(static_cast<float>(input_w) / image.cols, static_cast<float>(input_h) / image.rows);
    int scaled_w = static_cast<int>(image.cols * scale);
    int scaled_h = static_cast<int>(image.rows * scale);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(scaled_w, scaled_h));

    cv::Mat padded_image(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    cv::Point2f pad_xy((input_w - scaled_w) / 2.0f, (input_h - scaled_h) / 2.0f);
    resized_image.copyTo(padded_image(cv::Rect(pad_xy.x, pad_xy.y, scaled_w, scaled_h)));

    // --- 转换为 NCHW 格式的 float 张量 ---
    float* input_data = input_tensor.data<float>();
    for (int h = 0; h < input_h; ++h) {
        for (int w = 0; w < input_w; ++w) {
            for (int c = 0; c < 3; ++c) {
                // BGR -> RGB, uchar -> float, HWC -> CHW
                input_data[c * input_h * input_w + h * input_w + w] = padded_image.at<cv::Vec3b>(h, w)[2 - c] / 255.0f;
            }
        }
    }
    return {scale, pad_xy};
}
YoloPose::~YoloPose()
{

};
std::vector<detection> YoloPose::postprocess(const ov::Tensor& output_tensor, float scale, const cv::Point2f& pad_xy, const cv::Size& original_shape) 
{
    const float* output_data = output_tensor.data<const float>();
    const ov::Shape& output_shape = output_tensor.get_shape();
    
    int num_proposals = output_shape[2]; // 33600
    int channels = output_shape[1]; // 28 (4 bbox + 12 classes + 12 kpts)

    // --- 数据转置以便于处理: [1, 28, 33600] -> [33600, 28] ---
    cv::Mat transposed_data(num_proposals, channels, CV_32F);
    for (int i = 0; i < num_proposals; ++i) 
    {
        for (int j = 0; j < channels; ++j) 
        {
            transposed_data.at<float>(i, j) = output_data[j * num_proposals + i];
        }
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::vector<cv::Point3f>> all_keypoints;

    // --- 解析每一个预测 ---
    for (int i = 0; i < num_proposals; ++i) {
        float* row_ptr = transposed_data.ptr<float>(i);
        cv::Mat class_scores(1, this->num_classes, CV_32F, row_ptr + 4);
        cv::Point max_class_loc;
        double max_score;
        cv::minMaxLoc(class_scores, 0, &max_score, 0, &max_class_loc);

        if (max_score > this->conf_threshold) {
            confidences.push_back(static_cast<float>(max_score));
            class_ids.push_back(max_class_loc.x);

            float cx = row_ptr[0], cy = row_ptr[1], w = row_ptr[2], h = row_ptr[3];
            boxes.emplace_back(cx - w / 2, cy - h / 2, w, h);

            std::vector<cv::Point3f> kpts;
            float* kpts_ptr = row_ptr + 4 + this->num_classes;
            for (int k = 0; k < this->num_keypoints; ++k) {
                kpts.emplace_back(kpts_ptr[k * 3], kpts_ptr[k * 3 + 1], kpts_ptr[k * 3 + 2]);
            }
            all_keypoints.push_back(kpts);
        }
    }

    // --- 非极大值抑制 (NMS) ---
    std::vector<int> nms_indices;
    cv::dnn::NMSBoxes(boxes, confidences, this->conf_threshold, this->nms_threshold, nms_indices);

    // --- 生成最终结果 ---
    std::vector<detection> final_results;
    for (int index : nms_indices) {
        detection res;
        res.class_id = class_ids[index];
        res.confidence = confidences[index];
        
        // 将坐标还原到原始图像尺寸
        cv::Rect box = boxes[index];
        res.box.x = static_cast<int>((box.x - pad_xy.x) / scale);
        res.box.y = static_cast<int>((box.y - pad_xy.y) / scale);
        res.box.width = static_cast<int>(box.width / scale);
        res.box.height = static_cast<int>(box.height / scale);
        // float middle_x=res.box.width/2;
        // float middle_y=res.box.height/2;
        // cv::Point2f p(middle_x,middle_y);
        // 裁剪以确保边界框在图像内
        res.box &= cv::Rect(0, 0, original_shape.width, original_shape.height);

        for (const auto& kpt : all_keypoints[index]) 
        {
            
            // double distance=sqrt((middle_x-kpt.x)*(middle_x-kpt.x)+(middle_y-kpt.y)*(middle_y-kpt.y));
            // if(distance>=DIS_MAX)
            // {
            //     kpt.x=middle_x;
            //     kpt.y=middle_y;
            // }
            res.keypoints.emplace_back((kpt.x - pad_xy.x) / scale, (kpt.y - pad_xy.y) / scale, kpt.z);
        }
        
        final_results.push_back(res);
    }
    return final_results;
}

std::vector<detection> YoloPose::detect(cv::Mat& image) {
    // --- 创建推理请求和输入张量 ---
    ov::InferRequest infer_request = this->compiled_model.create_infer_request();
    ov::Tensor input_tensor = infer_request.get_input_tensor();

    // --- 图像预处理 ---
    auto [scale, pad_xy] = this->preprocess(image, input_tensor);

    // --- 执行推理 ---
    infer_request.infer();
    
    // --- 获取输出并进行后处理 ---
    const ov::Tensor output_tensor = infer_request.get_output_tensor();
    return this->postprocess(output_tensor, scale, pad_xy, image.size());
}