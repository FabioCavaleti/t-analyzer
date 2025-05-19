#ifndef FRAME_PROCESSOR_HPP
#define FRAME_PROCESSOR_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

class FrameProcessor
{
    public:

        void write_frame_to_shm(const cv::Mat &frame, const std::string &shm_name);
        void process(cv::Mat &frame, const std::string &shm_name, std::string frame_id);
        float getThreshold();
        void setThreshold(float value);
        bool modelRegistered();

    private:
    void register_inference_model(const cv::Mat &frame);
    void run_inference(std::string frame_id);
    void draw(cv::Mat &frame, const std::string &result_json_path);
    float threshold_ = 0.1f;
    bool model_registered_ = false;

};


#endif