#ifndef FRAME_QUEUE_HPP
#define FRAME_QUEUE_HPP

#include <opencv2/opencv.hpp>
#include <condition_variable>
#include <mutex>
#include <queue>

// #include <iostream>

class FrameQueue
{
    public:
        FrameQueue();
        void push(const cv::Mat &frame);
        bool pop(cv::Mat &frame);
        void stop();
        bool isEmpty();
        bool stopped();

    private:

    std::queue<cv::Mat> Q_;  
    std::mutex mtx_;
    std::condition_variable cond_;
    bool stopped_ = false;

};


#endif //FRAME_QUEUE_HPP