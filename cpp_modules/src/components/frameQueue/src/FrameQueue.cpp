#include "FrameQueue.hpp"
#include "logging.hpp"

#include <opencv2/opencv.hpp>

FrameQueue::FrameQueue()
{
    logger::info("Frame Queue successfully initialized.");
}

bool FrameQueue::isEmpty()
{
    std::lock_guard<std::mutex> lock(mtx_);
    return Q_.empty();
}

bool FrameQueue::stopped()
{
    return stopped_;
}

void FrameQueue::push(const cv::Mat &frame)
{
    logger::info("Adding frame to queue...");
    {
        std::lock_guard<std::mutex> lg(mtx_);
        Q_.push(frame);
    }
    cond_.notify_one();
}


bool FrameQueue::pop(cv::Mat &frame)
{
    logger::info("Poping frame from queue...");
    std::unique_lock<std::mutex> lock(mtx_);
    cond_.wait(lock, [&]{
        return !Q_.empty() || stopped_;
    });

    if(Q_.empty())
    {
        logger::info("Tried to pop queue but empty.");
        return false;
    }
    frame = Q_.front();
    Q_.pop();
    return true;   
}

void FrameQueue::stop()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stopped_ = true;
    }
    cond_.notify_one();
}


