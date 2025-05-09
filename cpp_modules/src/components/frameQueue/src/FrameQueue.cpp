#include "FrameQueue.hpp"
#include "logging.hpp"

#include <opencv2/opencv.hpp>

FrameQueue::FrameQueue(size_t maxSize):max_size_(maxSize)
{
    logger::info("Frame Queue successfully initialized with max size: %zu.", max_size_);
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
    {
        std::unique_lock<std::mutex> lock(mtx_);
        full_cond_.wait(lock, [&](){
            return Q_.size() < max_size_ || stopped_;
        });

        if (stopped_)
            return;
        Q_.push(frame.clone());
    }
    cond_.notify_one();
}


bool FrameQueue::pop(cv::Mat &frame)
{
    std::unique_lock<std::mutex> lock(mtx_);
    cond_.wait(lock, [&]{
        return !Q_.empty() || stopped_;
    });

    if (Q_.empty())
    {
        if (stopped_)
            logger::info("Queue stopped and empty — no more frames to consume.");
        else
            logger::warning("Unexpected empty queue without stop signal.");
        return false;
    }
    frame = Q_.front();
    Q_.pop();
    full_cond_.notify_one();
    return true;   
}

void FrameQueue::stop()
{
    {
        std::lock_guard<std::mutex> lock(mtx_);
        stopped_ = true;
    }
    cond_.notify_one();
    full_cond_.notify_one();
}


