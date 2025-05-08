#include "videoReader.hpp"
#include <iostream>
#include "logging.hpp"

VideoReader::VideoReader(const std::string &inputPath) : cap(inputPath)
{
    if (!cap.isOpened())
    {
        logger::error("Error opening video: %s", inputPath.c_str());
    }
    logger::info("Video successfully opened.");
}

bool VideoReader::readFrame(cv::Mat &frame)
{
    bool success = cap.read(frame);
    if(!success)
        logger::warning("Failed to read frame from video stream.");
    return success;
}


bool VideoReader::isOpened() const
{
    return cap.isOpened();
}

int VideoReader::getFourcc()
{
    return cap.get(cv::CAP_PROP_FOURCC);
}

double VideoReader::getFps()
{
    return cap.get(cv::CAP_PROP_FPS);
}

cv::Size VideoReader::getFrameSize()
{
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    return cv::Size(frameWidth, frameHeight);
}

void VideoReader::release()
{
    logger::info("Releasing video reader...");
    cap.release();
}