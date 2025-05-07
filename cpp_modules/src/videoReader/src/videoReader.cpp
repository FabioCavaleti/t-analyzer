#include "videoReader.hpp"
#include <iostream>

VideoReader::VideoReader(const std::string &inputPath) : cap(inputPath)
{
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video" << std::endl;
    }
}

bool VideoReader::readFrame(cv::Mat &frame)
{
    return cap.read(frame);
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
    cap.release();
}