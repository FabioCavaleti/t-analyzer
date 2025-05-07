#include "videoWriter.hpp"
#include <iostream>

VideoWriter::VideoWriter(const std::string &outputPath, cv::Size frameSize, double fps, int codec) : writer(outputPath, codec, fps, frameSize)
{
    
}

bool VideoWriter::isOpened() const
{
    return writer.isOpened();
}

void VideoWriter::writeFrame(const cv::Mat &frame)
{
    if (frame.empty())
        return;

    writer.write(frame);

}

void VideoWriter::release()
{
    writer.release();
}