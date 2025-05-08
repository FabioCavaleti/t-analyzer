#include "videoWriter.hpp"
#include <iostream>
#include "logging.hpp"

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
    {
        logger::warning("Attempted to write empty frame. Skipping...");
        return;
    }

    writer.write(frame);

}

void VideoWriter::release()
{
    logger::info("Releasing video writer...");
    writer.release();
}