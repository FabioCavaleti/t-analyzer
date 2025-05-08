#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "videoWriter.hpp"
#include "videoReader.hpp"
#include "logging.hpp"

int main()
{
    std::string videoPath = "/project/resources/video_test.mp4";
    logger::info("Initializing video reader with path: %s...", videoPath.c_str());
    VideoReader reader(videoPath);

    if (!reader.isOpened())
    {
        logger::error("Failed to open video reader.");
        return -1;
    }
    logger::info("Video reader successfully initialized.");


    std::string outputPath = "/project/test_outputs/output.avi";
    logger::info("Initializing video writer with path: %s...", outputPath.c_str());
    VideoWriter writer(outputPath, reader.getFrameSize(), reader.getFps());

    if (!writer.isOpened())
    {
        logger::error("Failed to open video writer");
        return -1;
    }

    logger::info("Video writer successfully initialized.");

    cv::Mat frame;
    int cnt = 0;
    
    logger::info("Processing video...");
    while (reader.readFrame(frame))
    {

        writer.writeFrame(frame);

        cnt++;
    }

    reader.release();
    writer.release();

    logger::info("Resources released. Exiting application...");


    
    return 0;
}