#include <opencv2/opencv.hpp>
#include "videoWriter.hpp"
#include "videoReader.hpp"
#include "FrameQueue.hpp"
#include "logging.hpp"

#include <iostream>
#include <string>
#include <thread>

int main()
{
    std::string videoPath = "/project/resources/video_test.mp4";
    VideoReader reader(videoPath);

    if (!reader.isOpened())
        return -1;


    std::string outputPath = "/project/test_outputs/output.avi";
    VideoWriter writer(outputPath, reader.getFrameSize(), reader.getFps());

    if (!writer.isOpened())
        return -1;

    FrameQueue queue;
    
    std::thread reader_thread([&]() {
        cv::Mat frame;
        while (reader.readFrame(frame))
            queue.push(frame);  
        queue.stop();  
    });

    std::thread writer_thread([&]() {
        cv::Mat frame;
        while (queue.pop(frame))
            writer.writeFrame(frame);
        
    });

    reader_thread.join();
    writer_thread.join();

    reader.release();
    writer.release();
    logger::info("Resources released. Exiting application...");

    return 0;
}