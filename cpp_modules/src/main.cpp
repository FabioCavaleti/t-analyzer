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

    logger::info("Reader fps: %f", reader.getFps());

    if (!reader.isOpened())
        return -1;


    std::string outputPath = "/project/outputs/output.avi";
    VideoWriter writer(outputPath, reader.getFrameSize(), reader.getFps());

    if (!writer.isOpened())
        return -1;

    FrameQueue queue;

    int cnt_reader = 0;
    int cnt_writer = 0;
    
    std::thread reader_thread([&]() {
        cv::Mat frame;
        while (reader.readFrame(frame))
        {
            queue.push(frame);  
            cnt_reader++;
        }
        
        queue.stop();  
    });

    std::thread writer_thread([&]() {
        cv::Mat frame;
        while (queue.pop(frame))
        {
            writer.writeFrame(frame);
            cnt_writer++;
        }
        
    });

    
    reader_thread.join();
    writer_thread.join();
    
    logger::info("cnt value: %d", cnt_reader);
    logger::info("cnt value: %d", cnt_writer);
    reader.release();
    writer.release();
    logger::info("Resources released. Exiting application...");

    return 0;
}