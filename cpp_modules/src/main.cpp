#include <opencv2/opencv.hpp>
#include "videoWriter.hpp"
#include "videoReader.hpp"
#include "FrameQueue.hpp"
#include "FrameProcessor.hpp"
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

    FrameProcessor processor;

    FrameQueue raw_frame_queue;
    FrameQueue processed_frame_queue;

    
    std::thread reader_thread([&]() {
        cv::Mat frame;
        while (reader.readFrame(frame))
        {
            raw_frame_queue.push(frame);  
        }
        
    });

    std::thread frame_processor_thread([&]() {

        cv::Mat frame;
        int frame_id = 0;
        std::string shm_name;
        while(raw_frame_queue.pop(frame))
        {
            shm_name = "frame_" + std::to_string(frame_id);
            processor.process(frame, shm_name, std::to_string(frame_id));
            frame_id++;
            processed_frame_queue.push(frame);
        }

    });
    


    std::thread writer_thread([&]() {
        cv::Mat frame;
        while (processed_frame_queue.pop(frame))
        {
            writer.writeFrame(frame);
        }
        
    });

    
    reader_thread.join();
    writer_thread.join();
    
    reader.release();
    writer.release();
    logger::info("Resources released. Exiting application...");

    return 0;
}