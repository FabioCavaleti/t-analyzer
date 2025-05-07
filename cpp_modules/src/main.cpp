#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "videoWriter.hpp"
#include "videoReader.hpp"

int main()
{
    std::string videoPath = "/project/resources/video_test.mp4";
    VideoReader reader(videoPath);

    if (!reader.isOpened())
    {
        std::cerr << "Could not open reader." << std::endl;
        return -1;
    }
    std::cout << "Video Reader instanciado" << std::endl;

    std::string outputPath = "/project/test_outputs/output.avi";
    VideoWriter writer(outputPath, reader.getFrameSize(), reader.getFps());

    if (!writer.isOpened())
    {
        std::cerr << "Could not open writer." << std::endl;
        return -1;
    }

    std::cout << "Video Writer instanciado" << std::endl;

    cv::Mat frame;
    int cnt = 0;
    while (reader.readFrame(frame))
    {
        std::cout << "Frame: " << cnt << std::endl;

        writer.writeFrame(frame);

        cnt++;
    }

    std::cout << "Saiu do while" << std::endl;

    reader.release();
    writer.release();


    
    return 0;
}