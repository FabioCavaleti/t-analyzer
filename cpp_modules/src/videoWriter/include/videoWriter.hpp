#ifndef VIDEO_WRITER_HPP
#define VIDEO_WRITER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <iostream>

class VideoWriter
{
    public:
        VideoWriter(const std::string &outputPath, cv::Size frameSize, double fps = 30.0, int codec = cv::VideoWriter::fourcc('X', 'V', 'I', 'D'));

        bool isOpened() const;
        
        void writeFrame(const cv::Mat &frame);

        void release();

    private:
        cv::VideoWriter writer;
};

#endif

