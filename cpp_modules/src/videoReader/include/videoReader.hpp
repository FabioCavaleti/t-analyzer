#ifndef VIDEO_READER_HPP
#define VIDEO_READER_HPP

#include <opencv2/opencv.hpp>
#include <string>

class VideoReader
{
    public:
        VideoReader(const std::string &inputPath);

        bool readFrame(cv::Mat &frame);
        bool isOpened() const;
        void release();
        int getFourcc();
        double getFps();
        cv::Size getFrameSize();

    private:
        cv::VideoCapture cap;
};

#endif // VIDEO_READER_HPP