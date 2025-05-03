#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::VideoCapture cap("/home/fcavaleti/bt-analyzer/resources/video_test.mp4");
    
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while(true)
    {
        cap >> frame;

        if (frame.empty())
            break;
        
            cv::imshow("Video", frame);

            if (cv::waitKey(30) == 'q')
            {
                break;
            }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}