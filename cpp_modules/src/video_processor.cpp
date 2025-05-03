#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::VideoCapture cap("/project/resources/video_test.mp4");
    
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video" << std::endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    cv::VideoWriter out("output.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), 30.0, cv::Size(frame_width, frame_height));

    cv::Mat frame;

    while(true)
    {
        cap >> frame;

        if (frame.empty())
            break;

        
        out.write(frame);
        
            // cv::imshow("Video", frame);

            // if (cv::waitKey(30) == 'q')
            // {
            //     break;
            // }
    }

    cap.release();
    out.release();
    cv::destroyAllWindows();

    return 0;
}