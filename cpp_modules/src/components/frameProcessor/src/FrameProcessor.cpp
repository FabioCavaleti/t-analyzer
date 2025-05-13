#include "FrameProcessor.hpp"
#include "logging.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <curl/curl.h>


void FrameProcessor::write_frame_to_shm(const cv::Mat &frame, const std::string &shm_name)
{
    size_t size = frame.total() * frame.elemSize();
    int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    ftruncate(fd, size);
    void *ptr = mmap(0, size, PROT_WRITE, MAP_SHARED, fd, 0);
    memcpy(ptr, frame.data, size);
    munmap(ptr, size);
    close(fd);
}

void FrameProcessor::run_inference(std::string frame_id)
{
    CURL *curl = curl_easy_init();
    if (curl)
    {
        std::string url = "http://localhost:8000/infer?frame_id=" + frame_id;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }
}

void FrameProcessor::draw(cv::Mat &frame, const std::string &result_json_path)
{
    std::ifstream ifs(result_json_path);
    nlohmann::json result;
    ifs >> result;

    for (auto &det: result["detectinos"])
    {
        cv::rectangle(frame,
                        cv::Rect(det["x"], det["y"], det["w"], det["h"]),
                        cv::Scalar(0, 255, 0), 2);
    }
}

void FrameProcessor::process(cv::Mat &frame, const std::string &shm_name, std::string frame_id)
{
    write_frame_to_shm(frame, shm_name);
    run_inference(frame_id);
    std::string result_path = "/tmp/results/" + frame_id + ".json";
    draw(frame, result_path);

}
