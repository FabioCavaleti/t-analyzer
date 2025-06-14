#include "FrameProcessor.hpp"
#include "types.hpp"
#include "logging.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <nlohmann/json.hpp>
#include <curl/curl.h>

#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>

FrameProcessor::FrameProcessor(std::string url) : URL_(url) {}

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

bool FrameProcessor::modelRegistered()
{
    return model_registered_;
}

bool FrameProcessor::call_api(std::string &endpoint)
{
    CURL *curl = curl_easy_init();
    if (!curl)
        return false;

    std::string url = URL_ + endpoint;
    curl_easy_setopt(curl, CURLOPT_URL, endpoint.c_str());
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    return !res;
}

bool FrameProcessor::register_shape_(int height, int width)
{

    std::string url = URL_ + "/register_shape?" +
                      "height=" + std::to_string(height) +
                      "&width=" + std::to_string(width);
    bool res = call_api(url);

    return res;
}

bool FrameProcessor::register_court_detector_()
{
    std::string url = URL_ + "/register_court_detector";
    bool res = call_api(url);

    return res;
}

bool FrameProcessor::register_player_detector_()
{
    std::string url = URL_ + "/register_player_detector";
    bool res = call_api(url);

    return res;
}

bool FrameProcessor::register_ball_detector_()
{
    std::string url = URL_ + "/register_ball_detector";
    bool res = call_api(url);

    return res;
}

void FrameProcessor::register_inference_model_(const cv::Mat &frame)
{
    int height = frame.rows;
    int width = frame.cols;
    bool model_registered = register_shape_(height, width);
    model_registered = model_registered && register_court_detector_();
    model_registered = model_registered && register_player_detector_();
    model_registered = model_registered && register_ball_detector_();
    if (!model_registered)
    {
        logger::error("Error registering models");
        return;
    }
    model_registered_ = model_registered;
}

void FrameProcessor::run_inference(std::string frame_id)
{
    CURL *curl = curl_easy_init();
    if (curl)
    {
        std::string url = URL_ + "/infer?frame_id=" + frame_id;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_perform(curl);
        curl_easy_cleanup(curl);
    }
}

float FrameProcessor::getThreshold()
{
    return threshold_;
}

void FrameProcessor::setThreshold(float value)
{
    threshold_ = value;
}

void drawBoxes(cv::Mat &frame, const std::vector<types::Detection> &detections)
{
    for (const types::Detection &det : detections)
    {
                     const types::BoundingBox &box = det.box;
                     cv::rectangle(frame,
                                   cv::Rect(box.x, box.y, box.width, box.height),
                                   cv::Scalar(0, 255, 0), 2);

                     cv::putText(frame, det.label,
                                 cv::Point(box.x, box.y - 5),
                                 cv::FONT_HERSHEY_SIMPLEX,
                                 0.5, cv::Scalar(0, 255, 0), 1);
    }
}

void drawKeypoints(cv::Mat &frame, const std::vector<float> keypoints)
{
    for (size_t i = 0; i + 1 < keypoints.size(); i += 2)
    {
        float x = keypoints[i];
        float y = keypoints[i + 1];

        cv::circle(frame, cv::Point(x, y), 5, cv::Scalar(255, 0, 0), -1);
    }
}

void FrameProcessor::draw(cv::Mat &frame, const std::string &result_json_path)
{

    std::ifstream ifs(result_json_path);
    if (!ifs.is_open())
    {
        logger::error("Failed to open result JSON file: %s", result_json_path.c_str());
        return;
    }

    ifs.seekg(0, std::ios::end);
    if (ifs.tellg() == 0)
    {
        logger::error("Result JSON file is empty: %s", result_json_path.c_str());
        return;
    }
    ifs.seekg(0, std::ios::beg);
    nlohmann::json result;
    try
    {
        ifs >> result;
    }
    catch (const nlohmann::json::parse_error &e)
    {
        logger::error("JSON parse error in %s: %s", result_json_path.c_str(), e.what());
        return;
    }
    if (!result.contains("detections"))
    {
        logger::warning("Result JSON does not contain 'detections' key.");
        return;
    }
    if (!result.contains("keypoints"))
    {
        logger::warning("Result JSON does not contain 'keypoints' key.");
        return;
    }

    // logger::info("Printing result json \n %s", result.dump(4).c_str());

    std::vector<types::Detection> detections;
    for (auto &item : result["detections"])
    {
        float conf = item.value("conf", 0.0f);
        if (conf < getThreshold())
            continue;
        types::BoundingBox bbox(item.value("x", 0),
                                item.value("y", 0),
                                item.value("w", 0),
                                item.value("h", 0));
        types::Detection det;
        det.box = bbox;
        det.conf = item.value("conf", 0.0f);
        det.classId = item.value("classId", -1);
        det.label = item.value("label", "");

        detections.push_back(det);
    }
    drawBoxes(frame, detections);
    drawKeypoints(frame, result["keypoints"]);
}

void FrameProcessor::process(cv::Mat &frame, const std::string &shm_name, std::string frame_id)
{
    write_frame_to_shm(frame, shm_name);
    if (!modelRegistered())
        register_inference_model_(frame);
    run_inference(frame_id);
    std::string result_path = "/tmp/results/" + frame_id + ".json";
    while (true)
    {
        if (std::filesystem::exists(result_path))
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    shm_unlink(shm_name.c_str());
    draw(frame, result_path);
    std::filesystem::remove(result_path);
}
