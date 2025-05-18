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
    for(const types::Detection &det: detections)
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


void FrameProcessor::draw(cv::Mat &frame, const std::string &result_json_path)
{
    logger::info("Attempting to open result JSON: %s", result_json_path.c_str());
    
    std::ifstream ifs(result_json_path);
    if (!ifs.is_open()) {
        logger::error("Failed to open result JSON file: %s", result_json_path.c_str());
        return;
    }

    ifs.seekg(0, std::ios::end);
    if (ifs.tellg() == 0) {
        logger::error("Result JSON file is empty: %s", result_json_path.c_str());
        return;
    }
    ifs.seekg(0, std::ios::beg);
    nlohmann::json result;
     try {
        ifs >> result;
    } catch (const nlohmann::json::parse_error &e) {
        logger::error("JSON parse error in %s: %s", result_json_path.c_str(), e.what());
        return;
    }
    logger::info("Successfully parsed result JSON.");
    if (!result.contains("detections")) {
        logger::warning("Result JSON does not contain 'detections' key.");
        return;
    }

    std::vector<types::Detection> detections;
    for (auto &item : result["detections"])
    {
        float conf = item.value("conf", 0.0f);
        if(conf < getThreshold())
            continue;
        types::BoundingBox bbox(item.value("x", 0),
                            item.value("y", 0),
                            item.value("w", 0),
                            item.value("h", 0));
        types::Detection det;
        det.box = bbox;
        det.conf = item.value("conf",0.0f);
        det.classId = item.value("classId", -1);
        det.label = item.value("label", "");

        detections.push_back(det);
    }

    drawBoxes(frame, detections);
}

void FrameProcessor::process(cv::Mat &frame, const std::string &shm_name, std::string frame_id)
{
    write_frame_to_shm(frame, shm_name);
    logger::info("write_frame_OK!");
    run_inference(frame_id);
    logger::info("Run inference OK!!");
    std::string result_path = "/tmp/results/" + frame_id + ".json";
    while(true)
    {
        logger::info("Teste");
        if (std::filesystem::exists(result_path))
            break;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    shm_unlink(shm_name.c_str());
    draw(frame, result_path);
    std::filesystem::remove(result_path);

}
