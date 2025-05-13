cmake_minimum_required(VERSION 3.16)

project(frame_processor)

find_package(OpenCV REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CURL REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_library(frame_processor
    include/FrameProcessor.hpp
    src/FrameProcessor.cpp
)

install(TARGETS frame_processor)

target_link_libraries(frame_processor
    PUBLIC
        log
        ${OpenCV_LIBS}
        nlohmann_json::nlohmann_json
        CURL::libcurl
)

target_include_directories(frame_processor
    PUBLIC
        include
        ${OpenCV_INCLUDE_DIRS}
)