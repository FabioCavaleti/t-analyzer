cmake_minimum_required(VERSION 3.16)

project(video_writer)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_library(video_writer
    include/videoWriter.hpp
    src/videoWriter.cpp
)

install(TARGETS video_writer)


target_link_libraries(video_writer
    PUBLIC
        ${OpenCV_LIBS}  # Link OpenCV libraries
)

target_include_directories(video_writer
    PUBLIC include
        include
        $(OpenCV_INCLUDE_DIRS)
)
