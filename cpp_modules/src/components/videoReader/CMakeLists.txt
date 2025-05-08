cmake_minimum_required(VERSION 3.16)

project(video_reader)

find_package(OpenCV REQUIRED)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)



add_library(video_reader
    include/videoReader.hpp
    src/videoReader.cpp
)

install(TARGETS video_reader)

target_link_libraries(video_reader
    PUBLIC
        log
        ${OpenCV_LIBS}  # Link OpenCV libraries
)

target_include_directories(video_reader
    PUBLIC 
        include
        ${OpenCV_INCLUDE_DIRS}
)
