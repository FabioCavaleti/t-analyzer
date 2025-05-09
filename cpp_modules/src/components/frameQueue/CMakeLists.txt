cmake_minimum_required(VERSION 3.16)

project(frame_queue)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

add_library(frame_queue
    include/FrameQueue.hpp
    src/FrameQueue.cpp
)

install(TARGETS frame_queue)

target_link_libraries(frame_queue
    PUBLIC
        log
        ${OpenCV_LIBS}  
)

target_include_directories(frame_queue
    PUBLIC
        include
        ${OpenCV_INCLUDE_DIRS}
)
