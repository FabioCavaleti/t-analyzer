cmake_minimum_required(VERSION 3.16)
project(Bt-Analyzer)

find_package(OpenCV REQUIRED)


add_subdirectory(common)
add_subdirectory(components)



add_executable(bt-analyzer main.cpp)

target_link_libraries(bt-analyzer
    PRIVATE
        log
        video_reader
        video_writer
        ${OpenCV_LIBS}  # Link OpenCV libraries
)