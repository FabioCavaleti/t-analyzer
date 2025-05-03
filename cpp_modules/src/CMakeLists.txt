add_executable(bt-analyzer video_processor.cpp)

target_link_libraries(bt-analyzer PRIVATE ${OpenCV_LIBS})