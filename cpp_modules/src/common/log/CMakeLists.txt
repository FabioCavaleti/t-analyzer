cmake_minimum_required(VERSION 3.16)
project(log)

add_library(log
    include/loguru.hpp
    include/logging.hpp
    src/loguru.cpp
)

target_link_libraries(log
    pthread
    dl
)

target_include_directories(log
    PUBLIC include
)

install(TARGETS log)

set_target_properties(log PROPERTIES POSITION_INDEPENDENT_CODE ON)