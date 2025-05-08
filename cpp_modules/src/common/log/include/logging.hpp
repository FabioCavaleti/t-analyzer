#pragma once

#include "loguru.hpp"
#include <string>
#include <iostream>

namespace logger {
    template <typename... Args>
    inline void info(const char* format, Args&&... args){
        LOG_F(INFO, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void error(const char* format, Args&&... args) {
        LOG_F(ERROR, format, std::forward<Args>(args)...);
    }

    template <typename... Args>
    inline void warning(const char* format, Args&&... args){
        LOG_F(WARNING, format, std::forward<Args>(args)...);
    }
} //namespace log