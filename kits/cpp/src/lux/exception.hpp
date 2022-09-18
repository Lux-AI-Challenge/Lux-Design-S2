#pragma once

#include <exception>
#include <string>

namespace lux {
    class Exception : public std::exception {
        std::string msg;

       public:
        Exception(const std::string &what) : std::exception(), msg(what) {}

        const char *what() const noexcept override { return msg.c_str(); }
    };
}  // namespace lux
