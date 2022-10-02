#include "lux/log.hpp"

#include <fstream>

namespace lux {
    void dumpJsonToFile(const char *path, json data) {
        std::fstream file(path, std::ios::trunc | std::ios::out);
        if (file.is_open()) {
            file << data;
            file.close();
        }
    }
}  // namespace lux
