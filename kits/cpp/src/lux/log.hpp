#pragma once

#include <iostream>

#include "lux/json.hpp"

/**
 * \brief Simple macro to write logs in debug mode.
 *
 * Will only generate log statements if built in debug mode.
 *
 * \note Since logs can only be written to stderr, they will be
 * included in the error log of the agent. It is for debugging
 * purposes only.
 *
 * Example usage: LUX_LOG("this should be 5: " << aValue);
 */
#ifdef DEBUG_BUILD
#    define LUX_LOG(...) std::cerr << __VA_ARGS__ << std::endl
#else
#    define LUX_LOG(...)
#endif

namespace lux {
    /**
     * \brief Dumps contents of json to a file.
     *
     * Will only create file and dump contents in debug mode.
     * The destination will always be truncated before writing.
     */
    void dumpJsonToFile(const char *path, json data);
}  // namespace lux
