#pragma once

#include <iostream>

/**
 * \brief Simple macro to write logs in debug mode.
 *
 * Will only generate log statements if built in debug mode.
 *
 * \note Since logs can only be written to stderr, they will be
 * included in the error log of the agent. It is for debugging
 * purposes only.
 *
 * Example usage: LOG("this should be 5: " << aValue);
 */
#ifdef DEBUG_BUILD
#    define LOG(...) std::cerr << __VA_ARGS__ << std::endl
#else
#    define LOG(...)
#endif
