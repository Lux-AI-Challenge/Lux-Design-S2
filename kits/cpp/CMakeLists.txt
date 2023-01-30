cmake_minimum_required(VERSION 3.12) # minimum for C++ 20

project(lux-ai-agent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(BINARY agent.out)

option(BUILD_DEBUG "Build in debug mode" OFF)
option(BUILD_WARNINGS "Build using all reasonable warnings" ON)

if(${BUILD_DEBUG})
    add_compile_definitions(
        DEBUG_BUILD
    )
endif()

if(${BUILD_WARNINGS})
    add_compile_options(
        -Wall
    )
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
        add_compile_options(
            -W3
        )
    else()
        add_compile_options(
            -Werror
            -Wextra
            -Wpedantic
            -Wcast-align
            -Wunused
            -Wshadow
        )
    endif()
endif()

include(sources.cmake)

set(LUX_SRC_FILES
    src/lux/action.cpp
    src/lux/board.cpp
    src/lux/common.cpp
    src/lux/config.cpp
    src/lux/factory.cpp
    src/lux/log.cpp
    src/lux/observation.cpp
    src/lux/unit.cpp
)

add_executable(${BINARY} ${AGENT_SRC_FILES} ${LUX_SRC_FILES})

target_include_directories(${BINARY} PUBLIC
    ${CMAKE_SOURCE_DIR}/src
)
