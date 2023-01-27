#!/bin/bash

help() {
    echo "Compilation script for cpp agent. By default with all warnings and optimized."
    echo "NOTE: Script must be run from the directory it is located in!"
    echo "USAGE: ./compile.sh [OPTIONS]"
    echo "OPTIONS can be:"
    echo "  -w  --no-warnings   : disable compiler warnings (e.g. -pedantic)"
    echo "  -d  --debug         : build in debug mode (O0 and -g)"
    echo "  -b  --build-dir     : alternative build dir to use (default: build)"
    echo "  -h  --help          : print this help page"
}

abort() {
    echo "$1" 1>&2
    echo "Aborting..." 1>&2
    exit 1
}

[ -f "$PWD/compile.sh" ] || abort "script not running from within the build directory"
[ -z "$(which cmake)" ] && abort "cmake must be installed"
[ -z "$(which curl)" ] && abort "curl must be installed"

build_warnings="ON"
build_debug="OFF"
build_config="Release"
build_dir="build"

while [[ $# -gt 0 ]]; do
    case $1 in
        -w|--no-warnings)
            build_warnings="OFF"
            shift
            ;;
        -d|--debug)
            build_debug="ON"
            build_config="Debug"
            shift
            ;;
        -b|--build-dir)
            shift
            build_dir="$1"
            shift
            ;;
        -h|--help)
            help && exit 0
            ;;
    esac
done

json_header_path="./src/lux/nlohmann_json.hpp"
[ -f "$json_header_path" ] || curl -o "$json_header_path" "https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp"

mkdir -p $build_dir

cmake -B $build_dir -DBUILD_WARNINGS=$build_warnings -DBUILD_DEBUG=$build_debug

[ $? -ne 0 ] && abort "error during cmake configuration"

cmake --build $build_dir --config $build_config
