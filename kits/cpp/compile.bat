@echo off

Rem check for dependencies and cwd
if not exist %cd%/compile.bat call :abort "script is not running from within the kits directory" & goto :eof
where curl >nul 2>&1 || ( call :abort "curl needs to be installed and available in the PATH" & goto :eof )
where cmake >nul 2>&1 || ( call :abort "cmake needs to be installed and available in the PATH" & goto :eof )

Rem parse input parameters
set build_warnings="ON"
set build_config="Release"
set build_debug="OFF"
set build_dir="build"

:parameterloop
if "%1"=="" goto parameterloopend
if "%1"=="/h" call :help & goto :eof
if "%1"=="/w" set build_warnings="OFF"
if "%1"=="/d" (
    set build_debug="ON"
    set build_config="Debug"
)
if "%1"=="/b" (
    if "%2"=="" call :abort "no build directory provided" & goto :eof
    set build_dir="%2"
    shift
)
shift
goto parameterloop
:parameterloopend

Rem download json library
set json_header_path=".\src\lux\nlohmann_json.hpp"
if not exist %json_header_path% curl -o %json_header_path% "https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp"
if not exist %json_header_path% call :abort "something went wrong when downloading the json library" & goto :eof

Rem run cmake
if not exist %build_dir% mkdir %build_dir%
cmake -B %build_dir% -DBUILD_WARNINGS=%build_warnings% -DBUILD_DEBUG=%build_debug% || ( call :abort "error during cmake configuration" & goto :eof )

Rem build the program
cmake --build %build_dir% --config %build_config% || ( call :abort "error during build of the agent" & goto :eof )

Rem done
goto :eof

Rem helper functions
:help
echo Compilation script for cpp agent. By default with all warnings and optimized.
echo NOTE: Script must be run from the directory it is located in!
echo USAGE: .\compile.bat [OPTIONS]
echo OPTIONS can be:
echo   /w            : disable compiler warnings (e.g. -pedantic)
echo   /d            : build in debug mode (O0 and -g)
echo   /b other_dir  : alternative build dir to use (default: build)
echo   /h            : print this help page
goto :eof

:abort
echo %*
echo Aborting...
pause
goto :eof
