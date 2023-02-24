@echo off

Rem check for dependencies and cwd
if not exist %cd%/create_submission.bat call :abort "script is not running from within the kits directory" & goto :eof
where docker >nul 2>&1 || ( call :abort "docker needs to be installed and available in the PATH" & goto :eof )

set container_name="luxai_rust_compiler"

rem build the image
set images_output=
for /f %%i in ('docker images -q %container_name%') do set "images_output=%%i"
if "%images_output%"=="" (
    docker build -t %container_name% .
    if %errorlevel% gtr 0 call :abort "error during image build" & goto :eof
)

rem start the container
docker ps | findstr %container_name% 1>nul
if %errorlevel% gtr 0 (
    docker run -it -d --name %container_name% -v %cd%:/root --rm %container_name% bash
)

rem build inside the container
docker exec -w /root -e CARGO_TARGET_DIR=docker_build %container_name% cargo build --release
if %errorlevel% gtr 0 (
    call :abort "error during build inside docker container" & goto :eof
)

rem create submission archive
set submisson_archive="submission.tar.gz"
if exist %submisson_archive% del %submisson_archive%
docker exec -w /root %container_name% tar --exclude=./%submisson_archive% --warning=no-file-changed -czvf %submisson_archive% .
if %errorlevel% gtr 1 (
    call :abort "error during archive creation" & goto :eof
)

rem done
echo "successfully built submission"
goto :eof

rem helper functions
:abort
echo %*
echo Aborting...
pause
goto :eof

