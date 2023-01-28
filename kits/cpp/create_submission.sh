#!/bin/bash

abort() {
    echo "$1" 1>&2
    echo "Aborting..." 1>&2
    exit 1
}

[ -f "$PWD/create_submission.sh" ] || abort "script not running from within the build directory"
[ -z "$(which docker)" ] && abort "docker not properly installed"

container_name="luxai_cpp_compiler"

if [ -z "$(docker images -q $container_name)" ]; then
    docker build -t $container_name .
    [ $? -ne 0 ] && abort "error during image build"
fi

if [ -z "$(docker ps | grep -w $container_name)" ]; then
    docker run -it -d --name $container_name -v $PWD:/root --rm $container_name bash
    [ $? -ne 0 ] && abort "error during container start"
fi

docker exec -w /root $container_name bash ./compile.sh -b docker_build

[ $? -ne 0 ] && abort "error during build inside docker container"

submisson_archive="submission.tar.gz"
[ -f "$submisson_archive" ] && rm "$submisson_archive"
tar --exclude=./$submisson_archive --warning=no-file-changed -czvf "$submisson_archive" . && echo "successfully built submission"
