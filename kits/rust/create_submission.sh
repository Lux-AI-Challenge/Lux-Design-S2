#!/bin/bash

abort() {
    echo "$1" 1>&2
    echo "Aborting..." 1>&2
    exit 1
}

[ -f "$PWD/create_submission.sh" ] || abort "script not running from within the build directory"
command -v docker &> /dev/null || abort "docker not properly installed"

container_name="luxai_rust_compiler"

if [ -z "$(docker images -q $container_name)" ]; then
    docker build -t "$container_name" . || abort "error during image build"
fi

if [ -z "$(docker ps | grep -w $container_name)" ]; then
    docker run -it -d --name "$container_name" -v "$PWD:/root" --rm "$container_name" bash || abort "error during container start"
fi

docker exec -w /root -e CARGO_TARGET_DIR=docker_build "$container_name" cargo build --release || abort "error during build inside docker container"

submisson_archive="submission.tar.gz"
[ -f "$submisson_archive" ] && rm "$submisson_archive"
tar --exclude=./$submisson_archive --warning=no-file-changed -czf "$submisson_archive" . && echo "successfully built submission"
