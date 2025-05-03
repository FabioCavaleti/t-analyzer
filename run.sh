if docker ps | grep -q "bt-analyzer"; then
    echo "Container is already running, using docker exec..."
    docker exec -it bt-analyzer bash
else
    echo "Container is not running, checking if the image is up-to-date..."

    echo "Pulling the latest version from docker hub..."
    docker pull fcavaleti/bt-analyzer:latest

    echo "Starting container using docker run..."
    xhost +local:root
    docker run \
    --rm \
    --name bt-analyzer \
    -v $(pwd):/project \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -w /project \
    --net=host \
    --privileged \
    -dt fcavaleti/bt-analyzer
fi

