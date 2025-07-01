#!/bin/bash

IMAGE_NAME="ros:sim"

docker run -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --user=$(id -un) \
  --restart unless-stopped -d \
  --name=sim-container \
  --network=host \
  --ipc=host \
  --pid=host \
  --env DISPLAY=$DISPLAY \
  --env XAUTHORITY=$XAUTHORITY \
  -v $XAUTHORITY:$XAUTHORITY \
  --privileged \
  $IMAGE_NAME
