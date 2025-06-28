#!/bin/bash

IMAGE_NAME="ros:cam"

docker run -it \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --user=$(id -un) \
  --restart unless-stopped -d \
  --name=camera-container \
  --network=host \
  --ipc=host \
  --pid=host \
  --env DISPLAY=$DISPLAY \
  --env XAUTHORITY=$XAUTHORITY \
  -v $XAUTHORITY:$XAUTHORITY \
  --privileged \
  --device /dev/bus/usb \
  -v /dev:/dev \
  -v /run/udev:/run/udev \
  --group-add $(getent group flirimaging | cut -d: -f3) \
  $IMAGE_NAME \
  ros2 launch spinnaker_camera_driver driver_node.launch.py camera_type:=blackfly_s serial:="'20531756'"
