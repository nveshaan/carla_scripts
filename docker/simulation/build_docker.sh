docker build -f "docker/simulation/Dockerfile"  --build-arg IMG="osrf/ros:humble-desktop-full" --build-arg USER_UID=$(id -u) --build-arg USERNAME=$(id -un) -t ros:sim .
