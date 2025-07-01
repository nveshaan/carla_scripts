arch=$(uname -m)

echo "Detected architecture: $arch"

if [[ "$arch" == "x86_64" ]]; then
  docker build --build-arg SPIN_PATH="spinnaker-4.2.0.46-amd64" --build-arg IMG="osrf/ros:humble-desktop-full" --build-arg USER_UID=$(id -u) --build-arg USERNAME=$(id -un) -t ros:cam .
elif [[ "$arch" == "aarch64" ]] || [[ "$arch" == "arm64" ]]; then
  docker build --build-arg SPIN_PATH="spinnaker-4.2.0.46-arm64" --build-arg ARCH="arm64" --build-arg IMG="arm64v8/ros:humble-ros-base-jammy" --build-arg USER_UID=$(id -u) --build-arg USERNAME=$(id -un) -t ros:cam .
else
  echo "Unknown architecture: $arch"
fi
