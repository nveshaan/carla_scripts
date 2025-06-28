ARG IMG=ros
FROM $IMG

ARG ARCH=amd64
# Clean old ROS keys and sources and add new ones (only if ARCH=arm64)
RUN if [ "$ARCH" = "arm64" ]; then \
      find /etc/apt/sources.list.d/ -name '*ros*' -exec rm -f {} \; && \
      rm -f /etc/apt/trusted.gpg.d/ros* && \
      apt-key del F42ED6FBAB17C654 || true && \
      apt-get update && apt-get install -y curl gnupg2 lsb-release && \
      mkdir -p /etc/apt/keyrings && \
      curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      | gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg && \
      echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/ros2.list; \
    fi

# install necessary tools
RUN apt-get update \
    && apt-get install -y \
    nano \
    vim \
    usbutils \
    net-tools \
    udev \
    build-essential \
    python3-pip
  

# adding a non-root user
ARG USERNAME
# USER_UID must be specified at build time with --build-arg USER_UID=$(id -u)
ARG USER_UID
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir /home/$USERNAME/.config && chown $USER_UID:$USER_GID /home/$USERNAME/.config

# install 'sudo'
RUN apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# install colcon and rosdep
RUN sudo apt-get update \
    && sudo apt-get install python3-colcon-common-extensions python3-rosdep -y \
    && rosdep update

# ouster-ros package installation
# install dependencies
RUN apt-get update && apt-get install -y\
    libeigen3-dev \
    libjsoncpp-dev \
    libspdlog-dev \
    libcurl4-openssl-dev \
    cmake

# git repo
RUN mkdir -p ros2_ws/src
WORKDIR /ros2_ws/src

RUN git clone -b ros2 --recurse-submodules https://github.com/ouster-lidar/ouster-ros.git 

WORKDIR /ros2_ws
RUN bash -c "source /opt/ros/humble/setup.bash && \
    apt-get update && \
    apt-get install -y \
        libeigen3-dev \
        libjsoncpp-dev \
        libspdlog-dev \
        libcurl4-openssl-dev \
        cmake && \
    rosdep update && \
    rosdep install --from-paths src -y --ignore-src && \
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# ouster-sdk python pkg
WORKDIR /
RUN bash -c " sudo apt-get -y install python3.10-venv && python3 -m venv ouster \
    && source /ouster/bin/activate \
    && pip3 install ouster-sdk"

# install rviz2
RUN sudo apt update && sudo apt-get install -y ros-humble-rviz2

# ready to run
COPY entrypoint.sh /entrypoint.sh
COPY bashrc /home/${USERNAME}/.bashrc

# fix permissions on user files
RUN chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc && \
    chmod 644 /home/${USERNAME}/.bashrc && \
    chmod +x /entrypoint.sh

RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
