ARG IMG=ros
FROM $IMG

ARG ARCH="amd64"

RUN find /etc/apt/sources.list.d/ -name '*ros*' -exec rm -f {} \; && \
    rm -f /etc/apt/trusted.gpg.d/ros* && \
    apt-key del F42ED6FBAB17C654 || true && \
    apt-get update && apt-get install -y curl gnupg2 lsb-release && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    | gpg --dearmor -o /etc/apt/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=arm64 signed-by=/etc/apt/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list;

# install necessary tools
RUN apt-get update \
    && apt-get install -y \
    nano \
    vim \
    usbutils \
    net-tools \
    udev \
    build-essential
  

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

# spinnaker sdk installation
ARG SPIN_PATH="spinnaker-4.2.0.46-amd64"
ADD $SPIN_PATH /root/spinnaker-debs
# RUN if [ "$ARCH" = "amd64" ]; then \
#       wget https://flir.netx.net/file/asset/71711/original/attachment -O /root/spinnaker-debs/; \
#     else \
#       wget https://flir.netx.net/file/asset/71712/original/attachment -O /root/spinnaker-debs/; \
#     fi
RUN echo "y" | /root/spinnaker-debs/install_spinnaker.sh $USERNAME

# spinnaker ros2 camera driver installation
RUN sudo apt-get update && sudo apt-get install ros-humble-spinnaker-camera-driver -y

# ready to run
COPY entrypoint.sh /entrypoint.sh
COPY bashrc /home/${USERNAME}/.bashrc

# fix permissions on user files
RUN chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc && \
    chmod 644 /home/${USERNAME}/.bashrc && \
    chmod +x /entrypoint.sh

RUN rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/spinnaker-debs

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD ["bash"]
