FROM --platform=linux/amd64 python:3.10.18

# install necessary tools
RUN apt-get update \
    && apt-get install -y \
    nano \
    vim \
    net-tools \
    tmux \
    && apt-get clean

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
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# install conda
RUN apt-get update && apt-get install -y wget bzip2 ca-certificates \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean --all -y

# set conda environment path
ENV PATH /opt/conda/bin:$PATH

# setup conda environment
RUN conda install mamba -c conda-forge && \
    mamba create -n carla-ros -y

# initialize mamba for shell
RUN mamba shell init --shell bash

# install ROS dependencies
RUN /bin/bash -c "source ~/.bashrc && mamba activate carla-ros && \
    conda config --env --add channels conda-forge && \
    conda config --env --remove channels defaults && \
    conda config --env --add channels robostack-humble && \
    mamba install ros-humble-desktop -y && \
    mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep -y"

# install additional dependencies
RUN /bin/bash -c "source ~/.bashrc && mamba activate carla-ros && \
    pip3 install torch torchvision pygame h5py"

# install carla api
# COPY carla-0.10.0-cp311-cp311-linux_x86_64.whl /tmp/carla-0.10.0-cp311-cp311-linux_x86_64.whl

# RUN /bin/bash -c "source ~/.bashrc && mamba activate carla-ros && \
#     pip3 install --no-cache-dir --force-reinstall --no-deps /tmp/carla-0.10.0-cp311-cp311-linux_x86_64.whl && \
#     rm -rf /root/.cache/pip && \
#     rm -rf /tmp/carla-0.10.0-cp311-cp311-linux_x86_64.whl"

# Download and install CARLA 0.9.15 from official URL
RUN /bin/bash -c "source ~/.bashrc && mamba activate carla-ros && \
    wget -q https://tiny.carla.org/carla-0-9-15-linux -O /tmp/carla.tar.gz && \
    mkdir -p /tmp/carla && \
    tar -xzf /tmp/carla.tar.gz -C /tmp/carla && \
    find /tmp/carla -name '*.whl' -exec pip3 install {} \; && \
    rm -rf /tmp/carla /tmp/carla.tar.gz /root/.cache/pip"

# finalize
ENV DISPLAY=host.docker.internal:0

RUN echo "source /opt/conda/bin/activate carla-ros" >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc && \
    chmod 644 /home/${USERNAME}/.bashrc

ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && exec bash"]
CMD ["bash"]
