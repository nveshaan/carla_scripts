FROM --platform=linux/amd64 condaforge/miniforge3:latest

# Use bash as default shell
SHELL ["/bin/bash", "-c"]

# Install necessary tools
RUN apt-get update && \
    apt-get install -y \
    nano \
    vim \
    net-tools \
    tmux \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    && apt-get clean

# Add non-root user
ARG USERNAME=nveshaan
ARG USER_UID=501
ARG USER_GID=${USER_UID}

RUN groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    mkdir /home/$USERNAME/.config && \
    chown $USER_UID:$USER_GID /home/$USERNAME/.config && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# # Install Miniconda
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
#     bash /tmp/miniconda.sh -b -p /opt/conda && \
#     rm /tmp/miniconda.sh && \
#     /opt/conda/bin/conda clean --all -y

# # Set conda in PATH
# ENV PATH="/opt/conda/bin:$PATH"

# RUN conda config --remove-key channels || true && \
#     conda config --add channels https://conda.anaconda.org/conda-forge && \
#     conda config --set channel_priority strict

# # Conda config and install mamba
# RUN . /opt/conda/etc/profile.d/conda.sh && conda activate base && \
#     conda install -n base -c conda-forge mamba -y && \
#     mamba create -n carla-ros -y

RUN mamba create -n carla-ros -y \
    && conda init bash \
    && conda config --set auto_activate_base false

# Activate conda in bash sessions
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Install ROS 2 and development tools
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate carla-ros && \
    conda config --env --add channels conda-forge && \
    conda config --env --add channels robostack-humble && \
    mamba install ros-humble-desktop -y && \
    mamba install compilers cmake pkg-config make ninja colcon-common-extensions catkin_tools rosdep -y

# Install Python dependencies
RUN . /opt/conda/etc/profile.d/conda.sh && conda activate carla-ros && \
    pip install torch torchvision pygame h5py

# Download and install CARLA 0.9.15
# RUN . /opt/conda/etc/profile.d/conda.sh && conda activate carla-ros && \
#     wget -q https://tiny.carla.org/carla-0-9-15-linux -O /tmp/carla.tar.gz && \
#     mkdir -p /tmp/carla && \
#     tar -xzf /tmp/carla.tar.gz -C /tmp/carla && \
#     find /tmp/carla -name '*.whl' -exec pip install {} \; && \
#     rm -rf /tmp/carla /tmp/carla.tar.gz /root/.cache/pip

# Finalize: Set up default user environment
ENV DISPLAY=host.docker.internal:0

RUN echo "source /opt/conda/bin/activate carla-ros" >> /home/${USERNAME}/.bashrc && \
    chown ${USERNAME}:${USERNAME} /home/${USERNAME}/.bashrc && \
    chmod 644 /home/${USERNAME}/.bashrc

# Optional cleanup to reduce image size
RUN conda clean --all -y

# Entry point
ENTRYPOINT ["/bin/bash", "--login", "-c"]
CMD ["bash"]