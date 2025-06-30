# Offroad Navigation

```bash
offroad_navigation
├── carla
│   ├── agent_testing/
│   └── data_collection
│       ├── dynamic_weather.py
│       ├── run_collector.py
│       └── supervisor.py
├── checkpoints/
├── configs
│   ├── sensor_config.yaml
│   └── train_config.yaml
├── data
│   └── marathon.hdf5
├── dataloader
│   └── dataset.py
├── docker/
├── env
│   ├── environment.yml
│   └── requirements.txt
├── models
│   ├── image_net.py
│   ├── network_utils.py
│   └── resnet.py
├── ros2_ws
│   └── src
│       └── torch_inference
│           ├── package.xml
│           ├── setup.cfg
│           ├── setup.py
│           └── torch_inference
│               ├── __init__.py
│               ├── inference.py
│               ├── models -> ./offroad_navigation/models
|               └── checkpoints -> ./offroad_navigation/checkpoints
├── scripts
│   └── train.py
├── utils
│   ├── infer_model.py
│   └── sample_data.py
└── visualization
    ├── hdf5_view.ipynb
    ├── eda.ipynb
    ├── hdf5_playback.py
    └── image_waypoint.py
```

## Setup
The code for data collection has been written to run on **Windows 10**, with **Carla 0.10.0** and **Python 3.9**. Although, with minor changes, it can be run on **Ubuntu** as well. An Nvidia GPU is a must.

PyTorch scripts are meant to be run on **Ubuntu 22.04** with **CUDA 12.1**. The Conda environment is stored in `env/environment.yml`. Run the below commands to setup the Python environment.
```bash
conda env create -f ./env/environment.yml -p ./env
conda activate ./env
pip3 install ./env/requirements.txt
```
This Python environment can be used for the rest of the workflow.

Other software to be used is listed below:
 - Docker
 - ROS2 Humble



<!-- dev: sync folders

copy the folders first

# Create symlink to shared models directory
ln -s /mnt/d/offroad_navigation/models /mnt/d/offroad_navigation/ros2_ws/src/torch_inference/torch_inference/models

# Do the same for checkpoints
ln -s /mnt/d/offroad_navigation/checkpoints /mnt/d/offroad_navigation/ros2_ws/src/torch_inference/torch_inference/checkpoints -->
