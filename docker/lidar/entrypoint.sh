#!/bin/bash

set -e

# Prompt for IP address if DISPLAY not set
if [[ -z "$DISPLAY" ]]; then
    echo "DISPLAY is not set. Let's configure it for X11 forwarding."
    read -p "Enter your host IP address (for DISPLAY): " ip_address

    if [[ $ip_address =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        export DISPLAY="${ip_address}:0.0"

        # Append to bashrc if not already present
        if ! grep -q "export DISPLAY=" ~/.bashrc; then
            echo "export DISPLAY=${ip_address}:0" >> ~/.bashrc
            echo "DISPLAY environment variable added to ~/.bashrc"
        fi
    else
        echo "Invalid IP address format. DISPLAY not configured."
    fi
else
    echo "DISPLAY is already set to $DISPLAY"
fi

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

exec $@
