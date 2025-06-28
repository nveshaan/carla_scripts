#!/bin/bash

set -o errexit

USERNAME="yourusername"  # <-- CHANGE this to the target Linux user or pass via $1

echo "Starting Spinnaker SDK installation (non-interactive)..."

set +o errexit
EXISTING_VERSION=$(dpkg -s spinnaker 2>/dev/null | grep '^Version:' | sed 's/^.*: //')
set -o errexit

if [ ! -z "$EXISTING_VERSION" ]; then
    echo "Previous Spinnaker installation found (Version: $EXISTING_VERSION). Proceeding anyway..."
fi

echo "Installing Spinnaker packages..."

cd /root/spinnaker-debs

sudo dpkg -i libgentl_*.deb
sudo dpkg -i libspinnaker_*.deb
sudo dpkg -i libspinnaker-dev_*.deb
sudo dpkg -i libspinnaker-c_*.deb
sudo dpkg -i libspinnaker-c-dev_*.deb
sudo dpkg -i libspinvideo_*.deb
sudo dpkg -i libspinvideo-dev_*.deb
sudo dpkg -i libspinvideo-c_*.deb
sudo dpkg -i libspinvideo-c-dev_*.deb
sudo apt-get install -y ./spinview-qt_*.deb
sudo dpkg -i spinview-qt-dev_*.deb
sudo dpkg -i spinupdate_*.deb
sudo dpkg -i spinupdate-dev_*.deb
sudo dpkg -i spinnaker_*.deb
sudo dpkg -i spinnaker-doc_*.deb

# Run udev config
echo "Launching udev configuration script..."
sudo sh configure_spinnaker.sh "$1"

# Configure USB-FS memory
echo "Configuring USB-FS memory..."
sudo ./configure_usbfs.sh

# Add Spinnaker prebuilt example paths
echo "Configuring Spinnaker paths..."
sudo ./configure_spinnaker_paths.sh

# Add GenTL path based on detected architecture
ARCH=$(ls libspinnaker_* | grep -oP '[0-9]_\K.*(?=.deb)' || true)
if [ "$ARCH" = "amd64" ]; then
    BITS=64
elif [ "$ARCH" = "i386" ]; then
    BITS=32
fi

if [ ! -z "$BITS" ]; then
    echo "Adding Spinnaker GenTL producer to GENICAM_GENTL${BITS}_PATH..."
    sudo ./configure_gentl_paths.sh "$BITS"
else
    echo "Warning: Could not determine architecture to add GenTL path."
fi

# GigEVision warning
echo "Reminder: If using GigEVision cameras, run network tuning via:"
echo "  sudo apt install ethtool"
echo "  sudo /opt/spinnaker/bin/gev_nettweak eth0"

echo
echo "Installation complete. Please reboot your system for all changes to take effect."
echo
echo "Thank you for installing the Spinnaker SDK."

exit 0
