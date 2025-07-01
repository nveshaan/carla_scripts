#!/bin/bash

set -o errexit


echo "This is a script to assist with installation of the Spinnaker SDK."
echo "Auto-confirmed: Installing all the Spinnaker SDK packages..."

set +o errexit
EXISTING_VERSION=$(dpkg -s spinnaker 2> /dev/null | grep '^Version:' | sed 's/^.*: //')
set -o errexit

if [ ! -z "$EXISTING_VERSION" ]; then
    echo "A previous installation of Spinnaker has been detected on this machine (Version: $EXISTING_VERSION). Proceeding anyway..."
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
sudo dpkg -i spinview-qt_*.deb
sudo dpkg -i spinview-qt-dev_*.deb
sudo dpkg -i spinupdate_*.deb
sudo dpkg -i spinupdate-dev_*.deb
sudo dpkg -i spinnaker_*.deb
sudo dpkg -i spinnaker-doc_*.deb

echo "Auto-confirmed: Adding udev entry..."
sudo sh configure_spinnaker.sh "$1"

echo "Auto-confirmed: Setting USB-FS memory size..."
sudo sh configure_usbfs.sh

echo "Auto-confirmed: Adding Spinnaker prebuilt examples to system path..."
sudo sh configure_spinnaker_paths.sh

ARCH=$(ls libspinnaker_* | grep -oP '[0-9]_\K.*(?=.deb)' || [[ $? == 1 ]])
if [ "$ARCH" = "arm64" ]; then
    BITS=64
elif [ "$ARCH" = "armhf" ]; then
    BITS=32
fi

if [ -z "$BITS" ]; then
    echo "Could not automatically add the Spinnaker GenTL Producer to the GenTL environment variable."
    echo "To use the Spinnaker GenTL Producer, please follow the GenTL Setup notes in the included README."
else
    echo
    echo "Auto-confirmed: Adding Spinnaker GenTL Producer to GENICAM_GENTL${BITS}_PATH..."
    sudo sh configure_gentl_paths.sh $BITS
fi

echo
echo "Auto-confirmed: GigEVision camera setup prompt acknowledged."
echo "To avoid packet loss when streaming with GEV cameras, please review section 4 of the README to configure your network adapter appropriately."
echo "To install ethtool, run the following command:"
echo "  sudo apt install ethtool"
echo "To adjust network interface eth0, use the following:"
echo "  sudo /opt/spinnaker/bin/gev_nettweak eth0"

echo
echo "Installation complete. You will need to reboot your system for all changes to take effect."
echo
echo "Thank you for installing the Spinnaker SDK."

exit 0

