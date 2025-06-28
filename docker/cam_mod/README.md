
> This container is a production environment made to publish images from FLIR camera using ros2 spinnaker camera driver.

---
# Setup
The Spinnaker SDK has to be installed on host PC to enable communication between container and camera. Navigate to the spinnaker sdk folder and follow the steps.

# Build
```bash
chmod +x build_docker.sh
./build_docker.sh
```

# Run
```bash
chmod +x run_docker.sh
./run_docker.sh
```
When nothing works, just `sudo reboot`

