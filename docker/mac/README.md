
> This is dev container aimed to run this codebase along side Carla on Apple Silicon Mac.

---

# Build
```bash
docker build --build-arg USER_UID=$(id -u) --build-arg USERNAME=$(id -un) -t mac .
```

# Run
```bash
xhost +localhost
docker run --platform linux/amd64 --name=dev -it -e DISPLAY=host.docker.internal:0. --user=$(id -un) --network=host --ipc=host --pid=host --privileged -v ~/Developer/offroad_navigation:/app -v /Volumes/Marsupium/marathon.hdf5:/app/data/marathon.hdf5 mac
```
Either use `--rm` or `--name=dev` tags.

---

Be sure to run the following command before installing any packages:
```bash
sudo apt-get update
```