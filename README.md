# F1Tenth RL Example

This is a simple example of reinforcement learning by [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) for the F1Tenth racing car. The environment is based on the [F1Tenth Gym](https://github.com/f1tenth/f1tenth_gym)

<p align="center">
  <img src="./media/sac.gif" width="300" alt="cartpole">
</p>


## Tested Native Environment
- Ubuntu Focal 20.04 (LTS)
- NVIDIA Driver 510 or later due to PyTorch 2.x

## Installation

<details>
<summary>Docker Installation</summary>

### Install Docker

[Installation guide](https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository)

```bash
# Install from get.docker.com
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo groupadd docker
sudo usermod -aG docker $USER
```

### Setup GPU for Docker
[Installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list 

sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit nvidia-container-runtime

sudo nvidia-ctk runtime configure --runtime=docker

sudo systemctl restart docker
```
</details>

### Setup with Docker

```bash
# build container (with GPU support)
make build-gpu
# or build container (without GPU support)
# make build-cpu
```

Open remote container via Vscode (Recommend)
1. Open the folder using vscode
2. Ctrl+P and select 'devcontainer rebuild and reopen in container'
Then, you can skip the following commands

```bash
# Or Run container via terminal (with GPU support)
make bash-gpu
# [Optional] Run container via terminal (without GPU support)
# make bash-cpu
```

### How to train and run

Train 
```bash
cd scripts
python3 train.py
```
Then, you can find the trained model in `scripts/models/`

Run (Need your trained model)
```bash
cd scripts
python3 run.py <path_to_model>
```
