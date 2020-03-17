# Installing Prerequisites  <br>

This codebase currently only supports Python3 with CUDA. 
Make sure you have an NVIDIA GPU with corresponding drivers installed.

There are 2 ways to install dependencies.
We **recommend** using [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) so that you run with specific GCC and CUDA versions verified to work with the arithmetic coder library [torchac](https://github.com/fab-jul/L3C-PyTorch#the-torchac-module-fast-entropy-coding-in-pytorch).
However, this is not required to install torchac.

Note that torchac is **not needed to train models** or **evaluate theoretical bitrates**, but it **needs to be installed to compress/decompress images**.
In theory, you should be able to train/evaluate models without using the exact versions of libraries we used.

## Option 1: Using nvidia-docker

[nvidia-docker](https://github.com/NVIDIA/nvidia-docker) allows you to run docker images with specific versions of CUDA and GCC. This is the recommended way to install torchac.

### Set up nvidia-docker
If you don't have docker installed, you can install docker using [instructions here](https://docs.docker.com/install/).

Install `nvidia-docker` using [instructions here](https://github.com/NVIDIA/nvidia-docker#quickstart).

Once you have `nvidia-docker` installed, you should register an account on [ngc.nvidia.com](https://ngc.nvidia.com/) and pull the following [PyTorch image](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch):
```
docker pull nvcr.io/nvidia/pytorch:19.06-py3
```

### Installing dependencies

First clone the repo. Then run the docker image:
```
docker run -it --runtime=nvidia --rm -v SReC:/SReC/ nvcr.io/nvidia/pytorch:19.06-py3 bash
```
Note that `-v [directory on disk]:[directory in container]` mounts SReC directory as `/SReC` inside the container. Running `cd /SReC` in container should get you inside SReC directory.

Now, you cann install all the dependencies besides torchac.
```
pip install -r requirements.txt
```

### Installing torchac

Installing `torchac` requires specific versions of NVCC and GCC. 
Because we use `nvidia-docker`, you should see the following versions of gcc and nvcc:
- GCC 5.4
- NVCC 10.1
You can run `gcc --version` and `nvcc -V` to obtain gcc and nvcc versions. 

Run the following to install torchac:
 ```
 cd torchac
 COMPILE_CUDA=force python3 setup.py install
 ```
This installs a package called `torchac-backend-gpu` in your `pip`. 

To test if it works, you can do
  ```
 cd torchac
 python3 -c "import torchac"
 ```
It should not print any error messages.

## Option 2: Installing without nvidia-docker

### Installing dependencies

First clone the repo. Once inside the repo, you can install all the dependencies besides torchac using `pip`.
```
pip install -r requirements.txt torch==1.2 torchvision==0.2.1
```

### Installing torchac
Installing `torchac` requires specific versions of NVCC and GCC. Note that we used **different** versions of NVCC and GCC than the L3C authors. We used:
- GCC 5.4
- NVCC 10.1

See [L3C](https://github.com/fab-jul/L3C-PyTorch#gpu-and-cpu-support) for other combinations of NVCC and GCC versions that are supported.

To install torchac, run the following:
 ```
 cd torchac
 COMPILE_CUDA=force python3 setup.py install
 ```
This installs a package called `torchac-backend-gpu` in your `pip`. 

To test if it works, you can do
  ```
 cd torchac
 python3 -c "import torchac"
 ```
It should not print any error messages.
