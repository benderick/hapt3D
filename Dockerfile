FROM nvidia/cuda:11.7.1-devel-ubuntu22.04
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        git \
        python3-pip \
        python3-dev \
        python3-opencv \
        libglib2.0-0

RUN pip3 install torch==1.12.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6"
##############################################

ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Install dependencies
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

# For faster build, use more jobs.
ENV MAX_JOBS=4
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--blas=openblas"

RUN pip install pytorch-lightning==1.9.0 --no-deps

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip install numpy==1.24.2
RUN pip install matplotlib==3.9.4

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

ARG USER_ID
ARG GROUP_ID

# Switch to same user as host system
RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
USER user

WORKDIR /packages/hapt3d