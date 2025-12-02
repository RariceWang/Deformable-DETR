FROM bit:5000/ubuntu18.04_cuda11.1_devel_cudnn8

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
# Force CUDA architecture for Ampere (RTX 3090) and others if needed, 
# but PyTorch extension build usually detects it. 
# However, setting TORCH_CUDA_ARCH_LIST helps.
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Alias python to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch and Torchvision (CUDA 11.1)
# PyTorch 1.8.1 is compatible with CUDA 11.1 and supports RTX 3090
RUN pip3 install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# Set working directory
WORKDIR /workspace/Deformable-DETR

# Copy the rest of the project
COPY . .

# Compile custom operators
RUN cd models/ops && sh make.sh
