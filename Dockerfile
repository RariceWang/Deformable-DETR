# 1. 使用您指定的带有 PyTorch 2.6 的基础镜像
FROM pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# =========================================================
# 步骤 1：更换 APT 源 (Ubuntu 22.04 Jammy)
# =========================================================
# PyTorch 官方镜像基于 Ubuntu 22.04
# 我们先清理旧源，换成 USTC 的 http 源，确保 apt-get update 不会卡死
RUN rm -rf /etc/apt/sources.list.d && \
    mkdir -p /etc/apt/sources.list.d && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu/ jammy-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list

# =========================================================
# 步骤 2：安装系统级依赖
# =========================================================
# pycocotools 和 cv2 (opencv) 通常需要 libgl1 和 git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# =========================================================
# 步骤 3：安装剩余 Python 库
# =========================================================
# 基础镜像里已经有 torch, torchvision, torchaudio, numpy 了
# 我们只需要补齐 pycocotools, tqdm, cython, scipy
# 使用清华源加速，并使用 --no-cache-dir 节省空间
RUN pip install --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple \
    cython \
    scipy \
    tqdm \
    pycocotools

WORKDIR /workspace