FROM bit:5000/ubuntu18.04_cuda10.1_devel_cudnn7

ENV DEBIAN_FRONTEND=noninteractive
# 防止 Python 在 add-apt-repository 时报编码错误
ENV LC_ALL=C.UTF-8 

# =========================================================
# 步骤 1：重置源配置 + 重建目录 + 安装基础工具
# =========================================================
RUN rm -rf /etc/apt/sources.list.d && \
    mkdir -p /etc/apt/sources.list.d && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu/ bionic main restricted universe multiverse" > /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse" >> /etc/apt/sources.list && \
    echo "deb http://mirrors.ustc.edu.cn/ubuntu/ bionic-security main restricted universe multiverse" >> /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    gpg-agent \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* 

# =========================================================
# 步骤 2：安装 Python 3.8
# =========================================================
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    && rm -rf /var/lib/apt/lists/* 

#RUN apt-get autoclean && apt-get autoremove
# =========================================================
# 步骤 3：配置 Python 环境
# =========================================================
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

# 安装 pip
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

# =========================================================
# 步骤 4：安装 Python 依赖
# =========================================================
RUN pip install --no-cache-dir -i https://pypi.mirrors.ustc.edu.cn/simple/ \
    cython \
    numpy \
    scipy \
    tqdm \
    pycocotools

WORKDIR /workspace