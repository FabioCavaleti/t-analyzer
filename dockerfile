FROM ubuntu:22.04

#Instalando pacotes essênciais
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    build-essential cmake gdb git python3 python3-pip \
    libopencv-dev nlohmann-json3-dev \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Instalando pacotes Python
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    numpy \
    opencv-python-headless \
    loguru \
    ultralytics \ 
    onnxruntime \
    tflite-runtime 

# using nvidia gpu inside docker, comment this file if you dont have one.
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

WORKDIR /project