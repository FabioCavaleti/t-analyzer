FROM ubuntu:22.04

#Instalando pacotes essênciais
RUN apt update && \
    DEBIAN_FRONTEND=noninteractive apt install -y \
    build-essential cmake gdb git python3 python3-pip \
    libopencv-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /project