#!/bin/bash

# Definindo o nome da imagem e do repositório
IMAGE_NAME="bt-analyzer"
DOCKER_USERNAME="fcavaleti"
DOCKER_REPO="$DOCKER_USERNAME/$IMAGE_NAME"
TAG="latest"

# Fazer o build da imagem
echo "Building the Docker image..."
docker build -t $IMAGE_NAME .

# Taggear a imagem para o Docker Hub
echo "Tagging the image..."
docker tag $IMAGE_NAME $DOCKER_REPO:$TAG

# Fazer login no Docker Hub (caso não esteja logado)
echo "Logging into Docker Hub..."
docker login 

# Fazer o push da imagem para o Docker Hub
echo "Pushing the image to Docker Hub..."
docker push $DOCKER_REPO:$TAG

echo "Image successfully pushed to Docker Hub as $DOCKER_REPO:$TAG"