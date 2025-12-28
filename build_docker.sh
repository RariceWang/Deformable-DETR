#!/bin/bash

# Build the docker image
# You might need to change the tag name to your username/project
IMAGE_NAME="bit:5000/wangyx3_detrv2"

echo "Building docker image ${IMAGE_NAME}..."
docker build -t ${IMAGE_NAME} .

echo "Build complete."
echo "To run the container interactively:"
echo "docker run --gpus all -it --rm ${IMAGE_NAME} /bin/bash"
