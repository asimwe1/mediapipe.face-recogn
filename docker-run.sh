#!/bin/bash
# Script to run the Face Recognition Pipeline in Docker with GUI support

echo "=================================="
echo "Face Recognition Pipeline - Docker"
echo "=================================="
echo ""

# Allow X11 forwarding for GUI
echo "Setting up X11 forwarding..."
xhost +local:docker

# Build the Docker image
echo ""
echo "Building Docker image..."
docker build -t face-recognition:latest .

if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

# Run the container
echo ""
echo "Starting container..."
docker run -it --rm \
    --name face-recognition-pipeline \
    --device=/dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/dataset:/app/dataset \
    -v $(pwd)/models:/app/models \
    --privileged \
    face-recognition:latest

# Cleanup X11 permissions
echo ""
echo "Cleaning up X11 permissions..."
xhost -local:docker

echo ""
echo "Container stopped."
