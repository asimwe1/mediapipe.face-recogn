# Face Recognition Pipeline (MediaPipe + LBPH)

This project implements a classical face recognition system using computer vision techniques without machine learning models. It combines MediaPipe for face detection with OpenCV's LBPH algorithm for feature recognition.

## Overview

The system consists of two core components:

- **MediaPipe Face Mesh** - Face detection and landmark identification
- **OpenCV LBPH (Local Binary Patterns Histograms)** - Face feature extraction and recognition

This approach satisfies the "AI Without ML" assignment requirement by using traditional algorithms.

## Project Structure

```
project/
├── main.py              # Main application entry point
├── src/
│   ├── capture.py       # Face image capture module
│   ├── train.py         # Model training module
│   ├── predict.py       # Face recognition module
│   └── utils.py         # Utility functions
├── tests/               # Test suite
├── dataset/             # Captured face images (generated)
├── models/              # Trained models (generated)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker container definition
├── docker-compose.yml   # Docker Compose configuration
└── Makefile            # Build automation
```

## Installation

### Quick Setup (Linux/Mac)

Run the automated setup script:

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

### Manual Setup

1. Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

The `opencv-contrib-python` package is required for LBPH algorithm support.

## Usage

### Running the Application

Start the main application:

```bash
python main.py
```

This launches an interactive menu with the following options:

1. **Capture Face Images** - Collect face samples for training
2. **Train Model** - Train the LBPH recognizer on captured data
3. **Run Face Recognition** - Perform real-time face recognition
4. **View Dataset Info** - Display statistics about captured data
5. **Exit** - Close the application

### Workflow

1. **Capture**: Enter a person's name and capture face images (press Q to stop)
2. **Train**: Train the model on all captured images
3. **Recognize**: Run real-time face recognition (press Q to stop)

The system supports multiple people. Repeat the capture-train cycle to add new identities or improve accuracy.

### Standalone Module Usage

Each module can also be run independently:

```bash
python -m src.capture   # Capture faces
python -m src.train     # Train model
python -m src.predict   # Run recognition
```

## Docker Deployment

### Quick Start with Docker

Using the automated script:

```bash
chmod +x docker-run.sh
./docker-run.sh
```

### Using Docker Compose

```bash
# Start the application
xhost +local:docker
docker-compose up

# Stop the application
docker-compose down
xhost -local:docker
```

### Manual Docker Commands

```bash
# Build image
docker build -t face-recognition:latest .

# Run container with camera and GUI access
docker run -it --rm \
    --device=/dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v $(pwd)/dataset:/app/dataset \
    -v $(pwd)/models:/app/models \
    --privileged \
    face-recognition:latest
```

### Using Makefile

```bash
make help           # Show all available commands
make setup          # Setup local environment
make run            # Run locally
make build          # Build Docker image
make docker-run     # Run in Docker
make clean          # Clean up
```

## Requirements

- Python 3.11+
- Webcam/Camera device
- X11 display server (for GUI on Linux)
- Docker (optional, for containerized deployment)
