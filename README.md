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
├── capture.py       # Collects and stores face samples
├── train.py         # Trains the LBPH model
├── predict.py       # Performs face recognition on video input
└── dataset/         # Directory containing captured face images
```

## Installation

Install required dependencies:

```bash
pip install opencv-python mediapipe
pip install opencv-contrib-python
```

The `opencv-contrib-python` package is required for LBPH algorithm support.

## Usage

### Step 1: Capture Face Images

Run the capture script:

```bash
python capture.py
```

When prompted, enter the person's name. Position the face in front of the camera and press Q to stop capturing. Images are automatically saved to `dataset/<name>/`.

### Step 2: Train the Model

Train the LBPH model with collected face data:

```bash
python train.py
```

This generates two files:

- `models/lbph_model.xml` - Trained model weights
- `models/label_map.json` - Mapping of labels to names

### Step 3: Run Face Recognition

Start face recognition on live video:

```bash
python predict.py
```

The camera feed displays:

- Green rectangle around detected faces
- Predicted person name
- LBPH confidence score

Press Q to exit.

## Workflow

The system supports multiple people. Repeat the capture-train-predict cycle as needed to add new identities or improve recognition accuracy.
