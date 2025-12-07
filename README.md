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
