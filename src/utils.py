"""Utility functions for face recognition pipeline"""

import os
import json
from typing import Dict, List, Tuple
import cv2
import numpy as np


def get_dataset_stats() -> Dict[str, int]:
    """Get statistics about the dataset"""
    DATASET_DIR = "dataset"
    
    if not os.path.exists(DATASET_DIR):
        return {"people": 0, "total_images": 0}
    
    stats = {"people": 0, "total_images": 0}
    
    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue
        
        stats["people"] += 1
        image_count = len([f for f in os.listdir(person_path) 
                          if f.endswith(('.jpg', '.jpeg', '.png'))])
        stats["total_images"] += image_count
    
    return stats


def get_model_info() -> Dict[str, any]:
    """Get information about trained model"""
    MODEL_PATH = "models/lbph_model.xml"
    LABEL_MAP_PATH = "models/label_map.json"
    
    info = {
        "exists": False,
        "people": [],
        "count": 0
    }
    
    if os.path.exists(MODEL_PATH) and os.path.exists(LABEL_MAP_PATH):
        info["exists"] = True
        
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)
            info["people"] = list(label_map.values())
            info["count"] = len(label_map)
    
    return info


def list_people_in_dataset() -> List[str]:
    """List all people in the dataset"""
    DATASET_DIR = "dataset"
    
    if not os.path.exists(DATASET_DIR):
        return []
    
    people = []
    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if os.path.isdir(person_path):
            people.append(person)
    
    return sorted(people)


def validate_environment() -> Tuple[bool, List[str]]:
    """Validate that all dependencies are available"""
    errors = []
    
    try:
        import cv2
    except ImportError:
        errors.append("OpenCV (opencv-python) is not installed")
    
    try:
        import mediapipe
    except ImportError:
        errors.append("MediaPipe is not installed")
    
    try:
        import numpy
    except ImportError:
        errors.append("NumPy is not installed")
    
    # Check for opencv-contrib
    try:
        import cv2
        cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        errors.append("opencv-contrib-python is not installed (required for LBPH)")
    except:
        pass
    
    return len(errors) == 0, errors


def load_dataset(dataset_dir: str) -> Tuple[List[np.ndarray], List[int], Dict[int, str]]:
    """Load dataset images and labels"""
    images = []
    labels = []
    label_map = {}
    current_label = 0

    for person in os.listdir(dataset_dir):
        person_path = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            images.append(img)
            labels.append(current_label)

        current_label += 1

    return images, labels, label_map


def save_model(recognizer, label_map: Dict[int, str], model_dir: str):
    """Save the trained model and label map"""
    os.makedirs(model_dir, exist_ok=True)
    recognizer.save(f"{model_dir}/lbph_model.xml")
    with open(f"{model_dir}/label_map.json", "w") as f:
        json.dump(label_map, f)
