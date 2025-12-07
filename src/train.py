import cv2
import os
import json
import numpy as np

def run_training():
    """Train LBPH model on captured face images"""
    DATASET_DIR = "dataset"
    MODEL_DIR = "models"
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found")
        print("Please capture face images first (Option 1)")
        return
    
    os.makedirs(MODEL_DIR, exist_ok=True)

    images = []
    labels = []
    label_map = {}
    current_label = 0

    # Load dataset
    print("\nLoading dataset...")
    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path):
            continue

        label_map[current_label] = person
        person_images = 0

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            images.append(img)
            labels.append(current_label)
            person_images += 1

        print(f"  - Loaded {person_images} images for '{person}'")
        current_label += 1

    if len(images) == 0:
        print("\nError: No valid images found in dataset")
        print("Please capture face images first (Option 1)")
        return

    labels = np.array(labels)

    # Train LBPH model
    print("\nTraining LBPH model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, labels)

    # Save model
    recognizer.save(f"{MODEL_DIR}/lbph_model.xml")

    # Save label map
    with open(f"{MODEL_DIR}/label_map.json", "w") as f:
        json.dump(label_map, f)

    print("\nTraining completed!")
    print(f"  - Total images trained: {len(images)}")
    print(f"  - Total people: {len(label_map)}")
    print(f"  - Model saved to: {MODEL_DIR}/lbph_model.xml")
    print(f"  - Label map saved to: {MODEL_DIR}/label_map.json")

if __name__ == "__main__":
    run_training()
