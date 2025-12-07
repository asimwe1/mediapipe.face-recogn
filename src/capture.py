"""
capture.py

This module handles the capture of face images from the webcam. The captured images are saved to the dataset directory.

Functions:
- run_capture: Captures face images, processes them using MediaPipe, and saves them to the dataset.
"""

import cv2
import mediapipe as mp
import os

def run_capture():
    """Capture face images from webcam"""
    name = input("Enter your name: ").strip()
    
    if not name:
        print("Error: Name cannot be empty")
        return

    save_dir = f"dataset/{name}"
    os.makedirs(save_dir, exist_ok=True)

    # Count existing images in the dataset directory
    existing_images = len([f for f in os.listdir(save_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
    count = existing_images

    mp_face = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access camera")
        return

    with mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as fm:

        print("\nCapturing faces... Press 'Q' to stop")
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to read from camera")
                break
                
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fm.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    xs = [int(lm.x * w) for lm in face_landmarks.landmark]
                    ys = [int(lm.y * h) for lm in face_landmarks.landmark]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)

                    # Draw rectangle
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                  (0, 255, 0), 2)

                    # Crop & save
                    face_crop = frame[y_min:y_max, x_min:x_max]
                    if face_crop.size > 0:
                        cv2.imwrite(f"{save_dir}/{count}.jpg", face_crop)
                        count += 1

            cv2.imshow("Capturing Faces", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Capturing process terminated by user.")
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nSaved {count - existing_images} new images to {save_dir}")

if __name__ == "__main__":
    run_capture()
