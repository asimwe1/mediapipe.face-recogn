"""
predict.py

This module handles real-time face recognition using the trained LBPH model and MediaPipe for face detection.

Functions:
- run_prediction: Loads the trained model and label map, captures video from the webcam, and performs real-time face recognition. The session is recorded and saved as an MP4 file when terminated.
"""

import cv2
import mediapipe as mp
import json
import os

def run_prediction():
    """Run real-time face recognition and record the session."""
    MODEL_PATH = "models/lbph_model.xml"
    LABEL_MAP_PATH = "models/label_map.json"

    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please train the model first.")

        if not os.path.exists(LABEL_MAP_PATH):
            raise FileNotFoundError(f"Label map '{LABEL_MAP_PATH}' not found. Please train the model first.")

        # Load model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)

        # Load label map
        with open(LABEL_MAP_PATH, "r") as f:
            label_map = json.load(f)

        mp_face = mp.solutions.face_mesh

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise RuntimeError("Could not access the camera. Ensure it is connected and not in use by another application.")

        # Ensure the 'recogn' directory exists
        output_dir = 'recogn'
        os.makedirs(output_dir, exist_ok=True)

        # Define video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'{output_dir}/face_recognition_output.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        print("\nStarting face recognition... Press 'Q' to stop")

        with mp_face.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
        ) as fm:

            while True:
                ret, frame = cap.read()

                if not ret:
                    print("Error: Failed to read from camera. Exiting...")
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

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                      (0, 255, 0), 2)

                        # Prepare face for prediction
                        face_crop = frame[y_min:y_max, x_min:x_max]

                        if face_crop.size > 0:
                            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

                            try:
                                label_id, confidence = recognizer.predict(gray)
                                name = label_map[str(label_id)]
                                text = f"{name} ({int(confidence)})"
                            except Exception as e:
                                text = "Unknown"
                                print(f"Prediction error: {e}")

                            cv2.putText(frame, text, (x_min, y_min - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                        (0, 255, 0), 2)

                # Write the frame to the video file
                out.write(frame)

                cv2.imshow("Face Recognition", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Face recognition stopped by user.")
                    break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print("\nVideo saved as 'recogn/face_recognition_output.mp4'")

    except FileNotFoundError as fnf_error:
        print(f"File error: {fnf_error}")
    except RuntimeError as rt_error:
        print(f"Runtime error: {rt_error}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    run_prediction()
