import cv2
import mediapipe as mp
import json
import os

def run_prediction():
    """Run real-time face recognition"""
    MODEL_PATH = "models/lbph_model.xml"
    LABEL_MAP_PATH = "models/label_map.json"
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found")
        print("Please train the model first (Option 2)")
        return
    
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"Error: Label map '{LABEL_MAP_PATH}' not found")
        print("Please train the model first (Option 2)")
        return

    # Load model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    # Load label map
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)

    mp_face = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access camera")
        return

    print("\nStarting face recognition... Press 'Q' to stop")

    with mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as fm:

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
                        except:
                            text = "Unknown"

                        cv2.putText(frame, text, (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                    (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("\nFace recognition stopped")

if __name__ == "__main__":
    run_prediction()
