"""
Face Recognizer Module
======================
Trains and recognizes faces using LBPH algorithm.
Handles model training, saving, loading, and prediction.
"""

import cv2
import numpy as np
import os
import json
from src.face_detector import FaceDetector


class FaceRecognizer:
    """Recognizes faces using LBPH (Local Binary Patterns Histograms)."""

    FACE_SIZE = (200, 200)

    def __init__(self, data_dir="data/known_faces", model_dir="models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        self.detector = FaceDetector()
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=200.0
        )
        self.label_map = {}     # id → name
        self.is_trained = False

        # Load existing model if available
        self._load_model()

    def train(self):
        """
        Train from images in data/known_faces/.
        Each subfolder = one person. Folder name = person name.
        """
        faces = []
        labels = []
        self.label_map = {}
        current_id = 0
        stats = {}

        for person_name in sorted(os.listdir(self.data_dir)):
            person_dir = os.path.join(self.data_dir, person_name)
            if not os.path.isdir(person_dir) or person_name.startswith("."):
                continue

            person_faces = []

            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    continue

                img_path = os.path.join(person_dir, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # Try to detect a face
                detected = self.detector.detect(image)

                if len(detected) > 0:
                    rect = max(detected, key=lambda f: f[2] * f[3])
                    face = self.detector.extract_face(image, rect, self.FACE_SIZE)
                else:
                    # Use entire image as face
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(gray, self.FACE_SIZE)
                    face = cv2.equalizeHist(face)

                if face is not None:
                    person_faces.append(face)

            # Only add this person if they have at least 1 face
            if len(person_faces) > 0:
                self.label_map[current_id] = person_name
                for face in person_faces:
                    faces.append(face)
                    labels.append(current_id)
                stats[person_name] = len(person_faces)
                current_id += 1

        if len(faces) < 2:
            return None, "Need at least 2 face images to train."

        if len(stats) < 2:
            return None, "Need at least 2 different people to train."

        # Recreate recognizer to clear any old state
        self.recognizer = cv2.face.LBPHFaceRecognizer_create(
            radius=1, neighbors=8, grid_x=8, grid_y=8, threshold=200.0
        )

        try:
            self.recognizer.train(faces, np.array(labels))
            self.is_trained = True
            self._save_model()
        except Exception as e:
            return None, f"Training error: {str(e)}"

        return stats, "Model trained successfully!"

    def recognize(self, image):
        """
        Detect and recognize faces in an image.

        Returns:
            (annotated_image, results_list)
        """
        faces = self.detector.detect(image)

        if len(faces) == 0:
            return image.copy(), []

        results = []
        labels = []

        for rect in faces:
            if self.is_trained:
                face_img = self.detector.extract_face(image, rect, self.FACE_SIZE)
                if face_img is not None:
                    try:
                        label_id, confidence = self.recognizer.predict(face_img)
                    except Exception:
                        results.append({"name": "Unknown", "match": 0, "confidence": 999})
                        labels.append("Unknown")
                        continue

                    if confidence < 150:
                        name = self.label_map.get(label_id, "Unknown")
                        match = max(0, round((150 - confidence) / 150 * 100, 1))
                    else:
                        name = "Unknown"
                        match = 0.0

                    results.append({"name": name, "match": match, "confidence": round(confidence, 1)})
                    labels.append(f"{name} ({match}%)")
                else:
                    results.append({"name": "Unknown", "match": 0, "confidence": 999})
                    labels.append("Unknown")
            else:
                results.append({"name": "Detected", "match": 0, "confidence": 0})
                labels.append("Face Detected")

        annotated = self.detector.draw_boxes(image, faces, labels)
        return annotated, results

    def register_face(self, image, person_name):
        """Save a face image for a person."""
        person_dir = os.path.join(self.data_dir, person_name)
        os.makedirs(person_dir, exist_ok=True)

        faces = self.detector.detect(image)
        if len(faces) == 0:
            return False, "No face detected in the image."

        rect = max(faces, key=lambda f: f[2] * f[3])
        face = self.detector.extract_face(image, rect, self.FACE_SIZE)
        if face is None:
            return False, "Failed to extract face."

        existing = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])
        filepath = os.path.join(person_dir, f"face_{existing + 1:04d}.jpg")
        cv2.imwrite(filepath, face)

        count = existing + 1
        return True, f"Face saved for '{person_name}' ({count} total images)."

    def get_people(self):
        """Get list of registered people."""
        people = {}
        if not os.path.exists(self.data_dir):
            return people
        for name in sorted(os.listdir(self.data_dir)):
            d = os.path.join(self.data_dir, name)
            if os.path.isdir(d) and not name.startswith("."):
                count = len([f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
                if count > 0:
                    people[name] = count
        return people

    def delete_person(self, name):
        """Delete a person's data."""
        import shutil
        person_dir = os.path.join(self.data_dir, name)
        if os.path.exists(person_dir):
            shutil.rmtree(person_dir)
            return True
        return False

    def _save_model(self):
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, "model.yml")
        labels_path = os.path.join(self.model_dir, "labels.json")
        try:
            self.recognizer.save(model_path)
            with open(labels_path, "w") as f:
                json.dump({str(k): v for k, v in self.label_map.items()}, f)
        except Exception as e:
            print(f"Error saving model: {e}")

    def _load_model(self):
        model_path = os.path.join(self.model_dir, "model.yml")
        labels_path = os.path.join(self.model_dir, "labels.json")
        if os.path.exists(model_path) and os.path.exists(labels_path):
            try:
                self.recognizer.read(model_path)
                with open(labels_path) as f:
                    self.label_map = {int(k): v for k, v in json.load(f).items()}
                self.is_trained = True
            except Exception:
                self.is_trained = False

