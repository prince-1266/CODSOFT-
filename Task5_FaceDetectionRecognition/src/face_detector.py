"""
Face Detector Module
====================
Detects faces in images using OpenCV Haar Cascade Classifiers.
Uses multiple cascades for better accuracy.
"""

import cv2
import numpy as np


class FaceDetector:
    """Detects faces using Haar Cascade Classifiers."""

    def __init__(self):
        # Load multiple cascades — try each until faces are found
        cascade_names = [
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_frontalface_alt.xml",
            "haarcascade_frontalface_default.xml",
        ]
        self.cascades = []
        for name in cascade_names:
            path = cv2.data.haarcascades + name
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                self.cascades.append(cascade)

    def detect(self, image):
        """
        Detect faces in an image.

        Args:
            image: BGR image (numpy array)

        Returns:
            List of (x, y, w, h) tuples for each face
        """
        if image is None or image.size == 0:
            return []

        h, w = image.shape[:2]

        # Resize if too small (prevents OpenCV assertion error)
        if h < 50 or w < 50:
            scale = max(50 / h, 50 / w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Try each cascade with standard parameters
        for cascade in self.cascades:
            try:
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=3,
                    minSize=(20, 20), flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
            except cv2.error:
                continue

        # Fallback: relaxed parameters
        try:
            faces = self.cascades[0].detectMultiScale(
                gray, scaleFactor=1.02, minNeighbors=2,
                minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE
            )
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        except cv2.error:
            return []

    def extract_face(self, image, rect, size=(200, 200)):
        """Crop, resize, and preprocess a face from the image."""
        x, y, w, h = rect
        img_h, img_w = image.shape[:2]

        # Add 10% padding
        pad = int(0.1 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)

        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return None

        # Convert to grayscale + resize + equalize
        if len(face.shape) == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(face, size)
        face = cv2.equalizeHist(face)
        return face

    def draw_boxes(self, image, faces, labels=None):
        """Draw bounding boxes and labels on the image."""
        output = image.copy()

        for i, (x, y, w, h) in enumerate(faces):
            # Green bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 100), 2)

            # Label
            label = labels[i] if labels and i < len(labels) else f"Face {i+1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)

            # Label background
            cv2.rectangle(output, (x, y - th - 10), (x + tw + 10, y), (0, 255, 100), -1)
            cv2.putText(output, label, (x + 5, y - 5), font, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        return output
