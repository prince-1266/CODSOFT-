"""
Capture Faces from Webcam
=========================
Opens your camera, detects faces, and saves them for registration.

Usage:
    python capture_faces.py --name "Your_Name" --count 10

Controls:
    c     = Capture a face photo
    a     = Auto-capture mode (captures automatically)
    q     = Quit
"""

import argparse
import os
import sys
import cv2
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.face_detector import FaceDetector


def main():
    parser = argparse.ArgumentParser(description="Capture faces from webcam")
    parser.add_argument("--name", type=str, required=True, help="Person's name")
    parser.add_argument("--count", type=int, default=10, help="Number of photos to capture (default: 10)")
    parser.add_argument("--data-dir", type=str, default="data/known_faces", help="Save directory")
    args = parser.parse_args()

    # Create folder for this person
    person_dir = os.path.join(args.data_dir, args.name)
    os.makedirs(person_dir, exist_ok=True)

    existing = len([f for f in os.listdir(person_dir) if f.endswith(".jpg")])

    detector = FaceDetector()

    print("\n" + "=" * 45)
    print(f"  Capturing faces for: {args.name}")
    print(f"  Target: {args.count} photos")
    print("=" * 45)
    print("\n  Controls:")
    print("    c = Capture photo")
    print("    a = Auto-capture mode")
    print("    q = Quit\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Could not open camera.")
        sys.exit(1)

    captured = 0
    auto_mode = False
    last_capture_time = 0

    while captured < args.count:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = detector.detect(frame)
        display = frame.copy()

        face_found = len(faces) > 0

        if face_found:
            # Draw green box on detected face
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            status = f"Face detected | Captured: {captured}/{args.count}"
            color = (0, 255, 0)
        else:
            status = f"No face found | Captured: {captured}/{args.count}"
            color = (0, 0, 255)

        # Mode indicator
        mode_text = "[AUTO MODE]" if auto_mode else "[Press C to capture]"
        cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow(f"Capture Faces - {args.name}", display)

        # Auto capture every 1 second
        should_capture = False
        current_time = time.time()

        if auto_mode and face_found and (current_time - last_capture_time) > 1.0:
            should_capture = True

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("a"):
            auto_mode = not auto_mode
            print(f"  Auto mode: {'ON' if auto_mode else 'OFF'}")
        elif key == ord("c") and face_found:
            should_capture = True

        if should_capture and face_found:
            # Save the largest face
            largest = max(faces, key=lambda f: f[2] * f[3])
            face_img = detector.extract_face(frame, largest)

            if face_img is not None:
                captured += 1
                filename = f"face_{existing + captured:04d}.jpg"
                filepath = os.path.join(person_dir, filename)
                cv2.imwrite(filepath, face_img)
                last_capture_time = current_time
                print(f"  [{captured}/{args.count}] Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    total = existing + captured
    print(f"\n  Done! Captured {captured} photos for {args.name}")
    print(f"  Total images: {total}")
    print(f"  Saved in: {person_dir}")
    print(f"\n  Next step: python train.py\n")


if __name__ == "__main__":
    main()
