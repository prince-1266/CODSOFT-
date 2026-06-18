"""
CLI: Face Detection & Recognition
Run: python detect.py --image photo.jpg
"""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.face_detector import FaceDetector
from src.face_recognizer import FaceRecognizer


def main():
    parser = argparse.ArgumentParser(description="Face Detection & Recognition CLI")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--output", type=str, help="Path to save result image")
    parser.add_argument("--recognize", action="store_true", help="Enable face recognition")
    parser.add_argument("--webcam", action="store_true", help="Use webcam for live detection")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")

    args = parser.parse_args()

    if not args.image and not args.webcam:
        parser.print_help()
        sys.exit(1)

    print("\n" + "=" * 45)
    print("  Face Detection & Recognition")
    print("=" * 45 + "\n")

    detector = FaceDetector()
    recognizer = None

    if args.recognize:
        recognizer = FaceRecognizer(model_dir=args.model_dir)
        if not recognizer.is_trained:
            print("  [WARNING] No trained model. Run train.py first.")
            print("  Falling back to detection only.\n")
            recognizer = None

    if args.webcam:
        run_webcam(detector, recognizer)
    else:
        run_image(args.image, args.output, detector, recognizer)


def run_image(image_path, output_path, detector, recognizer):
    if not os.path.exists(image_path):
        print(f"  [ERROR] Image not found: {image_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"  [ERROR] Could not read image: {image_path}")
        sys.exit(1)

    h, w = image.shape[:2]
    print(f"  Image: {image_path} ({w}x{h})")

    if recognizer:
        annotated, results = recognizer.recognize(image)
        print(f"  Faces found: {len(results)}\n")

        for i, r in enumerate(results):
            print(f"  Face #{i + 1}:")
            print(f"    Name       : {r['name']}")
            print(f"    Match      : {r['match']}%")
            print(f"    Confidence : {r['confidence']}")
            print()
    else:
        faces = detector.detect(image)
        print(f"  Faces found: {len(faces)}\n")

        for i, (x, y, w, h) in enumerate(faces):
            print(f"  Face #{i + 1}: x={x}, y={y}, w={w}, h={h}")

        annotated = detector.draw_boxes(image, faces)

    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"\n  Result saved to: {output_path}")
    else:
        cv2.imshow("Face Detection Result", annotated)
        print("  Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(detector, recognizer):
    print("  Starting webcam... (Press 'q' to quit)\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("  [ERROR] Could not open webcam.")
        sys.exit(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if recognizer:
            annotated, _ = recognizer.recognize(frame)
        else:
            faces = detector.detect(frame)
            annotated = detector.draw_boxes(frame, faces)

        cv2.imshow("Face Detection - Press 'q' to quit", annotated)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("  Webcam closed.")


if __name__ == "__main__":
    main()
