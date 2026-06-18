"""
CLI: Train Face Recognition Model
Run: python train.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.face_recognizer import FaceRecognizer


def main():
    parser = argparse.ArgumentParser(description="Train Face Recognition Model")
    parser.add_argument("--data-dir", type=str, default="data/known_faces", help="Face images directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model save directory")
    args = parser.parse_args()

    print("\n" + "=" * 45)
    print("  Face Recognition — Model Training")
    print("=" * 45)
    print(f"\n  Data dir  : {os.path.abspath(args.data_dir)}")
    print(f"  Model dir : {os.path.abspath(args.model_dir)}")

    if not os.path.exists(args.data_dir):
        print(f"\n  [ERROR] Data directory not found: {args.data_dir}")
        sys.exit(1)

    recognizer = FaceRecognizer(data_dir=args.data_dir, model_dir=args.model_dir)

    # Show registered people
    people = recognizer.get_people()

    if not people:
        print("\n  [ERROR] No person folders with images found.")
        print(f"  Add folders with face images to: {args.data_dir}")
        sys.exit(1)

    print(f"\n  Found {len(people)} person(s):")
    total_images = 0
    for name, count in people.items():
        print(f"    - {name}: {count} image(s)")
        total_images += count

    print(f"\n  Training LBPH model...\n")

    # Train
    stats, message = recognizer.train()

    print("\n" + "-" * 45)
    print("  Training Summary")
    print("-" * 45)

    if stats:
        total_faces = sum(stats.values())
        print(f"  People trained     : {len(stats)}")
        print(f"  Total faces used   : {total_faces}")
        for name, count in stats.items():
            print(f"    - {name}: {count} face(s)")
        print(f"\n  [OK] {message}")
    else:
        print(f"  [FAILED] {message}")

    print("=" * 45 + "\n")


if __name__ == "__main__":
    main()
