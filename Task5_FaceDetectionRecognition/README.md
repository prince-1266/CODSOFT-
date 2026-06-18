# Task 5: Face Detection & Recognition
### CODSOFT AI Internship

An AI application that detects and recognizes faces in images using
**OpenCV Haar Cascade** (detection) and **LBPH Algorithm** (recognition).

---

## How to Run

### Step 1: Install dependencies
```bash
cd D:\CODSOFT\Task5_FaceDetectionRecognition
pip install -r requirements.txt
```

### Step 2: Run the app
```bash
streamlit run app.py
```

### Step 3: Open in browser
```
http://localhost:8501
```

### CLI Commands (Optional)
```bash
python detect.py --image photo.jpg              # Detect faces
python detect.py --image photo.jpg --recognize   # Detect + Recognize
python detect.py --webcam                        # Live webcam
python train.py                                  # Train model from CLI
```

---

## Project Workflow

### Overall Flow
```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Upload      │────▶│  Detect      │────▶│  Draw Bounding   │
│  Image       │     │  Faces       │     │  Boxes on Faces  │
└──────────────┘     └──────┬───────┘     └──────────────────┘
                            │
                     ┌──────▼───────┐     ┌──────────────────┐
                     │  Crop &      │────▶│  Compare with    │
                     │  Extract     │     │  Trained Model   │
                     │  Face        │     │  (LBPH)          │
                     └──────────────┘     └──────┬───────────┘
                                                 │
                                          ┌──────▼───────────┐
                                          │  Return Name     │
                                          │  + Match %       │
                                          └──────────────────┘
```

### Registration & Training Flow
```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Upload      │────▶│  Detect &    │────▶│  Save to         │
│  Face Photos │     │  Crop Face   │     │  data/known_faces│
│  + Name      │     │              │     │  /PersonName/    │
└──────────────┘     └──────────────┘     └──────────────────┘
                                                 │
                                          ┌──────▼───────────┐
│  Click "Train Model"  │───────────────▶ │  LBPH Algorithm  │
                                          │  Learns Patterns │
                                          │  of Each Person  │
                                          └──────┬───────────┘
                                                 │
                                          ┌──────▼───────────┐
                                          │  Save Model to   │
                                          │  models/model.yml│
                                          └──────────────────┘
```

### Face Detection — How It Works
```
Input Image
    │
    ▼
Convert to Grayscale
    │
    ▼
Equalize Histogram (fix lighting)
    │
    ▼
Haar Cascade Classifier scans image
at multiple scales using sliding window
    │
    ▼
Each window is checked against
pre-trained face patterns
(eyes, nose, forehead shape)
    │
    ▼
If pattern matches → Face Found
    │
    ▼
Return coordinates (x, y, width, height)
```

### Face Recognition — How LBPH Works
```
TRAINING PHASE:
    Known Face Image
        │
        ▼
    Divide face into 8x8 grid cells
        │
        ▼
    For each pixel, compare with 8 neighbors:
        neighbor >= pixel → 1
        neighbor <  pixel → 0
    Creates 8-bit binary number (e.g. 10110101)
        │
        ▼
    Build histogram of all binary patterns per cell
        │
        ▼
    Concatenate all cell histograms = Face Fingerprint
        │
        ▼
    Store fingerprint with person's name

RECOGNITION PHASE:
    Unknown Face Image
        │
        ▼
    Compute LBP histogram (same as above)
        │
        ▼
    Compare with all stored fingerprints
    using Chi-Square distance
        │
        ▼
    Lowest distance = Best Match
        │
        ▼
    Return: Person Name + Match Percentage
```

---

## What Each File Does

### Main Files

| File | Purpose |
|------|---------|
| **app.py** | Streamlit web app. Has 4 tabs: Detect, Recognize, Register, People & Train. Handles image uploads, calls detection/recognition, shows results. |
| **detect.py** | CLI script. Run from terminal to detect/recognize faces in image files or live webcam feed. |
| **train.py** | CLI script. Trains the LBPH model from face images stored in `data/known_faces/`. |
| **requirements.txt** | Lists 4 Python packages needed: `opencv-contrib-python`, `streamlit`, `numpy`, `Pillow`. |

### Core AI Modules (`src/`)

| File | Functions | Purpose |
|------|-----------|---------|
| **face_detector.py** | `detect(image)` → returns list of face positions `(x, y, w, h)` | Finds WHERE faces are in the image using Haar Cascade |
| | `extract_face(image, rect)` → returns cropped grayscale face | Crops, resizes, and preprocesses a detected face |
| | `draw_boxes(image, faces, labels)` → returns annotated image | Draws green bounding boxes and name labels on faces |
| **face_recognizer.py** | `train()` → trains model from `data/known_faces/` | Reads all registered face images and trains LBPH model |
| | `recognize(image)` → returns `(annotated_image, results)` | Detects faces and identifies WHO each face belongs to |
| | `register_face(image, name)` → saves face to disk | Extracts face from image and saves it for a person |
| | `get_people()` → returns dict of registered people | Lists all people and their image counts |
| | `delete_person(name)` → deletes person's data | Removes a person and all their face images |
| **utils.py** | `load_image()`, `save_image()`, `resize_image()` | Helper functions for image loading, resizing, and validation |
| | `image_to_base64()`, `base64_to_image()` | Convert between image and base64 format |
| | `allowed_file()`, `validate_image()` | Check if file is a valid image |

### Data Directories

| Directory | Purpose |
|-----------|---------|
| **data/known_faces/** | Training images. Each subfolder = one person. Folder name = person's name. |
| **models/** | Saved model files (`model.yml` + `labels.json`) after training. |
| **.streamlit/** | Streamlit settings (disables animations, deploy button). |

---

## Folder Structure

```
Task5_FaceDetectionRecognition/
│
├── app.py                      ← Main Streamlit web app
├── detect.py                   ← CLI: detect/recognize from terminal
├── train.py                    ← CLI: train model from terminal
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
│
├── src/                        ← Core AI modules
│   ├── __init__.py
│   ├── face_detector.py        ← Haar Cascade face detection
│   ├── face_recognizer.py      ← LBPH face recognition
│   └── utils.py                ← Image helper functions
│
├── data/
│   └── known_faces/            ← Training images by person
│       ├── Person_A/
│       │   ├── img1.jpg
│       │   └── img2.jpg
│       └── Person_B/
│           └── img1.jpg
│
├── models/                     ← Saved trained models
│   ├── model.yml
│   └── labels.json
│
└── .streamlit/
    └── config.toml             ← UI settings
```

---

## Technologies Used

| Technology | Used For |
|-----------|----------|
| Python 3.11 | Programming language |
| OpenCV (Haar Cascade) | Face detection |
| OpenCV (LBPH Algorithm) | Face recognition |
| Streamlit | Web interface |
| NumPy | Image array processing |
| Pillow | Image file handling |

---

## CODSOFT AI Internship — Task 5
