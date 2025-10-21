Face Attendance App (Windows)

Overview

A simple face recognition attendance app using OpenCV (camera), face_recognition (dlib) for embeddings, Tkinter for GUI, pandas for CSV attendance, and reportlab for PDF export.

Quick setup (Windows, PowerShell)

1. Create and activate a venv (Python 3.10+)

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

2. Upgrade pip and install requirements

    python -m pip install --upgrade pip
    pip install -r requirements.txt

3. Prepare dataset

Place folders under `dataset/`, each folder name is the person's name, containing images (jpg/png) of that person.

Example:

    dataset\Alice\alice1.jpg
    dataset\Alice\alice2.jpg
    dataset\Bob\bob1.jpg

4. Build encodings (run once or when dataset changes)

    python build_encodings.py

This generates `encodings.pickle` in the project root.

5. Run the GUI

    python main.py

Files

- `build_encodings.py` - scans `dataset/`, computes face encodings and writes `encodings.pickle`.
- `main.py` - Tkinter GUI; Start/Stop camera, recognizes faces and marks attendance in `attendance/attendance_YYYY-MM-DD.csv`, and exports today CSV -> PDF.

Notes

- Tweak `TOLERANCE` and `FRAME_DOWNSCALE` in `main.py` as needed for accuracy/performance.
- If camera or encodings are missing, the app shows clear errors and guidance.
