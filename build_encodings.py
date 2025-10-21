"""build_encodings.py

Scans the dataset/ folder for person subfolders with images, computes face encodings using face_recognition,
and writes encodings.pickle containing {'encodings': [...], 'names': [...]}.

Skips images with no detected faces and prints a summary.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple

import face_recognition

# Constants
PROJECT_DIR = Path(__file__).parent
DATASET_DIR = PROJECT_DIR / "dataset"
ENCODINGS_PATH = PROJECT_DIR / "encodings.pickle"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def gather_image_files(dataset_dir: Path) -> List[Tuple[Path, str]]:
    """Return list of (image_path, person_name) tuples found under dataset_dir.

    Supports one level of subfolders: dataset/<person_name>/*.(jpg|png)
    """
    results = []
    if not dataset_dir.exists():
        print(f"Dataset folder not found: {dataset_dir}\nCreate '{dataset_dir}' and add subfolders per person with images.")
        return results

    for person_dir in sorted(dataset_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        person_name = person_dir.name
        for f in sorted(person_dir.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS and f.is_file():
                results.append((f, person_name))
    return results


def main():
    image_list = gather_image_files(DATASET_DIR)
    if not image_list:
        print("No images found in dataset/. Please add person subfolders with images and try again.")
        return

    encodings = []
    names = []

    total = 0
    skipped = 0

    for img_path, person in image_list:
        total += 1
        try:
            image = face_recognition.load_image_file(str(img_path))
            face_locations = face_recognition.face_locations(image, model="hog")
            if not face_locations:
                print(f"[SKIP] No faces found in {img_path}")
                skipped += 1
                continue

            # For dataset images we expect one face per image; encode the first detected face.
            face_enc = face_recognition.face_encodings(image, known_face_locations=face_locations)
            if not face_enc:
                print(f"[SKIP] Unable to encode faces in {img_path}")
                skipped += 1
                continue

            encodings.append(face_enc[0])
            names.append(person)
            print(f"[OK] Encoded {img_path} as {person}")
        except Exception as e:
            skipped += 1
            print(f"[ERROR] {img_path}: {e}")

    # Save to pickle
    try:
        data = {"encodings": encodings, "names": names}
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump(data, f)
        print(f"\nSaved encodings to {ENCODINGS_PATH}")
    except Exception as e:
        print(f"Failed to save encodings: {e}")
        return

    print("\nSummary:")
    print(f"  Total images scanned: {total}")
    print(f"  Encoded images: {len(encodings)}")
    print(f"  Skipped images: {skipped}")
    print(f"  Known persons: {len(set(names))}")


if __name__ == "__main__":
    main()
