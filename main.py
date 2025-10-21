"""
attendance_core.py — core logic for face-attendance
- Face recognition (HOG) + anti-flicker
- Check-In / Check-Out with robust CSV schema
- Pandas 2.x safe I/O (no DataFrame.append)
- Register person, export PDF
"""

from __future__ import annotations

import pickle
import shutil
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from typing import Callable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import face_recognition
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle


class AttendanceService:
    # ---------- Config ----------
    TOLERANCE = 0.5                 # smaller -> stricter match
    FRAME_DOWNSCALE = 0.5           # 0.5 = half-size processing for speed
    PROCESS_EVERY_N_FRAMES = 2      # run recognition every Nth frame
    HOLD_FRAMES = 8                 # draw persistence to prevent flicker

    # Attendance policy (for "Late")
    SESSION_START = dtime(9, 0, 0)  # 09:00
    LATE_AFTER_MIN = 10             # late if after 09:10

    # CSV schema (new format)
    CSV_COLS = ["Date", "Name", "CheckIn", "CheckOut", "DurationMin", "Status"]

    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir)
        self.encodings_path = self.project_dir / "encodings.pickle"
        self.attendance_dir = self.project_dir / "attendance"
        self.attendance_dir.mkdir(exist_ok=True)
        self.unknown_dir = self.project_dir / "unknown"
        self.unknown_dir.mkdir(exist_ok=True)

        # callbacks (GUI will set these)
        self.on_status: Callable[[str], None] | None = None
        self.on_active_changed: Callable[[List[str]], None] | None = None

        # recognition state
        self.known_encodings, self.known_names = self._load_encodings(self.encodings_path)
        self.cap = None
        self.frame_count = 0
        self.running = False

        # anti-flicker
        self._last_draw: List[Tuple[Tuple[int, int, int, int], str]] = []
        self._hold_count = 0
        self._unknown_tick = 0
        self._unknown_save_gap = 10

        # options toggled by GUI
        self.capture_unknowns: bool = True
        self.checkout_mode: bool = False

        # attendance state
        self.today = date.today()
        self.attendance_file = self.attendance_dir / f"attendance_{self.today.isoformat()}.csv"
        self._ensure_today_csv_schema()
        # active_today: Name -> checkin datetime
        self.active_today: dict[str, datetime] = {}
        # names that have any row today
        self.marked_names_today: set[str] = set()
        self._load_today_state()

    # ---------- public API for GUI ----------
    def set_checkout_mode(self, val: bool) -> None:
        self.checkout_mode = bool(val)

    def set_capture_unknowns(self, val: bool) -> None:
        self.capture_unknowns = bool(val)

    def get_active_names(self) -> List[str]:
        return sorted(self.active_today.keys())

    def start_camera(self) -> None:
        if self.running:
            return
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not self.cap or not self.cap.isOpened():
            raise RuntimeError("Unable to open camera. Check camera is connected and not in use.")
        # light-ish defaults
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.running = True
        self.frame_count = 0
        self._status("Camera started")

    def stop_camera(self) -> None:
        self.running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self._status("Camera stopped")

    def next_frame(self) -> Image.Image | None:
        """Run one iteration: read frame, recognize (periodically), draw overlays.
        Returns a PIL.Image (RGB) to display, or None if camera not running."""
        if not (self.running and self.cap):
            return None

        ok, frame = self.cap.read()
        if not ok:
            self._status("Failed to read from camera")
            self.stop_camera()
            return None

        frame = cv2.flip(frame, 1)  # selfie mirror
        self.frame_count += 1
        self._unknown_tick += 1

        # prepare
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (0, 0), fx=self.FRAME_DOWNSCALE, fy=self.FRAME_DOWNSCALE)
        names_in_frame: List[Tuple[Tuple[int, int, int, int], str]] = []

        # recognition on Nth frame
        if self.frame_count % self.PROCESS_EVERY_N_FRAMES == 0 and self.known_encodings:
            try:
                locs = face_recognition.face_locations(small, model="hog")
                encs = face_recognition.face_encodings(small, locs)
            except Exception as e:
                print("Face processing error:", e)
                locs, encs = [], []

            for loc, enc in zip(locs, encs):
                name = "Unknown"
                distances = face_recognition.face_distance(self.known_encodings, enc) \
                    if self.known_encodings else np.array([])
                if distances.size:
                    best = int(np.argmin(distances))
                    if distances[best] <= self.TOLERANCE:
                        name = self.known_names[best]
                        try:
                            if self.checkout_mode:
                                self._check_out(name)
                            else:
                                self._check_in(name)
                        except Exception as e:
                            print("Attendance write error:", e)
                names_in_frame.append((loc, name))

            if names_in_frame:
                self._last_draw = names_in_frame
                self._hold_count = self.HOLD_FRAMES
            else:
                if self._hold_count > 0 and self._last_draw:
                    names_in_frame = self._last_draw
                    self._hold_count -= 1
        else:
            if self._hold_count > 0 and self._last_draw:
                names_in_frame = self._last_draw
                self._hold_count -= 1

        # draw overlays on original BGR frame
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for (t, r, b, l), name in names_in_frame:
            t = int(t / self.FRAME_DOWNSCALE); r = int(r / self.FRAME_DOWNSCALE)
            b = int(b / self.FRAME_DOWNSCALE); l = int(l / self.FRAME_DOWNSCALE)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = name if name != "Unknown" else "Unknown"
            cv2.rectangle(bgr, (l, t), (r, b), color, 2)
            cv2.rectangle(bgr, (l, b - 20), (r, b), color, cv2.FILLED)
            cv2.putText(bgr, label, (l + 2, b - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # optional: save unknown crops
            if name == "Unknown" and self.capture_unknowns and (self._unknown_tick % self._unknown_save_gap == 0):
                crop = bgr[max(0, t):max(0, b), max(0, l):max(0, r)]
                if crop.size > 0:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(str(self.unknown_dir / f"{ts}.jpg"), crop)

        # convert to PIL RGB for GUI
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    def checkout_selected(self, name: str) -> None:
        self._check_out(name.strip())

    def register_person(self, name: str, target: int = 20) -> int:
        """Capture `target` face crops into dataset/<name>/ and rebuild encodings.
        Returns number of images captured."""
        name = name.strip()
        if not name:
            return 0
        save_dir = self.project_dir / "dataset" / name
        save_dir.mkdir(parents=True, exist_ok=True)

        # use current camera if running; else open a temporary one
        temp_cap = None
        cap = self.cap
        if not (cap and self.running and cap.isOpened()):
            temp_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap = temp_cap

        captured = 0
        while captured < target:
            ok, fr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            if locs:
                t, r, b, l = locs[0]
                face = fr[t:b, l:r]
                if face.size > 0:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(str(save_dir / f"{name}_{ts}.jpg"), face)
                    captured += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if temp_cap is not None:
            temp_cap.release()

        self._rebuild_encodings()
        self._status(f"Registered {name}: saved {captured} images and updated encodings.")
        return captured

    def export_today_pdf(self) -> Path:
        file = self.attendance_file
        if not file.exists():
            raise FileNotFoundError(f"No attendance CSV for today ({file.name}).")
        df = pd.read_csv(file, dtype=str, engine="python", on_bad_lines="skip")
        if df.empty:
            raise ValueError("No attendance records for today to export.")

        # ensure order
        for c in self.CSV_COLS:
            if c not in df.columns: df[c] = ""
        df = df[self.CSV_COLS]

        pdf_path = file.with_suffix(".pdf")
        doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
        data = [list(df.columns)] + df.astype(str).values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        doc.build([table])
        self._status(f"Exported PDF to: {pdf_path}")
        return pdf_path

    # ---------- internals ----------
    def _status(self, text: str) -> None:
        if self.on_status:
            now = datetime.now().strftime("%H:%M:%S")
            self.on_status(f"{now} - {text}")

    def _notify_active(self) -> None:
        if self.on_active_changed:
            self.on_active_changed(self.get_active_names())

    @staticmethod
    def _load_encodings(path: Path):
        if not path.exists():
            return [], []
        with open(path, "rb") as f:
            data = pickle.load(f)
        return list(data.get("encodings", [])), list(data.get("names", []))

    def _ensure_today_csv_schema(self):
        p = self.attendance_file
        if not p.exists():
            return
        try:
            df = pd.read_csv(p, dtype=str, engine="python")
        except Exception:
            try:
                df = pd.read_csv(p, dtype=str, engine="python", on_bad_lines="skip")
            except Exception:
                backup = p.with_suffix(".corrupt.bak.csv")
                try: shutil.copy2(p, backup)
                except Exception: pass
                p.write_text(",".join(self.CSV_COLS) + "\n", encoding="utf-8")
                return

        if list(df.columns) == ["Date", "Time", "Name", "Status"]:
            df = df.rename(columns={"Time": "CheckIn"})
            df["CheckOut"] = ""
            df["DurationMin"] = ""

        for c in self.CSV_COLS:
            if c not in df.columns:
                df[c] = ""
        df = df[self.CSV_COLS]
        df.to_csv(p, index=False)

    def _load_today_state(self):
        self.active_today.clear()
        self.marked_names_today.clear()
        p = self.attendance_file
        if not p.exists():
            return
        try:
            df = pd.read_csv(p, dtype=str, engine="python", on_bad_lines="skip")
            df = df[df.get("Date", "") == self.today.isoformat()]
            for _, row in df.iterrows():
                name = str(row.get("Name", ""))
                if not name:
                    continue
                self.marked_names_today.add(name)
                ci = str(row.get("CheckIn", ""))
                co = str(row.get("CheckOut", ""))
                if ci and (not co or co.lower() == "nan"):
                    try:
                        ci_dt = datetime.strptime(f"{self.today.isoformat()} {ci}", "%Y-%m-%d %H:%M:%S")
                        self.active_today[name] = ci_dt
                    except Exception:
                        pass
        except Exception:
            pass
        self._notify_active()

    def _check_in(self, name: str) -> None:
        if name == "Unknown" or name in self.active_today:
            return
        now = datetime.now()
        start_dt = datetime.combine(self.today, self.SESSION_START)
        status = "Late" if now > (start_dt + timedelta(minutes=self.LATE_AFTER_MIN)) else "Present"

        row = {
            "Date": self.today.isoformat(),
            "Name": name,
            "CheckIn": now.strftime("%H:%M:%S"),
            "CheckOut": "",
            "DurationMin": "",
            "Status": status,
        }
        try:
            header = not self.attendance_file.exists() or self.attendance_file.stat().st_size == 0
            if header:
                # ensure header exists and order is correct
                pd.DataFrame(columns=self.CSV_COLS).to_csv(self.attendance_file, index=False)
                header = False
            pd.DataFrame([row]).to_csv(self.attendance_file, mode="a", header=header, index=False)
            self.active_today[name] = now
            self.marked_names_today.add(name)
            self._notify_active()
            self._status(f"Check-In: {name} ({status})")
        except Exception as e:
            self._status(f"Attendance write error: {e}")

    def _check_out(self, name: str) -> None:
        if name == "Unknown" or name not in self.active_today:
            return

        out_dt = datetime.now()
        in_dt = self.active_today.get(name, out_dt)
        duration_min = max(0, round((out_dt - in_dt).total_seconds() / 60, 2))

        try:
            if self.attendance_file.exists():
                df = pd.read_csv(self.attendance_file, dtype=str, engine="python", on_bad_lines="skip")
            else:
                df = pd.DataFrame(columns=self.CSV_COLS)

            for c in self.CSV_COLS:
                if c not in df.columns:
                    df[c] = ""
            df = df[self.CSV_COLS]

            mask = (
                (df["Date"] == self.today.isoformat()) &
                (df["Name"] == name) &
                ((df["CheckOut"] == "") | (df["CheckOut"].isna()))
            )
            if mask.any():
                idx = df[mask].index[-1]
                df.at[idx, "CheckOut"] = out_dt.strftime("%H:%M:%S")
                df.at[idx, "DurationMin"] = str(duration_min)
            else:
                new_row = {
                    "Date": self.today.isoformat(),
                    "Name": name,
                    "CheckIn": in_dt.strftime("%H:%M:%S"),
                    "CheckOut": out_dt.strftime("%H:%M:%S"),
                    "DurationMin": str(duration_min),
                    "Status": "Present",
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            df.to_csv(self.attendance_file, index=False)
            self.active_today.pop(name, None)
            self._notify_active()
            self._status(f"Check-Out: {name} (Duration {duration_min} min)")
        except Exception as e:
            self._status(f"Checkout write error: {e}")

    def _rebuild_encodings(self) -> None:
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        root = self.project_dir / "dataset"
        new_enc, new_names = [], []
        if root.exists():
            for person_dir in root.iterdir():
                if person_dir.is_dir():
                    for img in person_dir.glob("*"):
                        if img.suffix.lower() in IMAGE_EXTS:
                            try:
                                image = face_recognition.load_image_file(str(img))
                                boxes = face_recognition.face_locations(image, model="hog")
                                encs = face_recognition.face_encodings(image, boxes)
                                for enc in encs:
                                    new_enc.append(enc)
                                    new_names.append(person_dir.name)
                            except Exception:
                                pass
        self.known_encodings, self.known_names = new_enc, new_names
        with open(self.encodings_path, "wb") as f:
            pickle.dump({"encodings": new_enc, "names": new_names}, f)
