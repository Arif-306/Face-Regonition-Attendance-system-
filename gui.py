
from __future__ import annotations
import numpy as np

# ------------- QImage to numpy array helper -------------
def qimage_to_ndarray(qimage):
    """Convert QImage to numpy array (RGB)."""
    qimage = qimage.convertToFormat(4)  # QImage.Format_RGB32
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    arr = np.array(ptr).reshape(height, width, 4)
    return arr[..., :3]  # Drop alpha channel

import pickle
import shutil
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk

import face_recognition
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

# ----------------- THEME (Emerald & Slate) -----------------
ACCENT = "#22C55E"     # emerald
SKY = "#0EA5E9"        # sky (secondary)
BG = "#111827"         # slate-900
BG2 = "#1F2937"        # slate-800
FG = "#E5E7EB"         # slate-200
WHITE = "#FFFFFF"

def apply_theme(root: tk.Tk):
    """Lightweight ttk theme (clam) with Emerald & Slate colors."""
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # Base
    root.configure(bg=BG)
    style.configure(".", background=BG, foreground=FG, font=("Segoe UI", 10))
    # Frames
    style.configure("TFrame", background=BG)
    style.configure("Card.TFrame", background=BG2)
    # Labels
    style.configure("TLabel", background=BG, foreground=FG)
    style.configure("Title.TLabel", background=BG, foreground=FG, font=("Segoe UI", 14, "bold"))
    style.configure("Chip.TLabel", background=ACCENT, foreground=WHITE, padding=4, font=("Segoe UI", 9, "bold"))
    # Buttons
    style.configure("TButton", background=BG2, foreground=FG, padding=6, relief="flat")
    style.map("TButton",
              background=[("active", "#263040"), ("pressed", "#2a3546")])
    style.configure("Accent.TButton", background=ACCENT, foreground=WHITE, padding=6)
    style.map("Accent.TButton",
              background=[("active", "#1fb455"), ("pressed", "#18a64c")])
    # Checkbutton / Combobox
    style.configure("TCheckbutton", background=BG, foreground=FG)
    style.configure("TCombobox",
                    fieldbackground=BG2, background=BG2, foreground=FG,
                    arrowcolor=FG)
    style.map("TCombobox",
              fieldbackground=[("readonly", BG2)],
              selectbackground=[("readonly", BG2)],
              selectforeground=[("readonly", FG)])

# ----------------- Config -----------------
TOLERANCE = 0.5
FRAME_DOWNSCALE = 0.5
PROCESS_EVERY_N_FRAMES = 2
HOLD_FRAMES = 8

CSV_COLS = ["Date", "Name", "CheckIn", "CheckOut", "DurationMin", "Status"]

SESSION_START = dtime(9, 0, 0)
LATE_AFTER_MIN = 10

PROJECT_DIR = Path(__file__).parent
ENCODINGS_PATH = PROJECT_DIR / "encodings.pickle"
ATTENDANCE_DIR = PROJECT_DIR / "attendance"
ATTENDANCE_DIR.mkdir(exist_ok=True)
UNKNOWN_DIR = PROJECT_DIR / "unknown"
UNKNOWN_DIR.mkdir(exist_ok=True)

# ----------------- Encodings -----------------
def load_encodings(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Encodings file not found: {path}. Run build_encodings.py first.")
    with open(path, "rb") as f:
        data = pickle.load(f)
    encs = data.get("encodings"); names = data.get("names")
    if encs is None or names is None:
        raise ValueError("encodings.pickle missing keys ('encodings','names').")
    return list(encs), list(names)

# ----------------- App -----------------
class FaceAttendanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Face Attendance")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        apply_theme(self)

        # --- Top bar (title + chip) ---
        topbar = ttk.Frame(self)
        topbar.pack(fill=tk.X, padx=12, pady=(12, 6))
        ttk.Label(topbar, text="Face Attendance", style="Title.TLabel").pack(side=tk.LEFT)
        self.chip = ttk.Label(topbar, text="Idle", style="Chip.TLabel")
        self.chip.pack(side=tk.RIGHT)

        # --- Middle layout ---
        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, padx=8 , pady=4)

        # Left controls
        left = ttk.Frame(mid, width=220)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.start_btn = ttk.Button(left, text="Start Camera", style="Accent.TButton", command=self.start_camera)
        self.start_btn.pack(fill=tk.X, pady=4)

        self.stop_btn = ttk.Button(left, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=4)

        self.register_btn = ttk.Button(left, text="Register Person", command=self.register_person)
        self.register_btn.pack(fill=tk.X, pady=8)

        self.capture_unknowns = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Capture Unknowns", variable=self.capture_unknowns).pack(anchor="w", pady=(8,2))

        self.checkout_mode = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text="Checkout Mode", variable=self.checkout_mode).pack(anchor="w", pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)

        ttk.Button(left, text="Export", command=self.export_today_pdf).pack(fill=tk.X, pady=4)

        # Right: video card
        right = ttk.Frame(mid)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        card = ttk.Frame(right, style="Card.TFrame")
        card.pack(fill=tk.BOTH, expand=True)
        # keep tk.Label for PhotoImage stability
        self.video_label = tk.Label(card, bg=BG2)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Bottom bar ---
        bot = ttk.Frame(self)
        bot.pack(fill=tk.X, padx=12, pady=(6, 12))
        ttk.Label(bot, text="Checked-in:").pack(side=tk.LEFT)
        self.active_cb = ttk.Combobox(bot, width=24, state="readonly", values=[])
        self.active_cb.pack(side=tk.LEFT, padx=6)
        ttk.Button(bot, text="Checkout Now", command=self.checkout_selected).pack(side=tk.LEFT, padx=6)

        # Status line
        self.status_label = ttk.Label(self, text="Ready")
        self.status_label.pack(fill=tk.X, padx=12, pady=(0, 12))

        # --- Recognition state ---
        try:
            self.known_encodings, self.known_names = load_encodings(ENCODINGS_PATH)
        except Exception as e:
            messagebox.showwarning("Encodings load", f"Could not load encodings: {e}")
            self.known_encodings, self.known_names = [], []

        self.cap = None
        self._job = None
        self.frame_count = 0
        self.running = False

        # anti-flicker
        self._last_draw: List[Tuple[Tuple[int, int, int, int], str]] = []
        self._hold_frames = HOLD_FRAMES
        self._hold_count = 0
        self._unknown_tick = 0
        self._unknown_save_gap = 10

        # attendance state
        self.today = date.today()
        self.attendance_file = self._attendance_file_for_date(self.today)
        try:
            self._ensure_today_csv_schema()
        except Exception:
            pass

        self.active_today: dict[str, datetime] = {}
        self.marked_names_today: set[str] = set()
        self._load_today_state()

    # ---------------- Helpers ----------------
    def _attendance_file_for_date(self, d: date) -> Path:
        return ATTENDANCE_DIR / f"attendance_{d.isoformat()}.csv"

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
                p.write_text(",".join(CSV_COLS) + "\n", encoding="utf-8")
                return
        if list(df.columns) == ["Date", "Time", "Name", "Status"]:
            df = df.rename(columns={"Time": "CheckIn"})
            df["CheckOut"] = ""; df["DurationMin"] = ""
        for c in CSV_COLS:
            if c not in df.columns: df[c] = ""
        df = df[CSV_COLS]
        df.to_csv(p, index=False)

    def _load_today_state(self):
        self.active_today.clear(); self.marked_names_today.clear()
        p = self.attendance_file
        if not p.exists():
            return
        try:
            df = pd.read_csv(p, dtype=str, engine="python", on_bad_lines="skip")
            df = df[df.get("Date", "") == self.today.isoformat()]
            for _, row in df.iterrows():
                name = str(row.get("Name", "")) or ""
                if not name: continue
                self.marked_names_today.add(name)
                ci = str(row.get("CheckIn", "")); co = str(row.get("CheckOut", ""))
                if ci and (not co or co.lower() == "nan"):
                    try:
                        ci_dt = datetime.strptime(f"{self.today.isoformat()} {ci}", "%Y-%m-%d %H:%M:%S")
                        self.active_today[name] = ci_dt
                    except Exception: pass
        except Exception:
            pass
        self._refresh_active_cb()

    def _refresh_active_cb(self):
        names = sorted(self.active_today.keys())
        self.active_cb["values"] = names
        if names:
            try: self.active_cb.current(0)
            except Exception: pass
        else:
            self.active_cb.set("")

    def _append_log_ui(self, text: str):
        now = datetime.now().strftime("%H:%M:%S")
        self.status_label.config(text=f"{now} - {text}")

    # ---------------- Camera control ----------------
    def start_camera(self):
        if self.running: return
        try:
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not self.cap or not self.cap.isOpened():
                raise RuntimeError("Unable to open camera. Check camera is connected and not in use.")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        except Exception as e:
            messagebox.showerror("Camera error", str(e)); return
        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.chip.config(text="Live")
        self._append_log_ui("Camera started")
        self.frame_count = 0
        self._schedule_frame()

    def stop_camera(self):
        if not self.running: return
        self.running = False
        if self._job:
            try: self.after_cancel(self._job)
            except Exception: pass
            self._job = None
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.chip.config(text="Idle")
        self._append_log_ui("Camera stopped")
        self.video_label.config(image="")

    def _schedule_frame(self):
        self._job = self.after(30, self._update_frame)  # ~30 FPS

    # ---------------- Frame loop ----------------
    def _update_frame(self):
        if not self.running or not self.cap:
            return
        ok, frame = self.cap.read()
        if not ok:
            self._append_log_ui("Failed to read from camera")
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)
        self.frame_count += 1
        self._unknown_tick += 1

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(rgb_frame, (0, 0), fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)

        names_in_frame: List[Tuple[Tuple[int, int, int, int], str]] = []

        if self.frame_count % PROCESS_EVERY_N_FRAMES == 0 and self.known_encodings:
            try:
                locations = face_recognition.face_locations(small_frame, model="hog")
                encodings = face_recognition.face_encodings(small_frame, locations)
            except Exception as e:
                print("Face processing error:", e)
                locations, encodings = [], []

            for face_loc, face_enc in zip(locations, encodings):
                name = "Unknown"
                distances = face_recognition.face_distance(self.known_encodings, face_enc) if self.known_encodings else np.array([])
                if distances.size:
                    best_idx = int(np.argmin(distances))
                    if distances[best_idx] <= TOLERANCE:
                        name = self.known_names[best_idx]
                        try:
                            if self.checkout_mode.get():
                                self._check_out(name)
                            else:
                                self._check_in(name)
                        except Exception as e:
                            print(f"Attendance write error: {e}")
                names_in_frame.append((face_loc, name))

            if names_in_frame:
                self._last_draw = names_in_frame
                self._hold_count = self._hold_frames
            elif self._hold_count > 0 and self._last_draw:
                names_in_frame = self._last_draw
                self._hold_count -= 1
        else:
            if self._hold_count > 0 and self._last_draw:
                names_in_frame = self._last_draw
                self._hold_count -= 1

        # draw overlays
        for (top, right, bottom, left), name in names_in_frame:
            top = int(top / FRAME_DOWNSCALE)
            right = int(right / FRAME_DOWNSCALE)
            bottom = int(bottom / FRAME_DOWNSCALE)
            left = int(left / FRAME_DOWNSCALE)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            label = name if name != "Unknown" else "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, label, (left + 2, bottom - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if name == "Unknown" and self.capture_unknowns.get() and (self._unknown_tick % self._unknown_save_gap == 0):
                face_crop = frame[max(0, top):max(0, bottom), max(0, left):max(0, right)]
                if face_crop.size > 0:
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(str(UNKNOWN_DIR / f"{ts}.jpg"), face_crop)

        # show image
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self._schedule_frame()

    # ---------------- Check-In / Check-Out ----------------
    def _check_in(self, name: str):
        if name == "Unknown" or name in self.active_today:
            return
        now = datetime.now()
        start_dt = datetime.combine(self.today, SESSION_START)
        status = "Late" if now > (start_dt + timedelta(minutes=LATE_AFTER_MIN)) else "Present"
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
                pd.DataFrame(columns=CSV_COLS).to_csv(self.attendance_file, index=False)
                header = False
            pd.DataFrame([row]).to_csv(self.attendance_file, mode="a", header=header, index=False)
            self.active_today[name] = now
            self.marked_names_today.add(name)
            self._refresh_active_cb()
            self._append_log_ui(f"Check-In: {name} ({status})")
        except Exception as e:
            self._append_log_ui(f"Attendance write error: {e}")

    def _check_out(self, name: str):
        if name == "Unknown" or name not in self.active_today:
            return
        out_dt = datetime.now()
        in_dt = self.active_today.get(name, out_dt)
        duration_min = max(0, round((out_dt - in_dt).total_seconds() / 60, 2))
        try:
            if self.attendance_file.exists():
                df = pd.read_csv(self.attendance_file, dtype=str, engine="python", on_bad_lines="skip")
            else:
                df = pd.DataFrame(columns=CSV_COLS)
            for c in CSV_COLS:
                if c not in df.columns: df[c] = ""
            df = df[CSV_COLS]
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
            self._refresh_active_cb()
            self._append_log_ui(f"Check-Out: {name} (Duration {duration_min} min)")
        except Exception as e:
            self._append_log_ui(f"Checkout write error: {e}")

    def checkout_selected(self):
        name = self.active_cb.get().strip()
        if not name:
            messagebox.showinfo("Checkout", "No active person selected.")
            return
        self._check_out(name)

    # ---------------- Register ----------------
    def register_person(self):
        name = simpledialog.askstring("Register", "Person name?")
        if not name:
            return
        save_dir = PROJECT_DIR / "dataset" / name.strip()
        save_dir.mkdir(parents=True, exist_ok=True)

        temp_cap = None
        cap = self.cap
        if not (cap and self.running and cap.isOpened()):
            temp_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW); cap = temp_cap

        captured, target = 0, 20
        while captured < target:
            ok, fr = cap.read()
            if not ok: break
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            locs = face_recognition.face_locations(rgb, model="hog")
            if locs:
                (t, r, b, l) = locs[0]
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
        messagebox.showinfo("Register", f"Saved {captured} images for {name} and updated encodings.")

    def _rebuild_encodings(self):
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        root = PROJECT_DIR / "dataset"
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
                                    new_enc.append(enc); new_names.append(person_dir.name)
                            except Exception:
                                pass
        self.known_encodings, self.known_names = new_enc, new_names
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump({"encodings": new_enc, "names": new_names}, f)

    # ---------------- Export ----------------
    def export_today_pdf(self):
        file = self.attendance_file
        if not file.exists():
            messagebox.showinfo("Export", f"No attendance CSV for today ({file.name}). Nothing to export.")
            return
        try:
            df = pd.read_csv(file, dtype=str, engine="python", on_bad_lines="skip")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to read CSV: {e}")
            return
        if df.empty:
            messagebox.showinfo("Export", "No attendance records for today to export.")
            return

        for c in CSV_COLS:
            if c not in df.columns: df[c] = ""
        df = df[CSV_COLS]

        pdf_path = file.with_suffix(".pdf")
        try:
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
            messagebox.showinfo("Export", f"Exported PDF to:\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("Export error", f"Failed to generate PDF: {e}")

    # ---------------- Close ----------------
    def on_close(self):
        self.stop_camera()
        self.destroy()

def main():
    app = FaceAttendanceApp()
    app.mainloop()

if __name__ == "__main__":
    main()
