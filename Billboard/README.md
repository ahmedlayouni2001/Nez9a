# Billboard Intelligence — CV Module

This folder is the **computer vision (CV) module** of the 3SG Billboard Intelligence platform.
It sits alongside the `frontend/` and `backend/` folders in the full project.

```
project-root/
├── frontend/          # Web dashboard
├── backend/           # API server (FastAPI + WebSocket)
└── Billboard/         # ← this folder (CV module)
    ├── roi_drawer.py
    ├── test.py
    ├── cv_engine.py
    ├── roi_config.txt
    ├── session_report.json
    ├── yolov8n.pt
    └── requirements.txt
```

---

## What this module does

Given a video file (or live camera), it:

1. **Counts people** — YOLOv8 detects and tracks every person frame by frame. Each new tracking ID adds 1 to the count (cumulative, never resets).
2. **Detects billboard fraud** — Takes a perceptual hash snapshot of the billboard region at frame 30 (the reference). Every 2 seconds it re-hashes the same region and compares. If the visual difference exceeds the threshold, fraud is declared and the exact timestamp is recorded.
3. **Produces a live terminal table** — Every second it prints a row with: timestamp, people seen that second, fraud status.
4. **Produces a session report** — When the video ends, it saves `session_report.json` with the full per-second log and a final summary.
5. **Sends metrics to the backend** — Results are pushed over WebSocket to `ws://localhost:8000/ws/cv` so the frontend dashboard can display them in real time.

---

## Files

| File | Role |
|---|---|
| `roi_drawer.py` | Interactive tool to draw the billboard rectangle on the video |
| `test.py` | Main CV pipeline — run this to process a video |
| `cv_engine.py` | Original full engine (includes ad overlay simulation) |
| `roi_config.txt` | Saved ROI coordinates (written by `roi_drawer.py`, read by `test.py`) |
| `session_report.json` | Output report generated after each run |
| `yolov8n.pt` | YOLOv8 nano model weights (auto-downloaded on first run if missing) |
| `requirements.txt` | Python dependencies |

---

## Setup

```bash
cd Billboard
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## Step 1 — Define the billboard region (ROI)

Run `roi_drawer.py` to draw a rectangle over the billboard area in your video:

```bash
python roi_drawer.py
```

- Click and drag to draw the rectangle over the billboard
- Press **`p`** to save the coordinates → writes `roi_config.txt`
- Press **`r`** to redraw
- Press **`q`** to quit

The terminal will print the 4 corner points and the exact `BILLBOARD_ROI` value.

> This step is required before running `test.py`. If `roi_config.txt` is missing, `test.py` uses the last hardcoded fallback value.

---

## Step 2 — Run the CV pipeline

```bash
python test.py
```

By default it reads the video path hardcoded at the top of the file. You can also pass arguments:

```bash
python test.py --source path/to/video.mp4
python test.py --source 0                        # webcam
python test.py --source 0 --billboard-id BB_001
python test.py --no-window                       # headless, no display
```

### What you see during the run

**Video window** — shows the processed video with:
- A green rectangle around the billboard ROI (turns red when fraud is detected)
- `People: N` — number of people seen in the current second
- `Fraud: NOTHING / FRAUD` — current fraud status
- `Since: YYYY-MM-DD HH:MM:SS` — timestamp of when fraud was first detected

**Terminal** — a table printed every second:

```
+---------------------+---------+--------------+
| Timestamp           | Persons | Fraud Status |
+---------------------+---------+--------------+
| 2026-05-02 14:32:01 |       3 | NOTHING      |
+---------------------+---------+--------------+
```

When fraud is first detected, an alert is also printed:
```
[FraudDetector] *** FRAUD DETECTED at 2026-05-02 14:32:55 ***
```

### What you get at the end

**Terminal summary:**
```
+-------------------+------------------------+
| SESSION SUMMARY                            |
+-------------------+------------------------+
| Video start       | 2026-05-02 14:32:00    |
| Video end         | 2026-05-02 14:33:45    |
| Total people      | 47                     |
| Fraud detected at | 2026-05-02 14:32:55    |
+-------------------+------------------------+
```

**`session_report.json`:**
```json
{
  "summary": {
    "video_start": "2026-05-02 14:32:00",
    "video_end": "2026-05-02 14:33:45",
    "total_people": 47,
    "fraud_detected_at": "2026-05-02 14:32:55"
  },
  "per_second": [
    { "timestamp": "2026-05-02 14:32:01", "persons": 3, "fraud_status": "nothing" },
    { "timestamp": "2026-05-02 14:32:02", "persons": 5, "fraud_status": "nothing" },
    ...
  ]
}
```

---

## Key configuration values

All tunable constants are at the top of `test.py`:

| Constant | Default | Meaning |
|---|---|---|
| `YOLO_CONF_THRESHOLD` | `0.4` | Minimum YOLO detection confidence |
| `PROCESS_EVERY_N_FRAMES` | `3` | Run YOLO on 1 out of every N frames (higher = faster) |
| `FRAUD_CHECK_INTERVAL_SEC` | `2` | How often to re-hash the billboard region |
| `FRAUD_HASH_THRESHOLD` | `15` | Hamming distance above which fraud is declared (higher = less sensitive) |
| `PUSH_INTERVAL_SEC` | `1.0` | How often to print the terminal table and push to WebSocket |

---

## Integration with the backend

`test.py` connects to the backend WebSocket at `ws://localhost:8000/ws/cv`.
Each second it sends a JSON message:

```json
{ "timestamp": "...", "persons": 3, "fraud_status": "nothing" }
```

The backend must be running before `test.py` starts, otherwise it falls back to standalone mode (terminal output only).

---

## How fraud detection works

1. At **frame 30** (~1 second in), a perceptual hash of the billboard ROI is saved as the **reference**.
2. Every `FRAUD_CHECK_INTERVAL_SEC` seconds, a new hash of the same region is computed and compared to the reference using Hamming distance.
3. If `distance > FRAUD_HASH_THRESHOLD` → status becomes `FRAUD` and the timestamp is locked.
4. The status and timestamp are included in every per-second log entry and in the final JSON report.

> `Fraud detected at: None` in the report means the billboard content did not change during the session — this is normal when there is no actual swap.

---

## How people counting works

- YOLOv8 assigns a **persistent tracking ID** to each detected person.
- Each new ID seen in a 1-second window is counted once for that second (`persons` column).
- Each new ID ever seen is added to the cumulative total (`total_people` in the summary).
- The same person keeping the same tracking ID is **not** double-counted within a second. If tracking is lost and a new ID is assigned, the person is counted again.
