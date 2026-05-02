"""
==============================================================================
3SG Billboard Intelligence - CV Engine
==============================================================================
Real-time pipeline that processes a video stream (webcam, file, or RTSP) and
extracts:
  1. People & vehicle counts (YOLOv8n, with tracking to avoid double counting)
  2. Attention rate (MediaPipe FaceMesh -> head pose estimation)
  3. Fraud detection (perceptual hashing of the billboard region)

Output is pushed both as JSON to a WebSocket and printed locally.

USAGE:
    python cv_engine.py --source 0                     # webcam
    python cv_engine.py --source ./videos/tunis.mp4    # video file
    python cv_engine.py --source 0 --billboard-id BB_HABIB_001

REQUIREMENTS: see requirements.txt
==============================================================================
"""

import argparse
import asyncio
import glob
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import numpy as np
import websockets
from PIL import Image
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIG - tune these for your demo
# ----------------------------------------------------------------------------
YOLO_MODEL = "yolov8n.pt"          # nano = fast on CPU, ~6MB auto-download
YOLO_CONF_THRESHOLD = 0.4           # detection confidence (0.0 - 1.0)
PROCESS_EVERY_N_FRAMES = 3          # skip frames for CPU speed
FRAUD_CHECK_INTERVAL_SEC = 2        # re-hash billboard region every N seconds
FRAUD_HASH_THRESHOLD = 10           # hamming distance >10 = different image
PUSH_INTERVAL_SEC = 1.0             # how often to send aggregated metrics
WS_URL = "ws://localhost:8000/ws/cv"  # backend websocket endpoint

# Ad overlay defaults (can be overridden via CLI). When --ads is provided,
# the engine composites the ad images onto the billboard ROI and rotates
# through them every AD_SWAP_SEC seconds. The fraud reference is locked to
# the FIRST ad, so subsequent swaps trigger the fraud alarm.
AD_SWAP_SEC = 30                    # seconds between automatic ad swaps
AD_BORDER_PX = 6                    # frame around the ad inside the ROI
DEFAULT_AD_GLOB = "ad_*.png"      # auto-load any ad image matching this pattern

# COCO class IDs we care about (YOLO default classes)
PERSON_CLASS = 0
VEHICLE_CLASSES = {1, 2, 3, 5, 7}   # bicycle, car, motorcycle, bus, truck

# ----------------------------------------------------------------------------
# BILLBOARD REGION OF INTEREST (ROI)
# This is the rectangle in the frame where the billboard is visible.
# For demo: point camera at a printed poster, set this to the poster bounds.
# Format: (x1, y1, x2, y2) as fraction of frame size (0.0 - 1.0)
# ----------------------------------------------------------------------------
BILLBOARD_ROI = (0.640625, 0.841796875, 1.0, 0.92578125)  # updated from roi_drawer.py

# Load custom ROI from roi_config.txt if exists
try:
    with open("roi_config.txt", "r") as f:
        line = f.readline().strip()
        if line:
            parts = line.split()
            if len(parts) == 4:
                loaded_roi = tuple(float(p) for p in parts)
                clamped_roi = tuple(max(0.0, min(1.0, v)) for v in loaded_roi)
                if clamped_roi != loaded_roi:
                    print(f"Loaded ROI had out-of-bounds values, clamping to {clamped_roi}")
                BILLBOARD_ROI = clamped_roi
                print(f"Loaded custom ROI: {BILLBOARD_ROI}")
                if BILLBOARD_ROI[0] >= BILLBOARD_ROI[2] or BILLBOARD_ROI[1] >= BILLBOARD_ROI[3]:
                    raise ValueError("Loaded ROI is invalid: x1 >= x2 or y1 >= y2")
except FileNotFoundError:
    pass
except Exception as e:
    print(f"Failed to load ROI from roi_config.txt: {e}")
    BILLBOARD_ROI = (0.05, 0.05, 0.45, 0.55)
    print(f"Reverted to default ROI: {BILLBOARD_ROI}")

# ----------------------------------------------------------------------------


class AdOverlay:
    """Composites real ad images onto the billboard ROI of every frame.

    This is the demo's killer feature for fraud detection: the underlying
    street video stays continuous (same people, same lighting, same camera),
    only the billboard creative changes -- exactly what happens in the real
    world when a sub-contractor swaps a poster.

    Usage flow:
      1. Load N ad images (e.g. Tunisie Telecom, then Giga Klem)
      2. For the first AD_SWAP_SEC seconds, composite ad #0 into the ROI
      3. After that, switch to ad #1 -- the perceptual hash diverges from
         the locked reference and the fraud alarm fires.
    """

    def __init__(self, ad_paths: list[str], swap_after_sec: float = AD_SWAP_SEC,
                 border_px: int = AD_BORDER_PX):
        self.ads: list[np.ndarray] = []
        self.ad_names: list[str] = []
        for p in ad_paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[AdOverlay] WARN - could not load {p}, skipping")
                continue
            self.ads.append(img)
            self.ad_names.append(Path(p).stem)
        if not self.ads:
            raise RuntimeError("AdOverlay: no valid ad images loaded")
        print(f"[AdOverlay] Loaded {len(self.ads)} ads: {self.ad_names}")

        self.swap_after_sec = swap_after_sec
        self.border_px = border_px
        self.start_time: float | None = None
        self.current_index = 0
        self._last_announced_index = -1

    def _select_ad(self) -> int:
        """Decide which ad index should be shown right now."""
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        # Stay on ad 0 until swap_after_sec, then advance to ad 1, etc.
        # Cap at the last ad (don't loop -- once swapped, fraud stays detected)
        idx = min(int(elapsed // self.swap_after_sec), len(self.ads) - 1)
        if idx != self._last_announced_index:
            print(f"[AdOverlay] t={elapsed:.1f}s -> displaying "
                  f"'{self.ad_names[idx]}'")
            self._last_announced_index = idx
        self.current_index = idx
        return idx

    def composite(self, frame: np.ndarray, roi_box: tuple[int, int, int, int]
                  ) -> np.ndarray:
        """Paste the current ad into the billboard ROI of the frame.

        Modifies the frame in-place AND returns it for chaining.
        Preserves the ad's aspect ratio with letterboxing inside the ROI.
        """
        ad_idx = self._select_ad()
        ad = self.ads[ad_idx]

        x1, y1, x2, y2 = roi_box
        roi_w, roi_h = x2 - x1, y2 - y1
        if roi_w <= 0 or roi_h <= 0:
            return frame

        # Inner area (subtract border)
        b = self.border_px
        inner_w = max(1, roi_w - 2 * b)
        inner_h = max(1, roi_h - 2 * b)

        # Resize ad preserving aspect ratio
        ah, aw = ad.shape[:2]
        scale = min(inner_w / aw, inner_h / ah)
        new_w = max(1, int(aw * scale))
        new_h = max(1, int(ah * scale))
        ad_resized = cv2.resize(ad, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Fill the entire ROI with a black "billboard frame", then paste ad centered
        cv2.rectangle(frame, (x1, y1), (x2, y2), (15, 15, 15), thickness=-1)
        # Outer border (white, like a real billboard frame)
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1),
                      (240, 240, 240), thickness=2)

        # Center the ad inside the inner area
        ox = x1 + b + (inner_w - new_w) // 2
        oy = y1 + b + (inner_h - new_h) // 2
        frame[oy:oy + new_h, ox:ox + new_w] = ad_resized
        return frame

    @property
    def current_ad_name(self) -> str:
        if 0 <= self.current_index < len(self.ad_names):
            return self.ad_names[self.current_index]
        return "nothing"



class FraudDetector:
    """Compares the current billboard ROI against a stored reference image
    using perceptual hashing. Detects if the physical poster has been swapped.
    """

    def __init__(self, reference_image_path: str | None = None):
        self.reference_hash = None
        self.last_check_time = 0.0
        self.last_status = "nothing"
        self.last_distance = 0
        self.fraud_detected_at: str | None = None
        if reference_image_path and Path(reference_image_path).exists():
            ref = Image.open(reference_image_path)
            self.reference_hash = imagehash.phash(ref)
            print(f"[FraudDetector] Loaded reference: {reference_image_path}")

    def set_reference_from_frame(self, frame_bgr, roi_box):
        """Capture the current billboard ROI as the new reference."""
        x1, y1, x2, y2 = roi_box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        self.reference_hash = imagehash.phash(pil)
        print("[FraudDetector] Reference hash set from current frame.")

    def check(self, frame_bgr, roi_box) -> dict:
        """Returns {'status': 'ok'|'fraud'|'unknown', 'distance': int, 'match_score': float}"""
        # Once fraud is detected it stays — a swapped poster doesn't un-swap itself
        if self.fraud_detected_at is not None:
            return {"status": "fraud", "distance": self.last_distance,
                    "match_score": max(0.0, 1.0 - (self.last_distance / 64.0)),
                    "fraud_detected_at": self.fraud_detected_at}

        now = time.time()
        if now - self.last_check_time < FRAUD_CHECK_INTERVAL_SEC:
            return {
                "status": self.last_status,
                "distance": self.last_distance,
                "match_score": 1.0 - (self.last_distance / 64.0),
            }

        self.last_check_time = now

        if self.reference_hash is None:
            return {"status": "nothing", "distance": 0, "match_score": 0.0}

        x1, y1, x2, y2 = roi_box
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return {"status": "nothing", "distance": 0, "match_score": 0.0}

        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        current_hash = imagehash.phash(pil)
        distance = self.reference_hash - current_hash

        status = "fraud" if distance > FRAUD_HASH_THRESHOLD else "nothing"
        match_score = max(0.0, 1.0 - (distance / 64.0))

        if status == "fraud":
            self.fraud_detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[FraudDetector] *** FRAUD DETECTED at {self.fraud_detected_at} ***")

        self.last_status = status
        self.last_distance = distance
        return {"status": status, "distance": distance, "match_score": match_score,
                "fraud_detected_at": self.fraud_detected_at}


class MetricsAggregator:
    def __init__(self):
        self.seen_person_ids: set[int] = set()
        self._interval_ids: set[int] = set()

    def add_track(self, kind: str, track_id: int):
        if kind == "person":
            self.seen_person_ids.add(track_id)
            self._interval_ids.add(track_id)

    def pop_interval(self) -> int:
        """Returns people count for this second, then resets the interval."""
        count = len(self._interval_ids)
        self._interval_ids = set()
        return count

    def total_people(self) -> int:
        return len(self.seen_person_ids)


class BillboardCVPipeline:
    def __init__(self, source, billboard_id: str,
                 reference_image: str | None = None,
                 ws_url: str = WS_URL,
                 show_window: bool = True,
                 ad_overlay: AdOverlay | None = None):
        self.source = source
        self.billboard_id = billboard_id
        self.ws_url = ws_url
        self.show_window = show_window
        self.ad_overlay = ad_overlay

        print(f"[Pipeline] Loading YOLO model: {YOLO_MODEL}")
        self.model = YOLO(YOLO_MODEL)
        self.fraud = FraudDetector(reference_image)
        self.metrics = MetricsAggregator()

        self.last_push_time = 0.0
        self.ws = None
        self.frame_count = 0
        self.last_interval_people = 0
        self.per_second_log: list[dict] = []
        self.output_video_path = Path(r"C:\Users\MSI\Desktop\HackathonJunior\Billboard\output_h264.mp4")
        self.writer: cv2.VideoWriter | None = None

    async def connect_ws(self):
        try:
            self.ws = await websockets.connect(self.ws_url)
            print(f"[Pipeline] Connected to backend at {self.ws_url}")
        except Exception as e:
            print(f"[Pipeline] WARN - WebSocket connect failed ({e}). "
                  f"Will run in standalone mode (printing only).")
            self.ws = None

    async def push_metrics(self, timestamp: str, persons: int, fraud_status: str):
        self.per_second_log.append({
            "timestamp": timestamp,
            "persons": persons,
            "fraud_status": fraud_status,
        })
        sep = "+---------------------+---------+--------------+"
        print(sep)
        print(f"| {'Timestamp':<19} | {'Persons':>7} | {'Fraud Status':<12} |")
        print(sep)
        print(f"| {timestamp:<19} | {persons:>7} | {fraud_status.upper():<12} |")
        print(sep)
        if self.ws is not None:
            payload = {"timestamp": timestamp, "persons": persons,
                       "fraud_status": fraud_status}
            try:
                await self.ws.send(json.dumps(payload))
            except Exception as e:
                print(f"[Pipeline] WS send failed: {e}. Reconnecting...")
                await self.connect_ws()

    def get_roi_box(self, frame_shape) -> tuple[int, int, int, int]:
        h, w = frame_shape[:2]
        rx1, ry1, rx2, ry2 = BILLBOARD_ROI
        return int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)

    async def run(self):
        await self.connect_ws()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source}")

        if self.show_window:
            cv2.namedWindow("3SG CV Engine", cv2.WINDOW_NORMAL)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        ref_set = self.fraud.reference_hash is not None
        fraud_result = {"status": "nothing", "match_score": 0.0, "distance": 0}
        video_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Init writer on first frame — guarantees correct dimensions
                if self.writer is None:
                    fh, fw = frame.shape[:2]
                    for codec in ("avc1", "mp4v"):
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        self.writer = cv2.VideoWriter(
                            str(self.output_video_path), fourcc, fps, (fw, fh)
                        )
                        if self.writer.isOpened():
                            print(f"[Pipeline] Recording {fw}×{fh} @ {fps:.1f}fps [{codec}] → {self.output_video_path}")
                            break
                        self.writer = None
                    if self.writer is None:
                        print("[Pipeline] WARN: could not open any video writer")

                self.frame_count += 1
                roi_box = self.get_roi_box(frame.shape)

                if self.ad_overlay is not None:
                    self.ad_overlay.composite(frame, roi_box)

                if not ref_set and self.frame_count == 30:
                    self.fraud.set_reference_from_frame(frame, roi_box)
                    ref_set = True

                if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
                    self._draw_overlay(frame, roi_box, fraud_result,
                                       self.last_interval_people)
                    if self.writer:
                        self.writer.write(frame)
                    if self.show_window:
                        cv2.imshow("3SG CV Engine", self._scale_for_display(frame))
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue

                # ---- YOLO with tracking ----
                results = self.model.track(
                    frame,
                    persist=True,
                    conf=YOLO_CONF_THRESHOLD,
                    verbose=False,
                    classes=[PERSON_CLASS] + list(VEHICLE_CLASSES),
                )

                if results and len(results) > 0 and results[0].boxes is not None:
                    boxes = results[0].boxes
                    if boxes.id is not None:
                        ids = boxes.id.cpu().numpy().astype(int)
                        cls = boxes.cls.cpu().numpy().astype(int)
                        for tid, c in zip(ids, cls):
                            if c == PERSON_CLASS:
                                self.metrics.add_track("person", int(tid))
                            elif c in VEHICLE_CLASSES:
                                self.metrics.add_track("vehicle", int(tid))

                # ---- Fraud detection ----
                fraud_result = self.fraud.check(frame, roi_box)

                # ---- Per-second table ----
                now = time.time()
                if now - self.last_push_time >= PUSH_INTERVAL_SEC:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.last_interval_people = self.metrics.pop_interval()
                    await self.push_metrics(ts, self.last_interval_people,
                                            fraud_result["status"])
                    self.last_push_time = now

                self._draw_overlay(frame, roi_box, fraud_result,
                                   self.last_interval_people)
                if self.writer:
                    self.writer.write(frame)
                if self.show_window:
                    cv2.imshow("3SG CV Engine", self._scale_for_display(frame))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            cap.release()
            if self.writer:
                self.writer.release()
                self.writer = None
                print(f"[Pipeline] Output video saved → {self.output_video_path}")
            if self.show_window:
                cv2.destroyAllWindows()
            if self.ws:
                await self.ws.close()

        video_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._print_summary(video_start, video_end)

    def _draw_overlay(self, frame, roi_box, fraud_result: dict, people: int = 0):
        x1, y1, x2, y2 = roi_box
        color = (0, 0, 255) if fraud_result.get("status") == "fraud" else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, "Billboard ROI", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        text_lines = [
            f"People: {people}",
            f"Fraud:  {fraud_result.get('status', 'nothing').upper()}",
        ]
        if fraud_result.get("fraud_detected_at"):
            text_lines.append(f"Since:  {fraud_result['fraud_detected_at']}")
        for i, line in enumerate(text_lines):
            cv2.putText(frame, line, (10, frame.shape[0] - 65 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _print_summary(self, video_start: str, video_end: str):
        total = self.metrics.total_people()
        fraud_at = self.fraud.fraud_detected_at or "None"

        print()
        sep = "+-------------------+------------------------+"
        print(sep)
        print(f"| {'SESSION SUMMARY':<37} |")
        print(sep)
        print(f"| {'Video start':<17} | {video_start:<22} |")
        print(f"| {'Video end':<17} | {video_end:<22} |")
        print(f"| {'Total people':<17} | {str(total):<22} |")
        print(f"| {'Fraud detected at':<17} | {fraud_at:<22} |")
        print(sep)

        report = {
            "summary": {
                "video_start": video_start,
                "video_end": video_end,
                "total_people": total,
                "fraud_detected_at": self.fraud.fraud_detected_at,
            },
            "output_video": str(self.output_video_path),
            "per_second": self.per_second_log,
        }
        report_path = Path("session_report.json")
        report_path.write_text(json.dumps(report, indent=2))
        print(f"\nReport saved to {report_path.resolve()}")

    def _scale_for_display(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        max_w, max_h = 1280, 800
        scale = min(1.0, max_w / w, max_h / h)
        if scale < 1.0:
            return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return frame


# ----------------------------------------------------------------------------
# CLI ENTRY
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="3SG Billboard CV Engine")
    p.add_argument("--source", default=r"C:\Users\MSI\Desktop\HackathonJunior\Billboard\WhatsApp Video 2026-05-01 at 23.15.31.mp4",
                   help="Video source: 0 for webcam, or path to video file/RTSP URL")
    p.add_argument("--billboard-id", default="BB_HABIB_001",
                   help="Billboard identifier (matches dashboard)")
    p.add_argument("--reference", default=None,
                   help="Path to reference creative image for fraud detection")
    p.add_argument("--ws-url", default=WS_URL,
                   help="Backend WebSocket URL")
    p.add_argument("--no-window", action="store_true",
                   help="Run headless (no preview window)")
    p.add_argument("--ads", nargs="+", default=None,
                   metavar="PATH",
                   help="One or more ad image paths to overlay onto the "
                        "billboard ROI. The first ad becomes the fraud "
                        "reference; subsequent ads trigger the fraud alarm "
                        "after --ad-swap-sec seconds each. If omitted, "
                        f"all files matching '{DEFAULT_AD_GLOB}' are used.")
    p.add_argument("--ad-swap-sec", type=float, default=AD_SWAP_SEC,
                   help=f"Seconds before swapping to the next ad "
                        f"(default: {AD_SWAP_SEC}s)")
    return p.parse_args()


async def main():
    args = parse_args()
    src = int(args.source) if args.source.isdigit() else args.source

    ads = args.ads
    if not ads:
        ads = sorted(glob.glob(DEFAULT_AD_GLOB))
        if ads:
            print(f"[Main] Auto-loading billboard ad images: {ads}")

    overlay = None
    if ads:
        overlay = AdOverlay(ads, swap_after_sec=args.ad_swap_sec)

    pipeline = BillboardCVPipeline(
        source=src,
        billboard_id=args.billboard_id,
        reference_image=args.reference,
        ws_url=args.ws_url,
        show_window=not args.no_window,
        ad_overlay=overlay,
    )
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())