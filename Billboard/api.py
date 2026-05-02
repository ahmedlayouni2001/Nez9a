#!/usr/bin/env python3
"""
Billboard API — Flask server exposing session_report.json + video for the frontend.

Endpoints:
    GET /api/billboard/session   — latest session_report.json
    GET /api/billboard/video     — stream the processed video file
    GET /api/health

Usage:
    pip install flask flask-cors
    python api.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from flask import Flask, jsonify, send_file, request, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

HERE = Path(__file__).parent

SESSION_REPORT = HERE / "session_report.json"

# Candidate video paths — first one that exists wins
VIDEO_CANDIDATES: list[Path] = [
    HERE / "output_h264.mp4",   # re-encoded H.264 — browser compatible
    HERE / "output.mp4",
    HERE / "output_annotated.mp4",
    HERE / "video.mp4",
    HERE / "WhatsApp Video 2026-05-01 at 23.15.31.mp4",
    Path(r"C:\Users\MSI\Desktop\HackathonJunior\Billboard\WhatsApp Video 2026-05-01 at 23.15.31.mp4"),
]


def _find_video() -> Path | None:
    for p in VIDEO_CANDIDATES:
        if p.exists():
            return p
    mp4s = list(HERE.glob("*.mp4"))
    return mp4s[0] if mp4s else None


# ── Routes ───────────────────────────────────────────────────────────────

@app.route("/api/billboard/session")
def get_session():
    if not SESSION_REPORT.exists():
        return jsonify({"error": "No session report found. Run test.py first."}), 404
    with open(SESSION_REPORT, encoding="utf-8") as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/billboard/video")
def get_video():
    video_path = _find_video()
    if not video_path:
        return jsonify({"error": "Video file not found"}), 404

    # Support HTTP range requests for proper browser video seeking
    file_size = video_path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        byte1, byte2 = 0, None
        match = range_header.replace("bytes=", "").split("-")
        byte1 = int(match[0])
        if match[1]:
            byte2 = int(match[1])
        byte2 = byte2 or file_size - 1
        length = byte2 - byte1 + 1

        with open(video_path, "rb") as f:
            f.seek(byte1)
            data = f.read(length)

        rv = Response(
            data,
            206,
            mimetype="video/mp4",
            direct_passthrough=True,
        )
        rv.headers.add("Content-Range", f"bytes {byte1}-{byte2}/{file_size}")
        rv.headers.add("Accept-Ranges", "bytes")
        rv.headers.add("Content-Length", str(length))
        return rv

    return send_file(str(video_path), mimetype="video/mp4", conditional=True)


@app.route("/api/health")
def health():
    video = _find_video()
    return jsonify({
        "status": "ok",
        "session_exists": SESSION_REPORT.exists(),
        "video_exists": video is not None,
        "video_path": str(video) if video else None,
    })


if __name__ == "__main__":
    print("MediaPulse Billboard API — http://localhost:5001")
    v = _find_video()
    print(f"  Session report : {'found' if SESSION_REPORT.exists() else 'NOT FOUND — run test.py first'}")
    print(f"  Video          : {v if v else 'NOT FOUND'}")
    app.run(host="0.0.0.0", port=5001, debug=False)
