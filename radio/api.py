#!/usr/bin/env python3
"""
api.py — Flask REST API wrapping scraper_tunisia_v5.py
Exposes GET /api/radio/stations for the MediaPulse frontend.

Usage:
    pip install flask flask-cors requests groq
    python api.py

Query params:
    ?fast=1   Skip AI pipeline (listener counts only, ~5 s)
              Default runs full AI pipeline (~30–60 s, then cached 3 min)
"""
from __future__ import annotations

import os
import sys
import time
import random
import threading
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

# Load .env from project root BEFORE importing the scraper so GROQ_API_KEY is in os.environ
load_dotenv(Path(__file__).parent.parent / ".env")

# Make sure the scraper module is importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from scraper_tunisia_v5 import STATIONS, run_once, GROQ_API_KEY
    from groq import Groq as _Groq
    _GROQ_CLASS = _Groq
    _GROQ_AVAILABLE = bool(GROQ_API_KEY)
except ImportError as exc:
    print(f"[api] Import warning: {exc}")
    _GROQ_CLASS = None
    _GROQ_AVAILABLE = False
    GROQ_API_KEY = ""
    # These will be imported lazily when the first request arrives.
    # If the import truly fails, the /api/radio/stations endpoint returns an error.

app = Flask(__name__)
CORS(app)  # allow all origins in development

# ── Cache ────────────────────────────────────────────────────────────────
_cache: dict = {
    "full": None, "full_ts": 0.0,
    "fast": None, "fast_ts": 0.0,
}
_lock = threading.Lock()
FULL_TTL = 180   # seconds — AI pipeline is expensive
FAST_TTL = 60    # seconds — listener-count only


# ── Helpers ──────────────────────────────────────────────────────────────
def _income(csp_plus: int, middle: int, csp_minus: int) -> str:
    m = max(csp_plus, middle, csp_minus)
    if m == csp_plus:
        return "High"
    if m == csp_minus:
        return "Low"
    return "Middle"


def _probe_to_dict(probe, station: dict, fake_range: tuple[int, int]) -> dict:
    """Convert a Probe dataclass + its STATIONS entry to a plain dict."""
    listeners = probe.listeners
    if listeners is None:
        lo, hi = fake_range
        listeners = random.randint(lo, hi)

    a = probe.audience or {}
    if "error" in a:
        a = {}

    br      = a.get("audience_breakdown", {})
    gender  = br.get("gender", {})
    age     = br.get("age", {})
    socio   = br.get("socio_economic", {})
    ctx     = a.get("broadcast_context", {})

    women_pct  = gender.get("women_pct") or 45
    age45      = (age.get("45_54_pct") or 0) + (age.get("55_plus_pct") or 0)
    csp_plus   = socio.get("csp_plus_pct") or 33
    mid        = socio.get("middle_pct") or 40
    csp_minus  = socio.get("csp_minus_pct") or 27

    profile = station.get("profile", {})
    region_hint = profile.get("expected_audience", "National").split(",")[0].strip()

    return {
        "name":        probe.name,
        "fm":          probe.fm,
        "status":      probe.status,
        "listeners":   listeners,
        "peak":        probe.peak,
        "now_playing": probe.title,
        "clickcount":  probe.clickcount,
        "clicktrend":  probe.clicktrend,
        "audience": {
            # Fields consumed by the existing AudiencePills component
            "womenPct":  women_pct,
            "age45Plus": age45,
            "region":    region_hint,
            "income":    _income(csp_plus, mid, csp_minus),
            # Extended AI fields (null when fast=1 or Groq unavailable)
            "persona":   a.get("live_persona_summary"),
            "topic":     ctx.get("panel_topic"),
            "tone":      ctx.get("broadcast_tone"),
            # Full breakdown
            "men_pct":    gender.get("men_pct") or 55,
            "age_15_24":  age.get("15_24_pct") or 15,
            "age_25_34":  age.get("25_34_pct") or 25,
            "age_35_44":  age.get("35_44_pct") or 30,
            "age_45_54":  age.get("45_54_pct") or 20,
            "age_55_plus": age.get("55_plus_pct") or 10,
            "csp_plus":   csp_plus,
            "csp_middle": mid,
            "csp_minus":  csp_minus,
        },
    }


def _run_pipeline(do_ai: bool) -> list[dict]:
    """Execute the scraper pipeline and return list of station dicts."""
    groq_client = None
    if do_ai and _GROQ_AVAILABLE and _GROQ_CLASS:
        groq_client = _GROQ_CLASS(api_key=GROQ_API_KEY)

    probes = run_once(groq_client, do_ai=do_ai, skip_rb=False)

    real_counts = [p.listeners for p in probes if p.listeners is not None]
    lo = min(real_counts) if real_counts else 300
    hi = max(real_counts) if real_counts else 5000

    return [_probe_to_dict(p, s, (lo, hi)) for p, s in zip(probes, STATIONS)]


# ── Routes ───────────────────────────────────────────────────────────────
@app.route("/api/radio/stations")
def get_stations():
    fast      = request.args.get("fast", "0") == "1"
    cache_key = "fast" if fast else "full"
    ttl       = FAST_TTL if fast else FULL_TTL

    with _lock:
        if (_cache[cache_key] is not None
                and (time.time() - _cache[f"{cache_key}_ts"]) < ttl):
            data = _cache[cache_key]
            return jsonify({
                "stations":        data,
                "total_listeners": sum(s["listeners"] for s in data),
                "cached":          True,
                "ai_enabled":      not fast and _GROQ_AVAILABLE,
            })

    try:
        data = _run_pipeline(do_ai=not fast)
        with _lock:
            _cache[cache_key]            = data
            _cache[f"{cache_key}_ts"]    = time.time()
        return jsonify({
            "stations":        data,
            "total_listeners": sum(s["listeners"] for s in data),
            "cached":          False,
            "ai_enabled":      not fast and _GROQ_AVAILABLE,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/health")
def health():
    return jsonify({
        "status":         "ok",
        "groq_available": _GROQ_AVAILABLE,
    })


if __name__ == "__main__":
    print("MediaPulse Radio API — http://localhost:5000")
    print(f"  Groq AI: {'enabled' if _GROQ_AVAILABLE else 'disabled (set GROQ_API_KEY)'}")
    app.run(host="0.0.0.0", port=5000, debug=False)
