#!/usr/bin/env python3
"""
scraper_tunisia_v5.py
═════════════════════
Fusion de :
  • tn_radio_listeners.py (v2)  → moteur listener count robuste
        (résolution playlists, redirection origine, 4 stratégies probe,
         fallback radio-browser.info)
  • scraper_tunisia_v4.py       → pipeline IA Groq
        (capture audio 15s → Whisper → LLM llama-3.3-70b → profil démographique)

Pipeline complet par cycle :
  1. probe_station()           → status, listeners, peak, song, clickcount, trend
  2. capture_audio_clip()      → 15s mp3 du flux (cycles avec pipeline IA)
  3. transcribe_audio()        → texte via Groq Whisper-large-v3-turbo
  4. predict_audience()        → JSON démographique via llama-3.3-70b
  5. print_snapshot()          → rendu console enrichi
  6. log_csv()                 → ligne par station par cycle

Dépendances : pip install requests groq
Variable    : export GROQ_API_KEY="gsk_..."
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import socket
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from typing import Optional
from urllib.parse import urlparse, urlunparse

import requests
from groq import Groq

# ══════════════════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════════════════
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

POLL_INTERVAL          = 60
AUDIO_CAPTURE_SECONDS  = 15
TRANSCRIBE_EVERY_N     = 3       # pipeline IA 1 cycle sur 3
WHISPER_MODEL          = "whisper-large-v3-turbo"
LLM_MODEL              = "llama-3.3-70b-versatile"

USER_AGENT   = "tn-radio-probe/2.0 (+listener stats client)"
TIMEOUT      = 6
RB_TIMEOUT   = 8
MAX_PARALLEL = 12
OUTPUT_CSV   = "audience_log.csv"

DEBUG = False

# ══════════════════════════════════════════════════════════════════════════
#  CATALOGUE DES STATIONS
#  Chaque entrée : stream technique + rb_name (radio-browser) + profile (LLM)
# ══════════════════════════════════════════════════════════════════════════
STATIONS: list[dict] = [
    {"name": "Mosaïque FM", "fm": "94.9 / 88.2-107.8 MHz",
     "stream": "http://radio.mosaiquefm.net:8000/mosalive",
     "rb_name": "Mosaique FM",
     "profile": {"editorial_line": "privée leader généraliste",
                 "language": "Arabic dialect/French",
                 "format": "actualité, talk-shows, musique tunisienne et internationale",
                 "expected_audience": "20-55 ans, tous milieux, fortement urbain"}},

    {"name": "Shems FM", "fm": "88.7-107.6 MHz",
     "stream": "http://stream6.tanitweb.com/shems",
     "rb_name": "Shems FM",
     "profile": {"editorial_line": "privée généraliste",
                 "language": "Arabic dialect",
                 "format": "talk-shows société, musique pop arabe, divertissement",
                 "expected_audience": "20-45 ans, classe moyenne urbaine"}},

    {"name": "Radio Jawhara FM", "fm": "89.4 / 102.5 / 104.4 / 107.3",
     "stream": "http://streaming2.toutech.net:8000/jawharafm",
     "rb_name": "Jawhara FM",
     "profile": {"editorial_line": "privée régionale Sahel",
                 "language": "Arabic dialect",
                 "format": "actualité Sahel, musique variée, divertissement",
                 "expected_audience": "20-50 ans, Sousse-Monastir-Mahdia"}},

    {"name": "Radio IFM", "fm": "100.6 MHz",
     "stream": "http://5.135.142.50:8000/direct",
     "rb_name": "Radio Ifm",
     "profile": {"editorial_line": "privée humour & musique",
                 "language": "Arabic dialect/French",
                 "format": "humour, sketches, tubes, jeux",
                 "expected_audience": "18-40 ans, public léger"}},

    {"name": "Diwan FM", "fm": "97.3 MHz (Sfax)",
     "stream": "http://stream8.tanitweb.com/diwanfm",
     "rb_name": "Diwan FM",
     "profile": {"editorial_line": "régionale Sfax",
                 "language": "Arabic",
                 "format": "actualité locale Sfax-sud, débats, musique",
                 "expected_audience": "25-60 ans, sud tunisien"}},

    {"name": "Express FM", "fm": "103.6 / 104.0 MHz",
     "stream": "http://expressfm.ice.infomaniak.ch/expressfm-64.mp3",
     "rb_name": "Express FM",
     "profile": {"editorial_line": "privée économique",
                 "language": "Arabic/French",
                 "format": "économie, business, marchés financiers, entreprenariat",
                 "expected_audience": "cadres, entrepreneurs, 25-55 ans, CSP+"}},

    {"name": "Radio Zitouna FM", "fm": "various",
     "stream": "http://stream.zitounafm.net:8080/zitounafm",
     "rb_name": "Radio Zitouna FM",
     "profile": {"editorial_line": "religieuse islamique",
                 "language": "Arabic",
                 "format": "Coran, conférences religieuses, débats théologiques",
                 "expected_audience": "30+, pratiquants, conservateurs"}},

    {"name": "KnOOz FM", "fm": "90.6 MHz",
     "stream": "http://streaming.knoozfm.net:8000/knoozfm",
     "rb_name": "KnOOz FM",
     "profile": {"editorial_line": "musique & jeux",
                 "language": "Arabic",
                 "format": "musique populaire, jeux, animation",
                 "expected_audience": "20-40 ans, populaire"}},

    {"name": "Radio Nationale Tunisienne", "fm": "various",
     "stream": "http://rtstream.tanitweb.com/nationale",
     "rb_name": "Radio Nationale Tunisienne",
     "profile": {"editorial_line": "publique généraliste",
                 "language": "Arabic",
                 "format": "actualité, débats, musique tunisienne classique",
                 "expected_audience": "30-65 ans, urbain & rural, mixte"}},

    {"name": "Radio Tunis Chaîne Internationale (RTCI)", "fm": "98.0 MHz",
     "stream": "http://rtstream.tanitweb.com/rtci",
     "rb_name": "Radio Tunis Chaîne Internationale",
     "profile": {"editorial_line": "publique internationale",
                 "language": "French/English/Spanish",
                 "format": "culture, actualité internationale, musiques du monde",
                 "expected_audience": "intellectuels, expatriés, francophones cultivés"}},
]


# ══════════════════════════════════════════════════════════════════════════
#  DATA CLASS — étendu de ton Probe + champs IA
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class Probe:
    name: str
    fm: str
    stream: str

    # Live counts
    status: str = "unknown"            # ok | locked | hls | offline | error
    listeners: Optional[int] = None
    peak: Optional[int] = None
    bitrate: Optional[int] = None
    title: Optional[str] = None
    server_kind: Optional[str] = None
    resolved_origin: Optional[str] = None

    # radio-browser
    clickcount: Optional[int] = None
    clicktrend: Optional[int] = None
    rb_codec: Optional[str] = None
    rb_bitrate: Optional[int] = None
    rb_lastcheckok: Optional[bool] = None

    # IA pipeline
    transcript: Optional[str] = None
    keywords: list = field(default_factory=list)
    audience: Optional[dict] = None    # JSON returned by LLM

    # Diagnostic
    detail: str = ""
    attempts: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        if not DEBUG:
            d.pop("attempts", None)
        return d


def _dbg(probe: Probe, msg: str):
    if DEBUG:
        probe.attempts.append(msg)


# ══════════════════════════════════════════════════════════════════════════
#  HTTP HELPER
# ══════════════════════════════════════════════════════════════════════════
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})


def _get(url: str, *, timeout: int = TIMEOUT, stream: bool = False,
         allow_redirects: bool = True) -> Optional[requests.Response]:
    try:
        return SESSION.get(url, timeout=timeout, stream=stream,
                           allow_redirects=allow_redirects)
    except (requests.RequestException, socket.error):
        return None


# ══════════════════════════════════════════════════════════════════════════
#  STEP 1 — résolution playlists + redirections
# ══════════════════════════════════════════════════════════════════════════
def resolve_stream(url: str, probe: Probe) -> str:
    lower = url.lower()
    if lower.endswith((".pls", ".m3u", ".m3u8")) or "playlist" in lower:
        r = _get(url)
        if r and r.status_code == 200:
            for line in r.text.splitlines():
                line = line.strip()
                if line.startswith("File") and "=" in line:
                    cand = line.split("=", 1)[1].strip()
                    _dbg(probe, f"PLS resolved {url} -> {cand}")
                    return cand
                if line and not line.startswith("#") and \
                   line.lower().startswith(("http://", "https://")):
                    _dbg(probe, f"M3U resolved {url} -> {line}")
                    return line
    try:
        r = SESSION.get(url, timeout=TIMEOUT, stream=True, allow_redirects=True)
        final = r.url
        r.close()
        if final != url:
            _dbg(probe, f"redirect {url} -> {final}")
        return final
    except (requests.RequestException, socket.error):
        return url


def _origin(stream_url: str) -> str:
    p = urlparse(stream_url)
    return urlunparse((p.scheme, p.netloc, "", "", "", ""))


# ══════════════════════════════════════════════════════════════════════════
#  STEP 2 — probes spécifiques par protocole
# ══════════════════════════════════════════════════════════════════════════
def _walk_icecast_sources(stats: dict):
    sources = stats.get("source")
    if sources is None:
        return 0, 0, None, None
    if isinstance(sources, dict):
        sources = [sources]
    listeners = peak = 0
    bitrate = None
    title = None
    for s in sources:
        try: listeners += int(s.get("listeners", 0) or 0)
        except (TypeError, ValueError): pass
        try: peak += int(s.get("listener_peak", 0) or 0)
        except (TypeError, ValueError): pass
        if not bitrate:
            bitrate = s.get("bitrate") or s.get("audio_bitrate")
        if not title:
            title = s.get("title") or s.get("yp_currently_playing")
    return listeners, peak, bitrate, title


def probe_icecast_json(origin: str, probe: Probe) -> Optional[dict]:
    for path in ("/status-json.xsl", "/status.json"):
        url = f"{origin}{path}"
        r = _get(url)
        _dbg(probe, f"GET {url} -> {r.status_code if r else 'fail'}")
        if not r or r.status_code != 200: continue
        try: data = r.json()
        except (ValueError, json.JSONDecodeError): continue
        stats = data.get("icestats") or data.get("server") or data
        listeners, peak, bitrate, title = _walk_icecast_sources(stats)
        if listeners or peak or stats.get("source"):
            return {"listeners": listeners, "peak": peak or None,
                    "bitrate": bitrate, "title": title,
                    "server_kind": "icecast2"}
    return None


def probe_shoutcast_v2(origin: str, probe: Probe) -> Optional[dict]:
    for path in ("/stats?json=1", "/stats?sid=1&json=1", "/statistics?json=1"):
        url = f"{origin}{path}"
        r = _get(url)
        _dbg(probe, f"GET {url} -> {r.status_code if r else 'fail'}")
        if not r or r.status_code != 200: continue
        body = r.text.lstrip()
        if not body.startswith("{"): continue
        try: d = r.json()
        except (ValueError, json.JSONDecodeError): continue
        cur = d.get("currentlisteners")
        if cur is None: continue
        return {"listeners": int(cur),
                "peak": d.get("peaklisteners"),
                "bitrate": d.get("bitrate"),
                "title": d.get("songtitle"),
                "server_kind": "shoutcast-v2"}
    return None


def probe_shoutcast_v1(origin: str, probe: Probe) -> Optional[dict]:
    for path in ("/7.html", "/index.html?sid=1"):
        url = f"{origin}{path}"
        r = _get(url)
        _dbg(probe, f"GET {url} -> {r.status_code if r else 'fail'}")
        if not r or r.status_code != 200: continue
        m = re.search(r"<body[^>]*>([^<]*)</body>", r.text, re.IGNORECASE)
        if not m: continue
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) >= 7 and parts[0].isdigit():
            try:
                return {"listeners": int(parts[0]),
                        "peak":      int(parts[2]),
                        "bitrate":   int(parts[5]) if parts[5].isdigit() else None,
                        "title":     ",".join(parts[6:]).strip() or None,
                        "server_kind": "shoutcast-v1"}
            except ValueError: pass
    return None


_RX_LISTENERS = re.compile(
    r"(?:Current\s*Listeners|Listeners?(?:\s*Connected)?)\s*[:=]?\s*</?[^>]*>?\s*(\d+)",
    re.IGNORECASE)
_RX_PEAK = re.compile(
    r"(?:Listener\s*Peak|Peak\s*Listeners)\s*[:=]?\s*</?[^>]*>?\s*(\d+)",
    re.IGNORECASE)
_RX_BITRATE = re.compile(
    r"(?:Bitrate|Stream\s*Bitrate)\s*[:=]?\s*</?[^>]*>?\s*(\d+)",
    re.IGNORECASE)
_RX_TITLE = re.compile(
    r"(?:Current\s*Song|Now\s*Playing|Stream\s*Title)\s*[:=]?\s*</?[^>]*>?\s*([^<\n\r]{1,120})",
    re.IGNORECASE)


def probe_html_scrape(origin: str, probe: Probe) -> Optional[dict]:
    for path in ("/status.xsl", "/status.html", "/"):
        url = f"{origin}{path}"
        r = _get(url)
        _dbg(probe, f"GET {url} -> {r.status_code if r else 'fail'}")
        if not r or r.status_code != 200: continue
        body = r.text
        m_l = _RX_LISTENERS.search(body)
        if not m_l: continue
        m_p = _RX_PEAK.search(body)
        m_b = _RX_BITRATE.search(body)
        m_t = _RX_TITLE.search(body)
        return {"listeners": int(m_l.group(1)),
                "peak":      int(m_p.group(1)) if m_p else None,
                "bitrate":   int(m_b.group(1)) if m_b else None,
                "title":     m_t.group(1).strip() if m_t else None,
                "server_kind": "html-scrape"}
    return None


# ══════════════════════════════════════════════════════════════════════════
#  STEP 3 — orchestration probe par station
# ══════════════════════════════════════════════════════════════════════════
def probe_station(station: dict) -> Probe:
    p = Probe(name=station["name"], fm=station["fm"], stream=station["stream"])

    if station["stream"].lower().endswith(".m3u8"):
        p.status = "hls"
        p.detail = "HLS stream — listener count is not published"
        return p

    real = resolve_stream(station["stream"], p)
    p.resolved_origin = _origin(real)

    alive = _get(real, stream=True)
    if alive is None:
        p.status = "offline"
        p.detail = "stream did not respond"
        return p
    try: alive.close()
    except Exception: pass

    for fn in (probe_icecast_json, probe_shoutcast_v2,
               probe_shoutcast_v1, probe_html_scrape):
        info = fn(p.resolved_origin, p)
        if info and info.get("listeners") is not None:
            p.status      = "ok"
            p.listeners   = info["listeners"]
            p.peak        = info.get("peak")
            p.bitrate     = info.get("bitrate")
            p.title       = info.get("title")
            p.server_kind = info.get("server_kind")
            return p

    p.status = "locked"
    p.detail = "stream is up, all stats endpoints refused or hidden"
    return p


# ══════════════════════════════════════════════════════════════════════════
#  STEP 4 — radio-browser.info popularity
# ══════════════════════════════════════════════════════════════════════════
RB_MIRRORS = (
    "https://de1.api.radio-browser.info",
    "https://fi1.api.radio-browser.info",
    "https://nl1.api.radio-browser.info",
    "https://at1.api.radio-browser.info",
)


def _rb_get(path: str, params: dict | None = None,
            mirrors: tuple = RB_MIRRORS) -> Optional[list | dict]:
    for base in mirrors:
        try:
            r = SESSION.get(base + path, params=params, timeout=RB_TIMEOUT,
                            headers={"User-Agent": USER_AGENT,
                                     "Accept": "application/json"})
            if r.status_code == 200:
                try: return r.json()
                except (ValueError, json.JSONDecodeError): continue
        except (requests.RequestException, socket.error): continue
    return None


def fetch_radiobrowser_index() -> dict[str, dict]:
    data = _rb_get("/json/stations/bycountrycodeexact/TN",
                   params={"hidebroken": "false"})
    if not isinstance(data, list):
        return {}
    out: dict[str, dict] = {}
    for st in data:
        key = (st.get("name") or "").strip().lower()
        if not key: continue
        existing = out.get(key)
        if (existing is None or
                int(st.get("clickcount", 0) or 0) >
                int(existing.get("clickcount", 0) or 0)):
            out[key] = st
    return out


def _norm(s: str) -> str:
    s = (s or "").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def attach_radiobrowser(probes: list[Probe], rb_index: dict[str, dict]) -> None:
    if not rb_index: return
    norm_index = {_norm(k): v for k, v in rb_index.items()}

    for p in probes:
        st = next(s for s in STATIONS if s["name"] == p.name)
        candidates = [st.get("rb_name"), st["name"]]
        match = None
        for c in candidates:
            if not c: continue
            match = rb_index.get(c.lower()) or norm_index.get(_norm(c))
            if match: break
        if not match:
            target = _norm(st.get("rb_name") or st["name"])
            for k, v in norm_index.items():
                if target and (target in k or k in target):
                    match = v; break
        if not match: continue
        try: p.clickcount = int(match.get("clickcount", 0) or 0)
        except (TypeError, ValueError): pass
        try: p.clicktrend = int(match.get("clicktrend", 0) or 0)
        except (TypeError, ValueError): pass
        p.rb_codec = match.get("codec") or None
        try: p.rb_bitrate = int(match.get("bitrate", 0) or 0) or None
        except (TypeError, ValueError): pass
        p.rb_lastcheckok = bool(match.get("lastcheckok"))


# ══════════════════════════════════════════════════════════════════════════
#  STEP 5 — CAPTURE AUDIO + TRANSCRIPTION GROQ WHISPER
# ══════════════════════════════════════════════════════════════════════════
def capture_audio_clip(stream_url: str, duration_sec: int) -> Optional[str]:
    """Télécharge ~duration_sec de mp3 dans un fichier temporaire."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    bytes_written = 0
    start = time.time()
    try:
        r = SESSION.get(stream_url, timeout=10, stream=True)
        if r.status_code >= 400:
            tmp.close(); os.unlink(tmp.name); return None
        bitrate_kbps = 128
        try:
            bitrate_kbps = int(str(r.headers.get("icy-br", "128")).split(",")[0])
        except ValueError: pass
        target_bytes = (bitrate_kbps * 1000 // 8) * duration_sec
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk: break
            tmp.write(chunk)
            bytes_written += len(chunk)
            if bytes_written >= target_bytes or (time.time() - start) > duration_sec + 5:
                break
        r.close()
        tmp.close()
        return tmp.name if bytes_written > 1000 else None
    except Exception:
        try: tmp.close(); os.unlink(tmp.name)
        except Exception: pass
        return None


def transcribe_audio(client: Groq, audio_path: str,
                     language_hint: str = "ar") -> str:
    try:
        with open(audio_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), f.read()),
                model=WHISPER_MODEL,
                language=language_hint,
                response_format="text",
                temperature=0.0,
            )
        return (transcription if isinstance(transcription, str)
                else getattr(transcription, "text", "")).strip()
    except Exception as e:
        return f"[transcription_error: {str(e)[:80]}]"


def extract_keywords(text: str, max_kw: int = 8) -> list:
    if not text or text.startswith("["):
        return []
    words = re.findall(r"\b[\w\u0600-\u06FF]{4,}\b", text)
    seen, kw = set(), []
    for w in words:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl); kw.append(w)
        if len(kw) >= max_kw: break
    return kw


# ══════════════════════════════════════════════════════════════════════════
#  STEP 6 — PROFILAGE LLM
# ══════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are the Real-Time Audience Profiling Engine for Tunisian radio stations.

Predict the demographic breakdown of CURRENT listeners based on:
1. The station's static editorial profile (always reliable)
2. The transcribed speech captured from the live broadcast in the last 15 seconds (may be noisy or empty)

You MUST output ONLY a valid JSON object with this exact structure:
{
  "broadcast_context": {
    "panel_topic": "<infer 3-5 word topic from transcribed words>",
    "core_keywords": ["<kw1>", "<kw2>", "<kw3>"],
    "broadcast_tone": "<one of: news/talk/music/comedy/religious/economic/cultural>"
  },
  "audience_breakdown": {
    "gender": {"men_pct": <int>, "women_pct": <int>},
    "age": {"15_24_pct": <int>, "25_34_pct": <int>, "35_44_pct": <int>, "45_54_pct": <int>, "55_plus_pct": <int>},
    "socio_economic": {"csp_plus_pct": <int>, "middle_pct": <int>, "csp_minus_pct": <int>},
    "primary_occupation": "<e.g. Students, Professionals, Retirees>"
  },
  "confidence": "<low|medium|high>",
  "live_persona_summary": "<one sentence describing the typical listener right now>"
}

Each percentage group MUST sum to exactly 100. Output JSON only — no preamble."""


def predict_audience(client: Groq, station_profile: dict,
                     transcribed_text: Optional[str], keywords: list) -> dict:
    user_content = f"""STATION STATIC PROFILE:
- Editorial line: {station_profile.get('editorial_line')}
- Language: {station_profile.get('language')}
- Format: {station_profile.get('format')}
- Expected audience baseline: {station_profile.get('expected_audience')}

LIVE BROADCAST DATA (last 15s):
- Transcribed speech: "{transcribed_text or '(no speech detected — likely music)'}"
- Extracted keywords: {keywords or '[]'}

Predict the audience profile right now."""
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_content},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=800,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)[:150]}


# ══════════════════════════════════════════════════════════════════════════
#  STEP 7 — FUSION : probe + IA
# ══════════════════════════════════════════════════════════════════════════
def enrich_with_ai(probe: Probe, station: dict, groq_client: Groq) -> None:
    """Ajoute transcript + keywords + audience à un Probe déjà rempli."""
    if probe.status not in ("ok", "locked"):
        # offline ou hls → on saute le pipeline IA
        return

    profile = station.get("profile", {})
    real_url = probe.resolved_origin
    # Pour la capture, on reprend l'URL résolue + le path original si nécessaire
    # Le plus simple : repartir du stream original (qui suivra les redirections)
    stream_url = station["stream"]

    # 1. Capture audio
    audio_path = capture_audio_clip(stream_url, AUDIO_CAPTURE_SECONDS)

    if audio_path:
        try:
            # 2. Transcription
            lang_hint = "fr" if "French" in profile.get("language", "") else "ar"
            transcript = transcribe_audio(groq_client, audio_path, lang_hint)
            probe.transcript = transcript[:300] if transcript else None
            probe.keywords   = extract_keywords(transcript)

            # 3. Profilage
            if profile:
                probe.audience = predict_audience(
                    groq_client, profile, probe.transcript, probe.keywords)
        finally:
            try: os.unlink(audio_path)
            except Exception: pass
    else:
        # Audio non capturable mais on profile quand même via le profil statique
        if profile:
            probe.audience = predict_audience(groq_client, profile, None, [])


# ══════════════════════════════════════════════════════════════════════════
#  RENDU CONSOLE
# ══════════════════════════════════════════════════════════════════════════
def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _fmt_trend(t: Optional[int]) -> str:
    if t is None: return "—"
    if t > 0: return f"+{t}"
    if t < 0: return f"{t}"
    return "0"


def render_table(probes: list[Probe]) -> str:
    """Table compacte (style v2) — vue d'ensemble."""
    rows = [("Station", "FM", "Status", "Live", "Peak",
             "Clicks", "Trend24h", "Now playing")]
    for r in probes:
        rows.append((
            r.name[:34], r.fm[:22], r.status,
            "—" if r.listeners is None else str(r.listeners),
            "—" if r.peak      is None else str(r.peak),
            "—" if r.clickcount is None else f"{r.clickcount:,}",
            _fmt_trend(r.clicktrend),
            (r.title or r.detail or "")[:38],
        ))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*rows[0]),
             fmt.format(*("-" * w for w in widths))]
    for row in rows[1:]:
        lines.append(fmt.format(*row))

    ok = [r for r in probes if r.status == "ok"]
    total_live = sum(r.listeners or 0 for r in ok)
    total_clicks = sum(r.clickcount or 0 for r in probes)
    lines += [
        "",
        f"Live listeners across {len(ok)} reachable stations: {total_live}",
        f"Total radio-browser clicks: {total_clicks:,}",
    ]
    return "\n".join(lines)


def render_audience_details(probes: list[Probe]) -> str:
    """Détails IA station par station (vue enrichie)."""
    out = []
    for r in probes:
        out.append("")
        L = "?" if r.listeners is None else str(r.listeners)
        out.append(f"  ▶ {_color(r.name, '1;36'):<32} "
                   f"{_color(L + ' auditeurs', '1;32'):<25} "
                   f"({r.status})")

        if r.title:
            out.append(f"     {_color('NOW PLAYING:', '90')} {r.title}")

        if r.transcript:
            tr = r.transcript[:140]
            out.append(f"     {_color('TRANSCRIPT :', '90')} {tr}"
                       f"{'...' if len(r.transcript) >= 140 else ''}")
            if r.keywords:
                out.append(f"     {_color('KEYWORDS   :', '90')} "
                           f"{', '.join(r.keywords)}")

        a = r.audience
        if a and "error" not in a:
            ctx = a.get("broadcast_context", {})
            br  = a.get("audience_breakdown", {})
            gender = br.get("gender", {})
            age    = br.get("age", {})
            socio  = br.get("socio_economic", {})

            out.append(f"     {_color('TOPIC      :', '90')} "
                       f"{ctx.get('panel_topic', '?')} "
                       f"({_color(ctx.get('broadcast_tone', '?'), '33')})")
            out.append(f"     {_color('GENDER     :', '90')} "
                       f"♂ {gender.get('men_pct', 0)}%   "
                       f"♀ {gender.get('women_pct', 0)}%")
            out.append(f"     {_color('AGE        :', '90')} "
                       f"15-24: {age.get('15_24_pct',0)}% │ "
                       f"25-34: {age.get('25_34_pct',0)}% │ "
                       f"35-44: {age.get('35_44_pct',0)}% │ "
                       f"45-54: {age.get('45_54_pct',0)}% │ "
                       f"55+: {age.get('55_plus_pct',0)}%")
            out.append(f"     {_color('SOCIO      :', '90')} "
                       f"CSP+: {socio.get('csp_plus_pct',0)}% │ "
                       f"Moyen: {socio.get('middle_pct',0)}% │ "
                       f"CSP-: {socio.get('csp_minus_pct',0)}%   "
                       f"({br.get('primary_occupation', '?')})")
            out.append(f"     {_color('PERSONA    :', '90')} "
                       f"{_color(a.get('live_persona_summary', '?'), '3;37')}")
        elif a and "error" in a:
            out.append(f"     {_color('LLM ERROR  :', '91')} {a['error']}")
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════
#  CSV LOG
# ══════════════════════════════════════════════════════════════════════════
def log_csv(probes: list[Probe]):
    exists = os.path.isfile(OUTPUT_CSV)
    rows = []
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    for r in probes:
        a = r.audience or {}
        if "error" in a: a = {}
        br = a.get("audience_breakdown", {})
        rows.append({
            "timestamp":     ts,
            "station":       r.name,
            "status":        r.status,
            "listeners":     r.listeners if r.listeners is not None else "",
            "peak":          r.peak if r.peak is not None else "",
            "clickcount":    r.clickcount if r.clickcount is not None else "",
            "clicktrend":    r.clicktrend if r.clicktrend is not None else "",
            "now_playing":   r.title or "",
            "transcript":    (r.transcript or "")[:200],
            "panel_topic":   a.get("broadcast_context", {}).get("panel_topic", ""),
            "tone":          a.get("broadcast_context", {}).get("broadcast_tone", ""),
            "men_pct":       br.get("gender", {}).get("men_pct", ""),
            "women_pct":     br.get("gender", {}).get("women_pct", ""),
            "age_15_24":     br.get("age", {}).get("15_24_pct", ""),
            "age_25_34":     br.get("age", {}).get("25_34_pct", ""),
            "age_35_44":     br.get("age", {}).get("35_44_pct", ""),
            "age_45_54":     br.get("age", {}).get("45_54_pct", ""),
            "age_55_plus":   br.get("age", {}).get("55_plus_pct", ""),
            "csp_plus":      br.get("socio_economic", {}).get("csp_plus_pct", ""),
            "csp_middle":    br.get("socio_economic", {}).get("middle_pct", ""),
            "csp_minus":     br.get("socio_economic", {}).get("csp_minus_pct", ""),
            "persona":       a.get("live_persona_summary", ""),
        })
    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists: w.writeheader()
        w.writerows(rows)


# ══════════════════════════════════════════════════════════════════════════
#  CYCLE COMPLET
# ══════════════════════════════════════════════════════════════════════════
def run_once(groq_client: Optional[Groq], do_ai: bool,
             skip_rb: bool = False) -> list[Probe]:
    # Phase 1 : probe + radio-browser en parallèle
    results: list[Probe] = [None] * len(STATIONS)  # type: ignore
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as pool:
        futures = {pool.submit(probe_station, s): i
                   for i, s in enumerate(STATIONS)}
        rb_future = None if skip_rb else pool.submit(fetch_radiobrowser_index)
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
        rb_index = {} if rb_future is None else rb_future.result()

    if not skip_rb:
        attach_radiobrowser(results, rb_index)

    # Phase 2 : enrichissement IA (séquentiel pour respecter rate-limits Groq)
    if do_ai and groq_client is not None:
        for probe, station in zip(results, STATIONS):
            try:
                enrich_with_ai(probe, station, groq_client)
            except Exception as e:
                if probe.audience is None:
                    probe.audience = {"error": f"pipeline_exception: {str(e)[:100]}"}

    return results


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    global DEBUG

    ap = argparse.ArgumentParser(
        description="Tunisian radio listener stats + AI audience profiler",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watch", type=int, metavar="SECONDS", default=POLL_INTERVAL,
                    help=f"re-probe every N seconds (default {POLL_INTERVAL}, use 0 for one-shot)")
    ap.add_argument("--no-ai", action="store_true",
                    help="skip the audio capture / Whisper / LLM pipeline")
    ap.add_argument("--no-rb", action="store_true",
                    help="skip radio-browser.info popularity lookup")
    ap.add_argument("--ai-every", type=int, default=TRANSCRIBE_EVERY_N,
                    help=f"run AI pipeline 1 cycle out of N (default {TRANSCRIBE_EVERY_N})")
    ap.add_argument("--json", action="store_true",
                    help="emit JSON instead of formatted output")
    ap.add_argument("--debug", action="store_true",
                    help="record every probe attempt")
    args = ap.parse_args()
    DEBUG = args.debug

    # Init Groq client
    groq_client = None
    if not args.no_ai:
        if not GROQ_API_KEY:
            print("⚠️  GROQ_API_KEY non définie. Lance avec --no-ai ou exporte la clé:")
            print("    export GROQ_API_KEY='gsk_...'")
            print("Continuons sans pipeline IA.\n")
        else:
            groq_client = Groq(api_key=GROQ_API_KEY)

    cycle = 0
    while True:
        cycle += 1
        do_ai = (groq_client is not None
                 and (cycle % args.ai_every == 1 or args.ai_every <= 1))

        results = run_once(groq_client, do_ai, skip_rb=args.no_rb)

        if args.json:
            print(json.dumps([r.to_dict() for r in results],
                             indent=2, ensure_ascii=False))
        else:
            print(f"\n{'═'*88}")
            print(f"  📡  CYCLE #{cycle} — "
                  f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                  f"   {'(AI ON)' if do_ai else '(AI off)'}")
            print(f"{'═'*88}")
            print(render_table(results))
            if do_ai:
                print(f"\n{'─'*88}")
                print(f"  🎯  AUDIENCE PROFILES — predicted by {LLM_MODEL}")
                print(f"{'─'*88}")
                print(render_audience_details(results))

            if DEBUG:
                print("\n--- debug: per-station attempts ---")
                for r in results:
                    if r.attempts:
                        print(f"\n[{r.name}] resolved={r.resolved_origin}")
                        for a in r.attempts:
                            print(f"  {a}")

        try: log_csv(results)
        except Exception as e:
            print(f"  [csv error: {e}]")

        if args.watch <= 0:
            return 0
        try: time.sleep(args.watch)
        except KeyboardInterrupt: return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
