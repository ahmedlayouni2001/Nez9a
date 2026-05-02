"""Requirments:
pytrends>=4.9.2
requests>=2.31.0
urllib3<2"
"""

"""
Hit Detection System — exact-time edition
-----------------------------------------
Inputs : product/keyword name + exact event datetime
Outputs: Google Trends interest at that time, before-vs-after growth,
                 and verdict (Hit Yes/No/Unclear, strength Low/Medium/High).

Free source only:
    - Google Trends (pytrends)        -> search interest (hourly + daily)
"""

import statistics
import sys
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional

import argparse

from pytrends.request import TrendReq


# ============================================================ #
# Data classes
# ============================================================ #
@dataclass
class SourceResult:
    name: str
    granularity: str = "n/a"            # "hourly" | "daily" | "count"
    value_at_event: float = 0.0          # interest/views/posts at the event time
    peak_after: float = 0.0
    peak_after_time: Optional[str] = None
    before_avg: float = 0.0
    after_avg: float = 0.0
    growth_pct: float = 0.0              # (after - before) / before * 100
    spike_ratio: float = 0.0             # after / before
    trend_direction: str = "unknown"     # rising | stable | declining
    sample_size: int = 0
    raw_notes: str = ""


@dataclass
class HitReport:
    keyword: str
    event_time: datetime
    before_days: int
    after_days: int
    sources: list = field(default_factory=list)
    hit_detected: str = "Unclear"        # Yes | No | Unclear
    strength: str = "Low"                # Low | Medium | High
    evidence: list = field(default_factory=list)


# ============================================================ #
# Helpers
# ============================================================ #
def _direction(values) -> str:
    """Return rising/stable/declining for the second half of a series."""
    if len(values) < 4:
        return "stable"
    mid = len(values) // 2
    first = statistics.mean(values[:mid])
    second = statistics.mean(values[mid:])
    if first == 0 and second == 0:
        return "stable"
    if first == 0:
        return "rising"
    if second > first * 1.15:
        return "rising"
    if second < first * 0.85:
        return "declining"
    return "stable"


def _growth(before: float, after: float):
    """Return (spike_ratio, growth_pct)."""
    if before > 0:
        return after / before, (after - before) / before * 100
    if after > 0:
        return float("inf"), float("inf")
    return 0.0, 0.0


# ============================================================ #
# 1. Google Trends — TWO passes: hourly (precise) + daily (broad)
# ============================================================ #
def check_google_trends(keyword: str,
                        event_time: datetime,
                        before_days: int,
                        after_days: int,
                        geo: str = "") -> SourceResult:
    """
    Tries hourly resolution first (only works if total window <= ~8 days).
    Always also runs a daily pass for the configured baseline window.
    Returns one merged SourceResult, preferring hourly numbers when available.
    """
    res = SourceResult(name="Google Trends")
    pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.5)

    # ---------- DAILY pass (always run; uses the user's full window) ----------
    daily_series = None
    try:
        d_start = (event_time - timedelta(days=before_days)).strftime("%Y-%m-%d")
        d_end   = (event_time + timedelta(days=after_days)).strftime("%Y-%m-%d")
        pytrends.build_payload([keyword], timeframe=f"{d_start} {d_end}", geo=geo)
        df = pytrends.interest_over_time()
        if not df.empty:
            df = df.drop(columns=["isPartial"], errors="ignore")
            daily_series = df[keyword]
    except Exception as e:
        res.raw_notes = f"daily error: {e}; "

    # ---------- HOURLY pass (only if window fits) ----------
    hourly_series = None
    total_span_days = before_days + after_days
    if total_span_days <= 8:
        try:
            h_start = (event_time - timedelta(days=before_days)).strftime("%Y-%m-%dT%H")
            h_end   = (event_time + timedelta(days=after_days)).strftime("%Y-%m-%dT%H")
            pytrends.build_payload([keyword], timeframe=f"{h_start} {h_end}", geo=geo)
            df = pytrends.interest_over_time()
            if not df.empty:
                df = df.drop(columns=["isPartial"], errors="ignore")
                hourly_series = df[keyword]
        except Exception as e:
            res.raw_notes += f"hourly error: {e}"

    # ---------- Pick the best series ----------
    series, granularity = (None, "n/a")
    if hourly_series is not None and len(hourly_series) > 0:
        series, granularity = hourly_series, "hourly"
    elif daily_series is not None and len(daily_series) > 0:
        series, granularity = daily_series, "daily"
    else:
        res.raw_notes += " no data."
        return res

    # ---------- Compute metrics ----------
    idx = series.index.tz_localize(None) if series.index.tz else series.index
    before = series[idx <  event_time]
    after  = series[idx >= event_time]

    res.granularity   = granularity
    res.sample_size   = len(series)
    res.before_avg    = float(before.mean()) if len(before) else 0.0
    res.after_avg     = float(after.mean())  if len(after)  else 0.0

    if len(after) > 0:
        res.peak_after      = float(after.max())
        res.peak_after_time = str(after.idxmax())
        # Interest at the exact event timestamp = first sample at/after event
        res.value_at_event  = float(after.iloc[0])

    res.spike_ratio, res.growth_pct = _growth(res.before_avg, res.after_avg)
    res.trend_direction = _direction(list(after.values)) if len(after) else "unknown"

    return res


# ============================================================ #
# Verdict
# ============================================================ #
def classify_hit(report: HitReport) -> HitReport:
    valid = [s for s in report.sources if s.sample_size > 0]
    if not valid:
        report.hit_detected = "Unclear"
        report.evidence.append("No usable data from any source.")
        return report

    ratios = [s.spike_ratio if s.spike_ratio != float("inf") else 10.0 for s in valid]
    high_n   = sum(1 for r in ratios if r >= 3.0)
    medium_n = sum(1 for r in ratios if r >= 2.0)
    huge     = any(r >= 5.0 for r in ratios)

    if huge or high_n >= 2:
        report.hit_detected, report.strength = "Yes", "High"
    elif high_n >= 1 or medium_n >= 2:
        report.hit_detected, report.strength = "Yes", "Medium"
    elif any(r >= 1.2 for r in ratios):
        report.hit_detected, report.strength = "Yes", "Low"
    else:
        report.hit_detected, report.strength = "No", "Low"

    for s in valid:
        if s.spike_ratio >= 1.2:
            growth_str = "+inf%" if s.growth_pct == float("inf") else f"{s.growth_pct:+.1f}%"
            report.evidence.append(
                f"{s.name} ({s.granularity}): {s.spike_ratio:.2f}x growth ({growth_str}), "
                f"trend={s.trend_direction}"
            )
        else:
            report.evidence.append(f"{s.name}: no notable change.")
    return report


# ============================================================ #
# Public API
# ============================================================ #
def detect_hit(keyword: str,
               event_time: datetime,
               before_days: int = 30,
               after_days: int = 7,
               geo: str = "") -> HitReport:
    """
    Args:
        keyword:      product/article/word to track
        event_time:   exact datetime of the airing/release
        before_days:  baseline length BEFORE the event (default 30)
        after_days:   measurement length AFTER the event (default 7)
        geo:          "" worldwide, or country code like "US", "TN"
    """
    report = HitReport(keyword=keyword,
                       event_time=event_time,
                       before_days=before_days,
                       after_days=after_days)

    print(f"[+] Google Trends — '{keyword}' "
          f"({before_days}d before vs {after_days}d after)...")
    report.sources.append(
        check_google_trends(keyword, event_time, before_days, after_days, geo)
    )

    return classify_hit(report)


# ============================================================ #
# Pretty printer
# ============================================================ #
def print_report(report: HitReport):
    print("\n" + "=" * 64)
    print(f"HIT REPORT — '{report.keyword}'")
    print(f"Event time : {report.event_time}")
    print(f"Window     : {report.before_days}d before  vs  {report.after_days}d after")
    print("=" * 64)

    for s in report.sources:
        print(f"\n[{s.name}]  granularity = {s.granularity}")
        if s.granularity == "count":
            print(f"  posts in 1st hour after event   = {int(s.value_at_event)}")
        else:
            print(f"  interest at event time          = {s.value_at_event:.1f}")
        print(f"  peak after event                = {s.peak_after:.1f}"
              + (f"  @ {s.peak_after_time}" if s.peak_after_time else ""))
        unit = "day" if s.granularity == "count" else "sample"
        print(f"  before avg (per-{unit})            = {s.before_avg:.2f}")
        print(f"  after  avg (per-{unit})            = {s.after_avg:.2f}")
        growth_str = "+inf%" if s.growth_pct == float("inf") else f"{s.growth_pct:+.1f}%"
        print(f"  growth                          = {growth_str}  ({s.spike_ratio:.2f}x)")
        print(f"  direction                       = {s.trend_direction}")
        print(f"  samples                         = {s.sample_size}")
        if s.raw_notes:
            print(f"  notes                           = {s.raw_notes}")

    print("\n" + "-" * 64)
    print(f"Hit detected      : {report.hit_detected}")
    print(f"Strength of impact: {report.strength}")
    print("Evidence:")
    for e in report.evidence:
        print(f"  - {e}")
    print("=" * 64)


# ============================================================ #
# CLI
# ============================================================ #
def _parse_event_time(raw_value: str) -> datetime:
    raw_value = raw_value.strip()
    formats = (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in formats:
        try:
            parsed = datetime.strptime(raw_value, fmt)
            if fmt == "%Y-%m-%d":
                return parsed.replace(hour=0, minute=0, second=0)
            return parsed
        except ValueError:
            continue
    raise ValueError(
        "Invalid time format. Use YYYY-MM-DD, YYYY-MM-DD HH:MM, or YYYY-MM-DD HH:MM:SS."
    )


def _prompt_text(prompt: str, title: str) -> str:
    if sys.stdin is not None and sys.stdin.isatty():
        return input(prompt).strip()

    try:
        import tkinter as tk
        from tkinter import simpledialog
    except Exception as exc:
        raise RuntimeError(
            "Interactive input is not available. Run this script in a terminal or pass --keyword and --event-time."
        ) from exc

    root = tk.Tk()
    root.withdraw()
    try:
        value = simpledialog.askstring(title, prompt, parent=root)
    finally:
        root.destroy()

    if value is None:
        raise RuntimeError("Input cancelled.")
    return value.strip()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Google Trends hit detector")
    parser.add_argument("--keyword", help="Keyword or article title to search for")
    parser.add_argument(
        "--event-time",
        dest="event_time",
        help="Event time in YYYY-MM-DD, YYYY-MM-DD HH:MM, or YYYY-MM-DD HH:MM:SS format",
    )
    parser.add_argument("--before-days", type=int, default=30, help="Days before the event")
    parser.add_argument("--after-days", type=int, default=7, help="Days after the event")
    parser.add_argument("--geo", default="", help="Optional Google Trends region code")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    keyword = args.keyword or _prompt_text("Enter the keyword or article title: ", "Keyword")
    event_time_input = args.event_time or _prompt_text(
        "Enter the event time (YYYY-MM-DD HH:MM or YYYY-MM-DD HH:MM:SS): ",
        "Event Time",
    )

    if not keyword:
        raise ValueError("Keyword/article title cannot be empty.")

    event_time = _parse_event_time(event_time_input)

    report = detect_hit(
        keyword=keyword,
        event_time=event_time,
        before_days=args.before_days,
        after_days=args.after_days,
        geo=args.geo,
    )
    print_report(report)


if __name__ == "__main__":
    main()
