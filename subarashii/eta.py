"""Progress and time-left estimates.

Each stage's expected time = overhead + rate * units, where units are lines for the line-based
stages and audio minutes otherwise. Rates start from built-in guesses and are re-learned from
every finished stage (stats.json in the data dir), so estimates improve after the first run.
The current stage's remaining time comes from its own measured speed once it has made progress.
"""

import json
import time
from pathlib import Path

from .project import REVIEW_GATE, STAGES, Project

LABELS = {
    "ingest": "Reading the video and extracting audio",
    "separate": "Separating voices from music",
    "transcribe": "Transcribing the original dialogue",
    "diarize": "Telling characters apart by voice",
    "profile": "Analysing voices and building voice references",
    "proofread": "Proofreading the transcript (LLM)",
    "characters": "Working out who the characters are (LLM)",
    "translate": "Translating and fitting lines to the timing (LLM)",
    "check": "Checking the English script makes sense (LLM)",
    "synthesize": "Voicing the English lines",
    "mix": "Mixing the new soundtrack",
    "export": "Writing the final video",
}

# seconds = overhead + rate * units; units: "min" = audio minutes, "line" = dialogue lines
DEFAULTS = {
    "ingest": (2, 0.3, "min"),
    "separate": (20, 45.0, "min"),
    "transcribe": (30, 0.15, "line"),
    "diarize": (15, 0.05, "line"),
    "profile": (20, 0.2, "line"),
    "proofread": (10, 0.4, "line"),
    "characters": (40, 0.05, "line"),
    "translate": (10, 1.2, "line"),
    "check": (10, 0.6, "line"),
    "synthesize": (10, 6.0, "line"),
    "mix": (3, 0.5, "min"),
    "export": (3, 1.0, "min"),
}
LINES_PER_MIN = 12  # guess until the transcript exists


def _stats_path(job_root: Path) -> Path:
    return job_root.parent.parent / "stats.json"


def load_stats(job_root: Path) -> dict:
    try:
        return json.loads(_stats_path(job_root).read_text())
    except (OSError, ValueError):
        return {}


def units(p: Project, stage: str) -> float:
    minutes = max((p.media.get("duration") or 60) / 60, 0.1)
    if DEFAULTS[stage][2] == "min":
        return minutes
    if not p.lines:
        return minutes * LINES_PER_MIN
    if stage == "synthesize" and "synthesize" not in p.stages_done and any(l.tts for l in p.lines):
        # re-render: only lines without an up-to-date take
        return max(1, sum(1 for l in p.lines if (l.dirty or not l.tts) and not l.skip and not l.lyric))
    return len(p.lines)


def expected(p: Project, stage: str, stats: dict) -> float:
    overhead, rate, _ = DEFAULTS[stage]
    rate = stats.get(stage, rate)
    return overhead + rate * units(p, stage)


def learn(job_root: Path, p: Project, stage: str, elapsed: float, stage_units: float):
    """Fold a finished stage's real duration into the per-unit rate (moving average)."""
    if stage_units <= 0:
        return
    stats = load_stats(job_root)
    overhead, default_rate, _ = DEFAULTS[stage]
    measured = max(0.0, elapsed - overhead) / stage_units
    old = stats.get(stage, default_rate)
    stats[stage] = round(measured if stage not in stats else 0.5 * old + 0.5 * measured, 4)
    try:
        _stats_path(job_root).write_text(json.dumps(stats, indent=1))
    except OSError:
        pass


def estimate(p: Project, job_root: Path) -> dict:
    stats = load_stats(job_root)
    now = time.time()
    done_secs = sum(p.stage_times.get(s, 0.0) for s in p.stages_done)
    remaining = [s for s in STAGES if s not in p.stages_done]
    current = p.stage if p.status == "running" and p.stage in remaining else ""

    cur_left, cur_elapsed = 0.0, 0.0
    if current:
        cur_elapsed = max(0.0, now - p.stage_started) if p.stage_started else 0.0
        exp = expected(p, current, stats)
        prog = p.progress
        # blend the stage's expected time with its measured speed, trusting the latter more as
        # progress grows (a stage parked at 10% during a long call must not explode the ETA)
        by_expectation = max(exp - cur_elapsed, 0.05 * exp)
        if prog >= 0.05 and cur_elapsed >= 5:
            by_speed = cur_elapsed * (1 - prog) / prog
            cur_left = prog * by_speed + (1 - prog) * by_expectation
        else:
            cur_left = by_expectation
    future = [s for s in remaining if s != current]
    future_secs = {s: expected(p, s, stats) for s in future}
    left = cur_left + sum(future_secs.values())

    gate_pending = p.review and not p.review_passed and REVIEW_GATE in remaining
    until_review = None
    if gate_pending:
        before_gate = [s for s in future if STAGES.index(s) < STAGES.index(REVIEW_GATE)]
        until_review = cur_left + sum(future_secs[s] for s in before_gate)

    spent = done_secs + cur_elapsed
    overall = 1.0 if p.status == "done" else spent / (spent + left) if spent + left > 0 else 0.0
    step = p.stage if p.stage in STAGES else ""
    return {
        "overall": round(min(overall, 0.999 if p.status != "done" else 1.0), 4),
        "eta_seconds": round(left) if p.status in ("running", "queued") else None,
        "eta_review_seconds": round(until_review) if until_review is not None and p.status in ("running", "queued") else None,
        "step": step,
        "step_index": STAGES.index(step) + 1 if step else len(p.stages_done),
        "step_count": len(STAGES),
        "step_label": LABELS.get(step, ""),
        "step_progress": round(p.progress, 4),
        "step_elapsed_seconds": round(cur_elapsed) if current else None,
        "step_eta_seconds": round(cur_left) if current else None,
    }
