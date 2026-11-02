"""Job state. Everything about a job lives in <data>/jobs/<id>/project.json plus audio files next to it."""

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

STAGES = [
    "ingest",
    "separate",
    "transcribe",
    "diarize",
    "profile",
    "proofread",
    "characters",
    "translate",
    "check",
    "synthesize",
    "mix",
    "export",
]
REVIEW_GATE = "synthesize"  # with review on, the pipeline pauses before this stage


class TTSTake(BaseModel):
    file: str = ""
    text: str = ""
    duration: float = 0.0
    duration_factor: float = 1.0
    stretch: float = 1.0
    gain_db: float = 0.0
    fit: str = ""  # ok | slowed | sped | stretched | overflow
    seed: int = 0


class SepRef(BaseModel):
    """One voice separated out of crosstalk: `file` (16 kHz, job-relative) covers [start, end] of the timeline."""

    file: str
    start: float
    end: float


class Line(BaseModel):
    id: int
    start: float
    end: float
    speaker: str = ""
    speaker_conf: float = 0.0
    ja: str = ""
    ja_asr: str = ""
    ja_alt: str = ""  # second recognizer's hypothesis, used by proofreading
    ja_clip: str = ""
    en: str = ""
    candidates: dict[str, str] = Field(default_factory=dict)
    en_edited: bool = False
    skip: bool = False
    lyric: bool = False
    syl_target: int = 0
    syl_max: int = 0
    slot_end: float = 0.0
    tts: TTSTake | None = None
    dirty: bool = False
    check_note: str = ""  # what the script check changed or still doubts
    check_unresolved: bool = False
    sep: list[SepRef] = Field(default_factory=list)  # set where this line was spoken over another

    @property
    def duration(self) -> float:
        return self.end - self.start


class Speaker(BaseModel):
    id: str
    name: str = ""
    gender: str = ""
    age_group: str = ""
    personality: str = ""
    speech_style: str = ""
    notes: str = ""
    acoustic: dict = Field(default_factory=dict)
    ref_audio: str = ""
    thumbs: list[str] = Field(default_factory=list)
    name_confident: bool = False
    ref_clips: list[int] = Field(default_factory=list)
    total_seconds: float = 0.0
    line_count: int = 0
    user_edited: bool = False

    @property
    def label(self) -> str:
        return self.name or self.id


class Project(BaseModel):
    id: str
    source: str
    source_name: str
    created: str
    status: str = "queued"  # queued | running | paused | review | done | error | cancelled
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    error: str = ""
    review: bool = False
    review_passed: bool = False
    stages_done: list[str] = Field(default_factory=list)
    stage_started: float = 0.0  # epoch seconds when the current stage began
    stage_times: dict[str, float] = Field(default_factory=dict)  # seconds each finished stage took
    overrides: dict = Field(default_factory=dict)
    media: dict = Field(default_factory=dict)
    language: str = ""  # source language code (lang.LANGS); set at ingest or detected while transcribing
    setting: str = ""
    glossary: list[dict] = Field(default_factory=list)
    speakers: dict[str, Speaker] = Field(default_factory=dict)
    lines: list[Line] = Field(default_factory=list)
    output: str = ""
    log: list[str] = Field(default_factory=list)


_save_lock = threading.Lock()


class Cancelled(BaseException):
    """Raised to stop a job. BaseException so a stage's `except Exception` can't swallow it.
    The message is the stop request: "cancel", "pause_now" (interrupt the running step) or "pause" (after it)."""


class Job:
    def __init__(self, root: Path, project: Project):
        self.root = root
        self.p = project
        self._last_save = 0.0
        self.stop_requested = lambda: ""  # set by the pipeline: "" | "pause" | "pause_now" | "cancel"

    # ------------------------------------------------------------- creation / io
    @classmethod
    def create(cls, jobs_dir: Path, source: Path, review: bool = False, overrides: dict | None = None):
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem)[:40].strip("_") or "video"
        jid = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{stem}-{uuid.uuid4().hex[:4]}"
        root = jobs_dir / jid
        root.mkdir(parents=True)
        p = Project(
            id=jid,
            source=str(source),
            source_name=source.name,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            review=review,
            overrides=overrides or {},
        )
        job = cls(root, p)
        job.save()
        return job

    @classmethod
    def load(cls, root: Path) -> "Job":
        return cls(root, Project.model_validate_json((root / "project.json").read_text()))

    def reload(self):
        self.p = Project.model_validate_json((self.root / "project.json").read_text())

    def save(self):
        with _save_lock:
            tmp = self.root / "project.json.tmp"
            tmp.write_text(self.p.model_dump_json(indent=1))
            os.replace(tmp, self.root / "project.json")
        self._last_save = time.monotonic()

    def path(self, rel: str) -> Path:
        return self.root / rel

    def rel(self, path: Path) -> str:
        return str(Path(path).relative_to(self.root))

    # ------------------------------------------------------------- status
    def progress(self, frac: float, message: str = "", force: bool = False):
        stop = self.stop_requested()
        if stop in ("cancel", "pause_now"):  # stages report progress often, so this stops them promptly
            raise Cancelled(stop)
        self.p.progress = max(0.0, min(1.0, frac))
        if message:
            self.p.message = message
        if force or time.monotonic() - self._last_save > 1.0:
            self.save()

    def log(self, msg: str):
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(f"[{self.p.id}] {line}", flush=True)
        self.p.log.append(line)
        self.p.log = self.p.log[-400:]

    def invalidate_from(self, stage: str):
        idx = STAGES.index(stage)
        if idx < STAGES.index(REVIEW_GATE):
            self.p.review_passed = False
        self.p.stages_done = [s for s in self.p.stages_done if STAGES.index(s) < idx]

    # ------------------------------------------------------------- helpers
    def speaker(self, sid: str) -> Speaker | None:
        return self.p.speakers.get(sid)

    def active_lines(self) -> list[Line]:
        return [l for l in self.p.lines if not l.skip and not l.lyric and l.en.strip()]
