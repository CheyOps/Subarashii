"""Web UI + JSON API. One worker thread runs jobs sequentially (one GPU)."""

import queue
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import pipeline
from ..config import (LLM_FIELDS, SECRET_FIELDS, data_dir, deep_merge, load_config, load_settings,
                      overrides_from_pairs, save_settings)
from ..lang import LANGS
from ..eta import estimate
from ..project import STAGES, Job

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".m4v", ".avi", ".webm", ".ts", ".wmv", ".flv"}
STATIC = Path(__file__).with_name("static")

app = FastAPI(title="Subarashii")
_queue: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
_running = {"id": None, "stop": ""}  # stop: "" | "pause" | "pause_now" | "cancel"
_edit_lock = threading.Lock()


@app.middleware("http")
async def _no_cache(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-cache"  # UI files ship inside the image; always revalidate
    return resp


def _cfg():
    return load_config()


def _jobs_dir() -> Path:
    return data_dir(_cfg(), "jobs")


def _job(jid: str) -> Job:
    root = _jobs_dir() / Path(jid).name
    if not (root / "project.json").is_file():
        raise HTTPException(404, "job not found")
    return Job.load(root)


def _editable(job: Job):
    if job.p.status in ("running", "queued"):
        raise HTTPException(409, "job is running; wait for it to pause or finish")


# ------------------------------------------------------------------ worker
def _worker():
    while True:
        jid, from_stage = _queue.get()
        _running.update(id=jid, stop="")
        try:
            job = Job.load(_jobs_dir() / jid)
            if job.p.status == "queued":  # may have been paused or cancelled while waiting
                pipeline.run(job, from_stage=from_stage, cancel=lambda: _running["stop"])
        except Exception as e:  # noqa: BLE001
            print(f"worker error on {jid}: {e}", flush=True)
        finally:
            _running.update(id=None, stop="")


def _enqueue(job: Job, from_stage: str | None = None):
    job.p.status, job.p.message = "queued", "queued"
    job.save()
    _queue.put((job.p.id, from_stage))


@app.on_event("startup")
def _startup():
    # jobs interrupted by a restart go back in the queue
    for d in sorted(_jobs_dir().iterdir()):
        if (d / "project.json").is_file():
            job = Job.load(d)
            if job.p.status in ("running", "queued"):
                _enqueue(job)
    threading.Thread(target=_worker, daemon=True).start()


# ------------------------------------------------------------------ jobs
@app.get("/api/jobs")
def list_jobs():
    out = []
    for d in sorted(_jobs_dir().iterdir(), reverse=True):
        if (d / "project.json").is_file():
            job = Job.load(d)
            p = job.p
            row = {k: getattr(p, k) for k in ("id", "source_name", "created", "status", "stage", "progress", "message", "error", "output")}
            out.append({**row, **estimate(p, job.root)})
    return out


@app.get("/api/inputs")
def list_inputs():
    d = data_dir(_cfg(), "input")
    return sorted(f.name for f in d.iterdir() if f.suffix.lower() in VIDEO_EXT)


@app.get("/api/config")
def get_config():
    c = _cfg()
    return {"stages": STAGES, "provider": c.llm.provider,
            "models": {k: c.llm[k].get("model") for k in ("llamacpp", "openai", "anthropic")}}


PROVIDERS = tuple(LLM_FIELDS)


class LLMSettings(BaseModel):
    provider: str
    # provider -> field -> value; a blank api_key keeps the saved one
    providers: dict[str, dict[str, str | None]] = {}


@app.get("/api/settings/llm")
def get_llm_settings():
    llm = _cfg().llm
    out = {}
    for prov, fields in LLM_FIELDS.items():
        out[prov] = {f: llm[prov].get(f) or "" for f in fields if f not in SECRET_FIELDS}
        for f in fields:
            if f in SECRET_FIELDS:
                out[prov][f + "_set"] = bool(llm[prov].get(f))  # never send keys to the browser
    return {"provider": llm.provider, "providers": out}


@app.put("/api/settings/llm")
def put_llm_settings(req: LLMSettings):
    if req.provider not in PROVIDERS:
        raise HTTPException(400, f"unknown provider: {req.provider}")
    cfg = load_config()
    settings = load_settings(cfg)
    llm = settings.setdefault("llm", {})
    llm["provider"] = req.provider
    for prov, values in req.providers.items():
        if prov not in LLM_FIELDS:
            raise HTTPException(400, f"unknown provider: {prov}")
        for field, value in values.items():
            if field not in LLM_FIELDS[prov]:
                raise HTTPException(400, f"unknown setting: {prov}.{field}")
            if value is None or (field in SECRET_FIELDS and not value.strip()):
                continue  # a blank key box keeps the saved key; use DELETE to clear it
            llm.setdefault(prov, {})[field] = value.strip()
    save_settings(cfg, settings)
    return get_llm_settings()


@app.delete("/api/settings/llm/{provider}/api_key")
def clear_llm_key(provider: str):
    if provider not in PROVIDERS:
        raise HTTPException(404, "unknown provider")
    cfg = load_config()
    settings = load_settings(cfg)
    settings.get("llm", {}).get(provider, {}).pop("api_key", None)
    save_settings(cfg, settings)
    return get_llm_settings()


@app.post("/api/settings/llm/test")
def test_llm_settings(provider: str = ""):
    """Reach a provider (default: the saved one) with its saved settings and list the models it offers."""
    import httpx

    llm = _cfg().llm
    prov = provider or llm.provider
    if prov not in PROVIDERS:
        raise HTTPException(400, f"unknown provider: {prov}")
    c = getattr(llm, prov)
    try:
        if prov == "anthropic":
            import anthropic

            models = [m.id for m in anthropic.Anthropic(api_key=c.get("api_key") or None).models.list(limit=100)]
        else:
            if not c.url:
                raise HTTPException(400, "set the endpoint URL first")
            root = c.url.rstrip("/").removesuffix("/v1")
            headers = {"Authorization": f"Bearer {c.api_key}"} if c.get("api_key") else {}
            r = httpx.get(root + "/v1/models", headers=headers, timeout=15)
            if r.status_code >= 400:
                raise HTTPException(502, f"{prov} answered HTTP {r.status_code}: {r.text[:200]}")
            models = [m["id"] for m in r.json().get("data", [])]
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"cannot reach {prov}: {e}") from e
    return {"ok": True, "provider": prov, "models": models}


@app.post("/api/jobs")
def create_job(
    file: UploadFile | None = File(None),
    input_name: str = Form(""),
    review: bool = Form(False),
    language: str = Form("auto"),
    provider: str = Form(""),
    model: str = Form(""),
    settings: str = Form(""),  # newline separated key=value
):
    cfg = _cfg()
    inputs = data_dir(cfg, "input")
    if file is not None and file.filename:
        name = Path(file.filename).name
        if Path(name).suffix.lower() not in VIDEO_EXT:
            raise HTTPException(400, "unsupported file type")
        src = inputs / name
        with src.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    elif input_name:
        src = inputs / Path(input_name).name
        if not src.is_file():
            raise HTTPException(404, "input file not found")
    else:
        raise HTTPException(400, "upload a file or choose one from the input folder")
    ov = overrides_from_pairs([s for s in settings.splitlines() if "=" in s])
    if language and language != "auto":
        if language not in LANGS:
            raise HTTPException(400, f"unsupported language: {language}")
        ov = deep_merge(ov, {"input": {"language": language}})
    if provider:
        ov = deep_merge(ov, {"llm": {"provider": provider}})
        if model:
            ov = deep_merge(ov, {"llm": {provider: {"model": model}}})
    job = Job.create(data_dir(cfg, "jobs"), src, review=review, overrides=ov)
    _enqueue(job)
    return {"id": job.p.id}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    job = _job(jid)
    return {**job.p.model_dump(), "estimate": estimate(job.p, job.root)}


@app.delete("/api/jobs/{jid}")
def delete_job(jid: str):
    job = _job(jid)
    if _running["id"] == job.p.id:
        raise HTTPException(409, "job is running; cancel it first")
    shutil.rmtree(job.root)
    return {"ok": True}


def _stop_job(jid: str, how: str, status: str):
    if _running["id"] == jid:
        _running["stop"] = how
        return {"ok": True, "note": "stopping"}
    job = _job(jid)
    if job.p.status == "queued":
        job.p.status = status
        job.save()
    return {"ok": True}


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    return _stop_job(jid, "cancel", "cancelled")


@app.post("/api/jobs/{jid}/pause")
def pause_job(jid: str, now: bool = False):
    """Stop after the current stage (or right away with now=true, losing that stage's progress);
    Run/resume continues from the first unfinished stage."""
    return _stop_job(jid, "pause_now" if now else "pause", "paused")


class RunRequest(BaseModel):
    from_stage: str | None = None
    approve: bool = False


@app.post("/api/jobs/{jid}/run")
def run_job(jid: str, req: RunRequest):
    with _edit_lock:
        job = _job(jid)
        _editable(job)
        if req.from_stage and req.from_stage not in STAGES:
            raise HTTPException(400, "unknown stage")
        if req.approve or job.p.status == "review":
            job.p.review_passed = True
        if req.from_stage:
            job.invalidate_from(req.from_stage)
            if req.from_stage == "synthesize":
                for l in job.p.lines:
                    l.dirty = True
        _enqueue(job)
    return {"ok": True}


@app.post("/api/jobs/{jid}/rerender")
def rerender(jid: str):
    """Re-synthesize edited/dirty lines only, then mix + export."""
    with _edit_lock:
        job = _job(jid)
        _editable(job)
        job.p.review_passed = True
        job.invalidate_from("synthesize")
        _enqueue(job)
    return {"ok": True}


class SpeakerPatch(BaseModel):
    name: str | None = None
    gender: str | None = None
    age_group: str | None = None
    personality: str | None = None
    speech_style: str | None = None
    notes: str | None = None
    merge_into: str | None = None
    ref_clips: list[int] | None = None


@app.patch("/api/jobs/{jid}/speakers/{sid}")
def patch_speaker(jid: str, sid: str, patch: SpeakerPatch):
    with _edit_lock:
        job = _job(jid)
        _editable(job)
        spk = job.p.speakers.get(sid)
        if spk is None:
            raise HTTPException(404, "speaker not found")
        if patch.merge_into:
            target = job.p.speakers.get(patch.merge_into)
            if target is None or patch.merge_into == sid:
                raise HTTPException(400, "bad merge target")
            for l in job.p.lines:
                if l.speaker == sid:
                    l.speaker, l.dirty = patch.merge_into, True
            del job.p.speakers[sid]
            target.line_count += spk.line_count
            target.total_seconds += spk.total_seconds
        else:
            for field in ("name", "gender", "age_group", "personality", "speech_style", "notes"):
                v = getattr(patch, field)
                if v is not None:
                    setattr(spk, field, v)
            if patch.ref_clips is not None:
                _set_reference(job, spk, patch.ref_clips)
                for l in job.p.lines:
                    if l.speaker == sid:
                        l.dirty = True
            spk.user_edited = True
        job.save()
        return job.p.model_dump()


def _set_reference(job: Job, spk, clip_ids: list[int]):
    import numpy as np

    from .. import audio

    lines = {l.id: l for l in job.p.lines}
    parts = []
    sr = None
    for cid in clip_ids:
        l = lines.get(cid)
        if l is None or not l.ja_clip:
            raise HTTPException(400, f"line {cid} has no clip")
        y, sr = audio.load(job.path(l.ja_clip), mono=True)
        parts += [y, np.zeros(int(0.25 * sr), dtype=np.float32)]
    if not parts:
        raise HTTPException(400, "choose at least one clip")
    ref = np.concatenate(parts)
    ref = ref * (0.9 / max(float(np.abs(ref).max()), 1e-6))
    audio.save(job.path(spk.ref_audio or f"speakers/{spk.id}_ref.wav"), ref, sr)
    spk.ref_audio = spk.ref_audio or f"speakers/{spk.id}_ref.wav"
    spk.ref_clips = clip_ids


class NewSpeaker(BaseModel):
    from_line: int


@app.post("/api/jobs/{jid}/speakers")
def new_speaker(jid: str, req: NewSpeaker):
    """Split a line off into a brand-new speaker (its own clip becomes the voice reference)."""
    with _edit_lock:
        job = _job(jid)
        _editable(job)
        from ..project import Speaker

        n = 1
        while f"S{n:02d}" in job.p.speakers:
            n += 1
        sid = f"S{n:02d}"
        line = next((l for l in job.p.lines if l.id == req.from_line), None)
        if line is None:
            raise HTTPException(404, "line not found")
        spk = Speaker(id=sid, user_edited=True, line_count=1, total_seconds=round(line.duration, 2))
        job.p.speakers[sid] = spk
        line.speaker, line.dirty = sid, True
        _set_reference(job, spk, [line.id])
        job.save()
        return job.p.model_dump()


class LinePatch(BaseModel):
    en: str | None = None
    speaker: str | None = None
    skip: bool | None = None
    lyric: bool | None = None


@app.patch("/api/jobs/{jid}/lines/{lid}")
def patch_line(jid: str, lid: int, patch: LinePatch):
    with _edit_lock:
        job = _job(jid)
        _editable(job)
        line = next((l for l in job.p.lines if l.id == lid), None)
        if line is None:
            raise HTTPException(404, "line not found")
        if patch.en is not None and patch.en.strip() != line.en.strip():
            line.en, line.en_edited, line.dirty = patch.en.strip(), True, True
        if patch.speaker is not None and patch.speaker != line.speaker:
            if patch.speaker not in job.p.speakers:
                raise HTTPException(400, "unknown speaker")
            line.speaker, line.dirty = patch.speaker, True
        if patch.skip is not None:
            line.skip = patch.skip
        if patch.lyric is not None:
            line.lyric = patch.lyric
        job.save()
        return line.model_dump()


# ------------------------------------------------------------------ files
@app.get("/api/jobs/{jid}/file/{rel:path}")
def job_file(jid: str, rel: str):
    job = _job(jid)
    path = (job.root / rel).resolve()
    if not path.is_relative_to(job.root.resolve()) or not path.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(path)


@app.get("/api/jobs/{jid}/download")
def download(jid: str):
    job = _job(jid)
    if not job.p.output or not Path(job.p.output).is_file():
        raise HTTPException(404, "no output yet")
    return FileResponse(job.p.output, filename=Path(job.p.output).name)


@app.get("/api/jobs/{jid}/source")
def source(jid: str):
    job = _job(jid)
    return FileResponse(job.p.source)


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
