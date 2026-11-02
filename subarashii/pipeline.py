"""Runs stages in order, resumably. Each finished stage is recorded in project.json."""

import importlib
import multiprocessing as mp
import time
import traceback
from pathlib import Path

from . import eta
from .config import Cfg, load_config
from .gpu import free_gpu
from .project import REVIEW_GATE, STAGES, Cancelled, Job

# stages that need the GPU for in-process models: free the TTS/LLM first
_GPU_STAGES = {"separate", "transcribe", "diarize", "profile"}


class StageError(RuntimeError):
    pass


def _stage_child(root: str, stage: str):
    job = Job.load(Path(root))
    try:
        importlib.import_module(f".stages.{stage}", __package__).run(job, job_config(job))
        job.save()
    except BaseException as e:  # noqa: BLE001
        job.save()
        (Path(root) / "stage_error.txt").write_text(f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
        raise SystemExit(1)


def _run_in_child(job: Job, stage: str):
    """Run a GPU stage in its own process so every byte of VRAM (CUDA context included) is
    returned when it ends; otherwise the local LLM may not fit afterwards."""
    err_file = job.path("stage_error.txt")
    err_file.unlink(missing_ok=True)
    proc = mp.get_context("spawn").Process(target=_stage_child, args=(str(job.root), stage))
    proc.start()
    while proc.is_alive():
        proc.join(1.0)
        stop = job.stop_requested()
        if stop in ("cancel", "pause_now"):
            proc.terminate()  # frees all of its VRAM, same as a normal exit
            proc.join()
            job.reload()
            raise Cancelled(stop)
    job.reload()
    if proc.exitcode != 0:
        detail = err_file.read_text() if err_file.is_file() else f"stage process exited with code {proc.exitcode}"
        print(detail, flush=True)
        raise StageError(detail.split("\n", 1)[0])


def job_config(job: Job) -> Cfg:
    return load_config(job.p.overrides)


def run(job: Job, from_stage: str | None = None, stop_at_review: bool = True, cancel=lambda: "") -> str:
    """Run until done, error, or the review gate. Returns the final status."""
    cfg = job_config(job)
    if from_stage:
        job.invalidate_from(from_stage)
    job.p.status, job.p.error = "running", ""
    job.stop_requested = cancel
    job.save()
    try:
        for stage in STAGES:
            if stage in job.p.stages_done:
                continue
            if stage == REVIEW_GATE and job.p.review and stop_at_review and not job.p.review_passed:
                job.p.status, job.p.stage, job.p.message = "review", stage, "waiting for review"
                job.log("paused for review: edit speakers/lines, then continue")
                job.save()
                return job.p.status
            stop = cancel()
            if stop:
                raise Cancelled(stop)
            job.p.stage, job.p.progress, job.p.message = stage, 0.0, ""
            job.p.stage_started = time.time()
            stage_units = eta.units(job.p, stage)
            job.log(f"stage {stage} started")
            job.save()
            if stage in _GPU_STAGES:
                free_gpu(cfg)
                _run_in_child(job, stage)
            else:
                importlib.import_module(f".stages.{stage}", __package__).run(job, cfg)
            elapsed = time.time() - job.p.stage_started
            job.p.stage_times[stage] = round(elapsed, 1)
            eta.learn(job.root, job.p, stage, elapsed, stage_units)
            job.p.stages_done.append(stage)
            job.progress(1.0, f"{stage} done", force=True)
        job.p.status, job.p.stage, job.p.message = "done", "", "finished"
        free_gpu(cfg)
    except Cancelled as e:
        paused = str(e).startswith("pause")
        job.p.status, job.p.message = ("paused", "paused") if paused else ("cancelled", "cancelled")
        job.log("paused" if paused else "cancelled")
    except Exception as e:  # noqa: BLE001 - surface every failure in the job
        job.p.status, job.p.error = "error", f"{type(e).__name__}: {e}"
        job.log(f"stage {job.p.stage} failed: {e}")
        print(traceback.format_exc(), flush=True)
        try:
            free_gpu(cfg)
        except Exception:
            pass
    job.save()
    return job.p.status


def approve_review(job: Job):
    job.p.review_passed = True
    job.save()
