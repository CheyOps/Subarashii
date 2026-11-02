"""IndexTTS-2.5 HTTP wrapper.

The model lives in a child process: /unload terminates it, which returns *all* of its VRAM
(including the CUDA context) so the local LLM can load. The next /synthesize starts it again.
Audio paths are shared-volume paths (both containers mount /data at the same place).
"""

import multiprocessing as mp
import os
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODEL_DIR = os.environ.get("INDEXTTS_MODEL_DIR", "/assets/models/indextts-2.5")
HF_REPO = os.environ.get("INDEXTTS_REPO", "IndexTeam/IndexTTS-2.5")
USE_BF16 = os.environ.get("INDEXTTS_BF16", "1") == "1"

app = FastAPI(title="Subarashii TTS")
_lock = threading.Lock()


def _worker_main(conn):
    import random

    import numpy as np
    import torch

    try:
        if not os.path.isfile(os.path.join(MODEL_DIR, "config.yaml")):
            from huggingface_hub import snapshot_download

            os.makedirs(MODEL_DIR, exist_ok=True)
            snapshot_download(repo_id=HF_REPO, local_dir=MODEL_DIR)
        from indextts.infer_v2_5 import IndexTTS2

        tts = IndexTTS2(
            cfg_path=os.path.join(MODEL_DIR, "config.yaml"),
            model_dir=MODEL_DIR,
            use_bf16=USE_BF16,
            use_cuda_kernel=False,
        )
    except Exception as e:  # noqa: BLE001
        conn.send(("error", f"model load failed: {e!r}"))
        return
    conn.send(("ready", None))

    while True:
        r = conn.recv()
        try:
            if r.get("seed") is not None:
                seed = int(r["seed"])
                random.seed(seed)
                np.random.seed(seed % (2**32))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            tts.infer(
                spk_audio_prompt=r["spk_audio"],
                text=r["text"],
                output_path=r["output"],
                lang=r["lang"],
                emo_audio_prompt=r.get("emo_audio"),
                emo_alpha=r["emo_alpha"],
                emo_vector=r.get("emo_vector"),
                duration_factor=max(0.5, min(2.0, r["duration_factor"])),
                text_normalization=r["text_normalization"],
                verbose=False,
            )
            conn.send(("ok", None))
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            conn.send(("oom", str(e)))
        except Exception as e:  # noqa: BLE001
            conn.send(("error", repr(e)))


class _Worker:
    def __init__(self):
        self.proc = None
        self.conn = None

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()

    def start(self):
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker_main, args=(child,), daemon=True)
        self.proc.start()
        self.conn = parent
        status, msg = self._recv()
        if status != "ready":
            self.stop()
            raise HTTPException(500, msg)

    def _recv(self):
        try:
            return self.conn.recv()
        except (EOFError, OSError):
            code = self.proc.exitcode if self.proc else None
            self.stop()
            raise HTTPException(500, f"TTS worker died (exit code {code}; out of memory?)")

    def call(self, req: dict):
        if not self.alive:
            self.start()
        self.conn.send(req)
        status, msg = self._recv()
        if status == "oom":
            raise HTTPException(507, f"CUDA out of memory: {msg}")
        if status != "ok":
            raise HTTPException(500, msg)

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
            self.proc.join(15)
            if self.proc.is_alive():
                self.proc.kill()
        self.proc, self.conn = None, None


_worker = _Worker()


class SynthRequest(BaseModel):
    text: str
    output: str
    spk_audio: str
    emo_audio: str | None = None
    emo_alpha: float = 0.8
    emo_vector: list[float] | None = None
    lang: str = "EN"
    duration_factor: float = 1.0
    seed: int | None = None
    text_normalization: bool = True


@app.get("/health")
def health():
    return {"ok": True, "loaded": _worker.alive}


@app.post("/load")
def load():
    with _lock:
        if not _worker.alive:
            _worker.start()
    return {"loaded": True}


@app.post("/unload")
def unload():
    with _lock:
        _worker.stop()
    return {"loaded": False}


@app.post("/synthesize")
def synthesize(r: SynthRequest):
    import soundfile as sf

    for p in (r.spk_audio, r.emo_audio):
        if p and not os.path.isfile(p):
            raise HTTPException(400, f"missing audio file: {p}")
    os.makedirs(os.path.dirname(r.output) or ".", exist_ok=True)
    with _lock:
        _worker.call(r.model_dump())
    if not os.path.isfile(r.output):
        raise HTTPException(500, "TTS produced no output")
    info = sf.info(r.output)
    return {"output": r.output, "duration": info.duration, "sample_rate": info.samplerate}
