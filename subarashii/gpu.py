"""VRAM handoff between stages. Only one large model should be resident on the GPU at a time."""

import gc
import time

import httpx

from .config import Cfg


class TTSClient:
    def __init__(self, cfg: Cfg):
        self.url = cfg.tts.url.rstrip("/")
        self.http = httpx.Client(base_url=self.url, timeout=cfg.tts.timeout)

    def health(self) -> dict:
        return self.http.get("/health", timeout=10).json()

    def unload(self):
        try:
            self.http.post("/unload", timeout=60)
        except httpx.HTTPError:
            pass

    def synthesize(self, **req) -> dict:
        for attempt in range(3):
            try:
                r = self.http.post("/synthesize", json=req)
                break
            except httpx.TransportError:
                if attempt == 2:
                    raise
                time.sleep(10)
        if r.status_code >= 400:
            raise RuntimeError(f"TTS error {r.status_code}: {r.text[:500]}")
        return r.json()


def empty_torch_cache():
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def free_gpu(cfg: Cfg, keep: str = ""):
    """Unload the TTS model and the local LLM unless the caller is about to use it."""
    if keep != "tts":
        TTSClient(cfg).unload()
    if keep != "llm":
        try:
            from .llm import get_llm

            get_llm(cfg).release()
        except Exception:
            pass
    empty_torch_cache()
