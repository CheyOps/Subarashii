"""Split the soundtrack into dialogue vocals and everything else (music + SFX).

The audio is separated in chunks (with crossfaded overlaps) so the stage can report progress
and long recordings don't need the whole track in VRAM at once.
"""

import shutil
from pathlib import Path

import numpy as np

from .. import audio
from ..config import Cfg, models_dir
from ..gpu import empty_torch_cache
from ..project import Job

OVERLAP = 2.0  # seconds of context on each side of a chunk, crossfaded with the neighbour


def _stems(files: list[Path]) -> tuple[Path, Path]:
    vocals = next((f for f in files if "(vocals)" in f.name.lower()), None)
    inst = next((f for f in files if f is not vocals and "(instrumental)" in f.name.lower()), None)
    if inst is None:
        inst = next((f for f in files if f is not vocals), None)
    if vocals is None or inst is None:
        raise RuntimeError(f"unexpected separator outputs: {[f.name for f in files]}")
    return vocals, inst


def _fit(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, ((0, n - len(y)), (0, 0)))


def run(job: Job, cfg: Cfg):
    from audio_separator.separator import Separator

    sr = cfg.mix.sample_rate
    work = job.path("sep")
    work.mkdir(exist_ok=True)
    sep = Separator(
        output_dir=str(work),
        model_file_dir=str(models_dir(cfg) / "separator"),
        output_format="WAV",
        sample_rate=sr,
    )
    job.progress(0.0, f"loading separation model ({cfg.separation.model})", force=True)
    sep.load_model(model_filename=cfg.separation.model)

    y, _ = audio.load(job.path("audio.wav"), sr=sr)
    n = len(y)
    step = int(cfg.separation.chunk_seconds * sr)
    ov = int(OVERLAP * sr)
    starts = list(range(0, n, step))
    if len(starts) > 1 and n - starts[-1] < 10 * sr:  # fold a tiny tail into the previous chunk
        starts.pop()
    vocals = np.zeros_like(y)
    inst = np.zeros_like(y)

    for i, s in enumerate(starts):
        a = max(0, s - ov)
        b = n if i == len(starts) - 1 else min(n, s + step + ov)
        chunk_path = work / f"chunk{i:04d}.wav"
        audio.save(chunk_path, y[a:b], sr)
        outs = [Path(f) if Path(f).is_absolute() else work / f for f in sep.separate(str(chunk_path))]
        v_path, i_path = _stems(outs)
        v, _ = audio.load(v_path, sr=sr)
        m, _ = audio.load(i_path, sr=sr)
        v, m = _fit(v, b - a), _fit(m, b - a)

        w = np.ones(b - a, dtype=np.float32)  # linear crossfade over the shared 2*OVERLAP region
        if a > 0:
            r = min(2 * ov, b - a)
            w[:r] = np.linspace(0, 1, r, dtype=np.float32)
        if b < n:
            r = min(2 * ov, b - a)
            w[-r:] = np.minimum(w[-r:], np.linspace(1, 0, r, dtype=np.float32))
        vocals[a:b] += v * w[:, None]
        inst[a:b] += m * w[:, None]
        for f in [chunk_path, *outs]:
            f.unlink(missing_ok=True)
        done_s = min(n, s + step) / sr
        job.progress((i + 1) / len(starts), f"separated {done_s / 60:.1f} of {n / sr / 60:.1f} min")

    del sep
    empty_torch_cache()
    shutil.rmtree(work, ignore_errors=True)
    audio.save(job.path("vocals.wav"), vocals, sr)
    audio.save(job.path("instrumental.wav"), inst, sr)

    # 16 kHz mono copy for ASR and speaker embeddings
    v16, _ = audio.load(job.path("vocals.wav"), sr=16000, mono=True)
    audio.save(job.path("vocals_16k.wav"), v16, 16000)
    job.log(f"separated vocals / instrumental in {len(starts)} chunks")
