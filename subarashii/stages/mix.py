"""Rebuild the soundtrack: music/SFX + English lines + (optionally) original non-dialogue vocals."""

import numpy as np

from .. import audio, overlap
from ..config import Cfg
from ..project import Job


def _ramp_mask(n: int, sr: int, regions: list[tuple[float, float]], ramp: float = 0.03) -> np.ndarray:
    """1.0 everywhere except inside regions (0.0), with short linear ramps at the edges."""
    mask = np.ones(n, dtype=np.float32)
    r = max(1, int(ramp * sr))
    for a, b in regions:
        ia, ib = max(0, int(a * sr)), min(n, int(b * sr))
        if ib <= ia:
            continue
        mask[ia:ib] = 0.0
        lo = max(0, ia - r)
        mask[lo:ia] = np.minimum(mask[lo:ia], np.linspace(1, 0, ia - lo, dtype=np.float32))
        hi = min(n, ib + r)
        mask[ib:hi] = np.minimum(mask[ib:hi], np.linspace(0, 1, hi - ib, dtype=np.float32))
    return mask


def run(job: Job, cfg: Cfg):
    m = cfg.mix
    sr = m.sample_rate
    inst, _ = audio.load(job.path("instrumental.wav"), sr=sr)
    voc, _ = audio.load(job.path("vocals.wav"), sr=sr)
    n = min(len(inst), len(voc))
    inst, voc = inst[:n], voc[:n]
    dub = np.zeros(n, dtype=np.float32)

    dubbed, regions = 0, []
    for line in job.p.lines:
        if line.skip or line.lyric or not line.tts or not job.path(line.tts.file).is_file():
            continue
        y, _ = audio.load(job.path(line.tts.file), sr=sr, mono=True)
        y = y * audio.db_to_amp(line.tts.gain_db + m.dub_gain_db)
        a = int(line.start * sr)
        b = min(n, a + len(y))
        if b > a:
            dub[a:b] += y[: b - a]
        regions.append((line.start - m.vocal_pad, max(line.end, line.start + len(y) / sr) + m.vocal_pad))
        dubbed += 1

    if m.keep_nonspeech_vocals:
        vmask = _ramp_mask(n, sr, regions)
    else:
        # only keep original vocals where we deliberately didn't dub (skipped lines, song lyrics)
        keep = [(l.start - m.vocal_pad, l.end + m.vocal_pad) for l in job.p.lines if l.skip or l.lyric]
        vmask = 1.0 - _ramp_mask(n, sr, keep)

    # a line kept in the original language but spoken over a dubbed one: the original vocals are
    # muted under the dub, so put its own separated voice back there
    silence = np.zeros(n, dtype=np.float32)
    for line in job.p.lines:
        if not (line.skip or line.lyric) or not line.sep:
            continue
        v = overlap.clip(job, silence, sr, line.start, line.end, line.sep)
        a = int(round(max(0.0, line.start) * sr))
        v = v[: n - a]
        dub[a : a + len(v)] += v * (1.0 - vmask[a : a + len(v)])
    out = inst + voc * vmask[:, None] + dub[:, None]

    audio.save(job.path("dub_vocals.wav"), dub, sr)
    audio.save(job.path("mix.wav"), out.astype(np.float32), sr)
    job.log(f"mixed {dubbed} dubbed lines; peak {float(np.abs(out).max(initial=0)):.2f}")
