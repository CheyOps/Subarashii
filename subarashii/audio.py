"""Audio helpers: ffmpeg wrappers, wav io, trimming, loudness, time-stretch."""

import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf


def run_ffmpeg(args: list[str]):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{res.stderr[-2000:]}")


def probe(path: Path) -> dict:
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {res.stderr}")
    return json.loads(res.stdout)


def load(path: Path, sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Returns float32 audio shaped (n,) if mono else (n, channels)."""
    y, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        y = y.mean(axis=1, keepdims=True)
    if sr and sr != file_sr:
        import librosa

        y = librosa.resample(y.T, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq").T
        file_sr = sr
    if mono:
        y = y[:, 0]
    return np.ascontiguousarray(y, dtype=np.float32), file_sr


def save(path: Path, y: np.ndarray, sr: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr, subtype="FLOAT" if np.abs(y).max(initial=0) > 1.0 else "PCM_16")


def segment(y: np.ndarray, sr: int, start: float, end: float) -> np.ndarray:
    a = max(0, int(round(start * sr)))
    b = min(len(y), int(round(end * sr)))
    return y[a:b]


def rms_db(y: np.ndarray) -> float:
    if y.size == 0:
        return -120.0
    r = float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))
    return 20 * np.log10(max(r, 1e-6))


def active_rms_db(y: np.ndarray, sr: int, frame: float = 0.05, top: float = 0.5) -> float:
    """Loudness of the louder half of frames, so pauses inside a line don't drag it down."""
    n = max(1, int(frame * sr))
    if y.size < n:
        return rms_db(y)
    frames = y[: len(y) // n * n].reshape(-1, n)
    e = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    e = np.sort(e)[::-1][: max(1, int(len(e) * top))]
    return 20 * np.log10(max(float(np.mean(e)), 1e-6))


def trim(y: np.ndarray, sr: int, top_db: float = 40.0, pad: float = 0.03) -> np.ndarray:
    import librosa

    if y.size == 0:
        return y
    _, (a, b) = librosa.effects.trim(y, top_db=top_db, frame_length=1024, hop_length=256)
    p = int(pad * sr)
    return y[max(0, a - p) : min(len(y), b + p)]


def fade(y: np.ndarray, sr: int, fin: float = 0.01, fout: float = 0.02) -> np.ndarray:
    y = y.copy()
    a, b = min(len(y), int(fin * sr)), min(len(y), int(fout * sr))
    if a:
        y[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    if b:
        y[-b:] *= np.linspace(1, 0, b, dtype=np.float32)
    return y


def time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """rate > 1 speeds up (shorter). Pitch preserved (rubberband)."""
    if abs(rate - 1.0) < 1e-3 or y.size == 0:
        return y
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "in.wav", Path(tmp) / "out.wav"
        sf.write(str(src), y, sr, subtype="FLOAT")
        res = subprocess.run(
            ["rubberband", "-q", "--fine", "--formant", "--tempo", f"{rate:.5f}", str(src), str(dst)],
            capture_output=True,
            text=True,
        )
        if res.returncode != 0:
            raise RuntimeError(f"rubberband failed: {res.stderr[-1000:]}")
        out, _ = sf.read(str(dst), dtype="float32")
    return out


def db_to_amp(db: float) -> float:
    return float(10 ** (db / 20))
