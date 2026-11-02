"""Probe the video and extract its original-language audio track."""

from pathlib import Path

from .. import audio
from ..config import Cfg
from ..lang import LANGS
from ..project import Job


def run(job: Job, cfg: Cfg):
    src = Path(job.p.source)
    if not src.is_file():
        raise FileNotFoundError(f"source video not found: {src}")
    info = audio.probe(src)
    streams = info.get("streams", [])
    a_streams = [s for s in streams if s.get("codec_type") == "audio"]
    if not a_streams:
        raise RuntimeError("video has no audio stream")
    want = cfg.input.language
    codes = [want] if want in LANGS else list(LANGS)
    tag_of = lambda s: ((s.get("tags") or {}).get("language") or "").lower()  # noqa: E731
    lang_of = {t: c for c in codes for t in getattr(cfg.languages, c).audio_tags}
    chosen = next((s for s in a_streams if tag_of(s) in lang_of), a_streams[0])
    job.p.language = ""
    if want in LANGS:
        job.p.language = want
    elif tag_of(chosen) in lang_of:
        job.p.language = lang_of[tag_of(chosen)]  # otherwise detected from the speech in transcribe
    a_index = a_streams.index(chosen)

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    job.p.media = {
        "duration": float(info["format"].get("duration") or chosen.get("duration") or 0),
        "audio_stream": a_index,
        "audio_streams": len(a_streams),
        "audio_language": (chosen.get("tags") or {}).get("language", ""),
        "has_video": video is not None,
        "video_codec": (video or {}).get("codec_name", ""),
        "subtitle_streams": sum(1 for s in streams if s.get("codec_type") == "subtitle"),
        "container": src.suffix.lower().lstrip("."),
    }
    sr = cfg.mix.sample_rate
    audio.run_ffmpeg(["-i", str(src), "-map", f"0:a:{a_index}", "-vn", "-ac", "2", "-ar", str(sr), str(job.path("audio.wav"))])
    job.log(f"extracted audio stream {a_index} ({job.p.media['audio_language'] or 'untagged'}"
            f"{', language ' + LANGS[job.p.language].name if job.p.language else ''}), "
            f"{job.p.media['duration']:.1f}s")
