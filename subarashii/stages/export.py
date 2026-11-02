"""Mux the dubbed audio (plus original audio and English subtitles) back into the video."""

from pathlib import Path

from .. import audio
from ..config import Cfg, data_dir
from ..lang import job_lang
from ..project import Job


def _ts(t: float) -> str:
    ms = int(round(max(t, 0) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(job: Job, path: Path):
    rows, n = [], 0
    for l in job.p.lines:
        text = (l.tts.text if l.tts else l.en).strip()
        if l.skip or not text:
            continue
        n += 1
        end = max(l.end, l.start + (l.tts.duration if l.tts else 0))
        rows.append(f"{n}\n{_ts(l.start)} --> {_ts(end)}\n{text}\n")
    path.write_text("\n".join(rows), encoding="utf-8")


def run(job: Job, cfg: Cfg):
    o = cfg.output
    src = Path(job.p.source)
    container = job.p.media.get("container", "")
    ext = "mp4" if container in ("mp4", "m4v", "mov") else "mkv"
    out = Path(job.p.output) if job.p.output else data_dir(cfg, "output") / f"{src.stem}.en-dub.{ext}"
    n = 2
    while not job.p.output and out.exists():  # another job of the same video already wrote this name
        out = out.with_name(f"{src.stem}.en-dub-{n}.{ext}")
        n += 1
    srt = job.path("dub.en.srt")
    write_srt(job, srt)

    args = ["-i", str(src), "-i", str(job.path("mix.wav"))]
    if o.subtitles:
        args += ["-i", str(srt)]
    args += ["-map", "0:v?", "-map", "1:a"]
    if o.keep_original_audio:
        args += ["-map", f"0:a:{job.p.media.get('audio_stream', 0)}"]
    if o.subtitles:
        args += ["-map", "2:s"]
    if ext == "mkv":
        args += ["-map", "0:s?", "-map", "0:t?"]  # keep existing subtitle tracks and fonts
    args += [
        "-c:v", "copy",
        "-c:a:0", "aac", "-b:a:0", o.audio_bitrate,
        "-filter:a:0", "alimiter=limit=0.95:level=disabled",
        "-metadata:s:a:0", "language=eng",
        "-metadata:s:a:0", "title=English (AI dub)",
        "-disposition:a:0", "default",
    ]
    if o.keep_original_audio:
        lang = job_lang(job, cfg)
        args += ["-c:a:1", "copy" if ext == "mkv" else "aac",
                 "-metadata:s:a:1", f"language={lang.iso3b if ext == 'mkv' else lang.iso3}",
                 "-metadata:s:a:1", f"title={lang.name} (original)", "-disposition:a:1", "0"]
    if o.subtitles:
        args += ["-c:s", "copy", "-c:s:0", "mov_text" if ext == "mp4" else "srt",
                 "-metadata:s:s:0", "language=eng", "-metadata:s:s:0", "title=English (dub script)"]
    if ext == "mp4":
        args += ["-movflags", "+faststart"]
    args.append(str(out))
    audio.run_ffmpeg(args)
    job.p.output = str(out)
    job.log(f"wrote {out}")
