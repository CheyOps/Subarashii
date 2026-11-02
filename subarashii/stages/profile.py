"""Per-speaker voice profile (age/gender/pitch/pace) and a clean reference clip for voice cloning.

Also cuts each line's original vocal into lines/NNNN_ja.wav: the review UI plays it, and
the TTS uses it as the emotion reference so the English take carries the same delivery.
Where the line was spoken over someone else, the clip is the separated voice. Voice analysis and
the cloning reference use lines without crosstalk whenever the speaker has any.
"""

import numpy as np

from .. import audio, overlap
from ..config import Cfg, models_dir
from ..gpu import empty_torch_cache
from ..lang import job_lang, speech_units
from ..project import Job


def _age_gender_model(cfg: Cfg):
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

    class ModelHead(nn.Module):
        def __init__(self, config, num_labels):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, num_labels)

        def forward(self, x):
            x = torch.tanh(self.dense(self.dropout(x)))
            return self.out_proj(self.dropout(x))

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = ModelHead(config, 1)
            self.gender = ModelHead(config, 3)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return self.age(hidden), torch.softmax(self.gender(hidden), dim=1)

    name, cache = cfg.profile.age_gender_model, str(models_dir(cfg) / "hf")
    proc = Wav2Vec2Processor.from_pretrained(name, cache_dir=cache)
    model = AgeGenderModel.from_pretrained(name, cache_dir=cache).cuda().eval()
    return proc, model


def _pitch_hz(y: np.ndarray, sr: int) -> float:
    import librosa

    f0, voiced, _ = librosa.pyin(y, fmin=70, fmax=1000, sr=sr, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)] if f0 is not None else np.array([])
    return float(np.median(f0)) if f0.size else 0.0


def run(job: Job, cfg: Cfg):
    import torch

    c = cfg.profile
    code = job_lang(job, cfg).code
    sr = cfg.mix.sample_rate
    voc, _ = audio.load(job.path("vocals.wav"), mono=True)
    inst, _ = audio.load(job.path("instrumental.wav"), mono=True)
    v16, _ = audio.load(job.path("vocals_16k.wav"), mono=True)

    # original line clips
    (job.root / "lines").mkdir(exist_ok=True)
    for n, line in enumerate(job.p.lines):
        if n % 25 == 0:
            job.progress(0.3 * n / max(1, len(job.p.lines)), "cutting original line clips")
        clip = audio.fade(overlap.clip(job, voc, sr, line.start - 0.05, line.end + 0.08, line.sep), sr)
        rel = f"lines/{line.id:04d}_ja.wav"
        audio.save(job.path(rel), clip, sr)
        line.ja_clip = rel

    proc, model = _age_gender_model(cfg)
    by_spk: dict[str, list] = {}
    for line in job.p.lines:
        by_spk.setdefault(line.speaker, []).append(line)

    for n, (sid, lines) in enumerate(by_spk.items()):
        spk = job.p.speakers.get(sid)
        if spk is None:
            continue
        job.progress(0.3 + 0.7 * n / max(1, len(by_spk)), f"profiling {sid}")
        spk.line_count = len(lines)
        spk.total_seconds = round(sum(l.duration for l in lines), 2)

        clean = [l for l in lines if not l.sep] or lines
        longest = sorted(clean, key=lambda l: -l.duration)[:8]
        ages, genders = [], []
        with torch.no_grad():
            for l in longest:
                seg = overlap.clip(job, v16, 16000, l.start, min(l.end, l.start + 8), l.sep)
                if seg.size < 16000 * 0.5:
                    continue
                x = proc(seg, sampling_rate=16000, return_tensors="pt").input_values.cuda()
                age, gender = model(x)
                ages.append(float(age[0, 0]) * 100)
                genders.append(gender[0].cpu().numpy())
        g = np.mean(genders, axis=0) if genders else np.array([0.0, 0.0, 0.0])
        pitch_src = np.concatenate([overlap.clip(job, voc, sr, l.start, l.end, l.sep) for l in longest[:4]]) if longest else voc[:0]
        units = sum(speech_units(l.ja, code) for l in lines)
        spk.acoustic = {
            "female": round(float(g[0]), 2),
            "male": round(float(g[1]), 2),
            "child": round(float(g[2]), 2),
            "age_estimate": round(float(np.mean(ages)), 1) if ages else None,
            "pitch_hz": round(_pitch_hz(pitch_src[: sr * 20], sr), 1) if pitch_src.size else 0.0,
            "units_per_sec": round(units / max(spk.total_seconds, 0.1), 2),  # morae or syllables
            "loudness_db": round(float(np.mean([audio.active_rms_db(overlap.clip(job, voc, sr, l.start, l.end, l.sep), sr) for l in longest])), 1) if longest else None,
        }
        _build_reference(job, cfg, spk, lines, voc, inst, sr)
        spk.thumbs = _thumbnails(job, spk, lines)

    del model, proc
    empty_torch_cache()
    job.log(f"profiled {len(by_spk)} speakers")


def _thumbnails(job, spk, lines, count: int = 4) -> list[str]:
    """A few frames from moments this character speaks, spread through the video, for the UI."""
    if not job.p.media.get("has_video"):
        return []
    longest = sorted(lines, key=lambda l: -l.duration)[: count * 3]
    longest.sort(key=lambda l: l.start)
    picks = longest[:: max(1, len(longest) // count)][:count]
    out = []
    for k, l in enumerate(picks):
        rel = f"speakers/{spk.id}_thumb{k}.jpg"
        try:
            audio.run_ffmpeg(["-ss", f"{(l.start + l.end) / 2:.2f}", "-i", job.p.source, "-frames:v", "1",
                              "-vf", "scale=320:-2", "-q:v", "4", str(job.path(rel))])
            out.append(rel)
        except RuntimeError:
            pass
    return out


def _build_reference(job, cfg, spk, lines, voc, inst, sr):
    """Pick the cleanest, most typical clips of this speaker and join them into one reference wav."""
    c = cfg.profile
    scored = []
    for l in lines:
        d = min(l.duration, c.ref_max_clip)
        if d < c.ref_min_clip:
            continue
        v = overlap.clip(job, voc, sr, l.start, l.start + d, l.sep)
        bg = audio.segment(inst, sr, l.start, l.start + d)
        snr = audio.rms_db(v) - audio.rms_db(bg)
        snr_f = float(np.clip((snr + 5) / 25, 0.1, 1.0))  # music bleed hurts cloning most
        dur_f = min(d, 6.0) / 6.0
        peak = float(np.abs(v).max(initial=0))
        clip_f = 0.5 if peak > 0.98 else 1.0  # avoid clipped screams as timbre reference
        sep_f = 0.3 if l.sep else 1.0  # separated crosstalk carries artifacts into the clone
        scored.append((max(l.speaker_conf, 0.05) * snr_f * dur_f * clip_f * sep_f, l, d))
    if not scored:  # fall back to whatever this speaker has
        scored = [(l.duration, l, min(l.duration, c.ref_max_clip)) for l in lines]
    scored.sort(key=lambda t: -t[0])

    parts, total, used = [], 0.0, []
    gap = np.zeros(int(0.25 * sr), dtype=np.float32)
    for _, l, d in scored:
        if total >= c.ref_target_seconds:
            break
        parts += [audio.fade(overlap.clip(job, voc, sr, l.start, l.start + d, l.sep), sr), gap]
        total += d
        used.append(l.id)
    ref = np.concatenate(parts) if parts else np.zeros(sr, dtype=np.float32)
    peak = float(np.abs(ref).max(initial=0))
    if peak > 0:
        ref = ref * (0.9 / peak)
    rel = f"speakers/{spk.id}_ref.wav"
    audio.save(job.path(rel), ref, sr)
    spk.ref_audio, spk.ref_clips = rel, used
