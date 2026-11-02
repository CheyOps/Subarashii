"""Find utterances and transcribe them.

1. Silero VAD on the vocal stem cuts speech at short pauses.
2. Utterances long enough to hold two people are checked for a voice change (ECAPA embeddings of
   adjacent 1 s windows) and split there; turn-taking often has almost no pause.
3. Crosstalk (two people at once) is split into one line per voice (see overlap.py).
4. Each utterance is transcribed by two independent recognizers, faster-whisper large-v3 and
   a Whisper fine-tuned for the source language. They fail on different lines; the proofread
   stage reconciles the two.
With input.language auto and no usable track tag, the language is detected from the speech first.
"""

import re

import numpy as np

from .. import audio, overlap
from ..config import Cfg, models_dir
from ..gpu import empty_torch_cache
from ..lang import LANGS, job_lang, lang_cfg
from ..project import Job, Line

SR = 16000
_HAS_TEXT = re.compile(r"[\w぀-ヿ一-鿿]")


def _usable(text: str, dur: float, phrases) -> bool:
    if not text or not _HAS_TEXT.search(text) or any(p in text for p in phrases):
        return False
    return len(text) <= 14 * dur + 8  # runaway/hallucinated repetition


def vad_segments(y: np.ndarray, c) -> list[tuple[float, float]]:
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(
        threshold=c.vad_threshold,
        min_speech_duration_ms=150,
        min_silence_duration_ms=c.vad_min_silence_ms,
        speech_pad_ms=c.vad_pad_ms,
        max_speech_duration_s=c.max_line_seconds,
    )
    segs = [(s["start"] / SR, s["end"] / SR) for s in get_speech_timestamps(y, opts, sampling_rate=SR)]
    # glue crumbs (< 0.3 s) onto a neighbour that is very close
    out: list[list[float]] = []
    for a, b in segs:
        if out and (b - a < 0.3 or out[-1][1] - out[-1][0] < 0.3) and a - out[-1][1] < 0.15:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def _low_energy_point(y: np.ndarray, a: float, b: float) -> float:
    seg = audio.segment(y, SR, a, b)
    n = int(0.02 * SR)
    if seg.size < 2 * n:
        return (a + b) / 2
    frames = seg[: len(seg) // n * n].reshape(-1, n)
    return a + (int(np.argmin((frames**2).mean(axis=1))) + 0.5) * n / SR


def split_speaker_changes(y: np.ndarray, segs, enc, threshold: float, win=1.0, hop=0.25, min_side=0.7):
    import torch

    k = int(round(win / hop))

    def split(a: float, b: float) -> list[tuple[float, float]]:
        if b - a < 2 * min_side + 0.3:
            return [(a, b)]
        starts = np.arange(a, b - win + 1e-6, hop)
        if len(starts) <= k:
            return [(a, b)]
        wins = np.stack([audio.segment(y, SR, s, s + win) for s in starts if len(audio.segment(y, SR, s, s + win)) == int(win * SR)])
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(wins).cuda()).squeeze(1).float().cpu().numpy()
        e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
        best_d, best_t = 0.0, None
        for i in range(len(e) - k):
            t = starts[i] + win
            if t - a < min_side or b - t < min_side:
                continue
            d = 1.0 - float(e[i] @ e[i + k])
            if d > best_d:
                best_d, best_t = d, t
        if best_t is None or best_d < threshold:
            return [(a, b)]
        t = _low_energy_point(y, best_t - 0.3, best_t + 0.3)
        return split(a, t) + split(t, b)

    out = []
    for a, b in segs:
        out += split(a, b)
    return out


def run(job: Job, cfg: Cfg):
    c = cfg.asr
    y, _ = audio.load(job.path("vocals_16k.wav"), mono=True)

    job.progress(0.02, "voice activity detection", force=True)
    segs = vad_segments(y, c)

    crosstalk = cfg.overlap.enabled and segs
    if (c.speaker_change_threshold or crosstalk) and segs:
        from speechbrain.inference.speaker import EncoderClassifier

        enc = EncoderClassifier.from_hparams(
            source=cfg.diarization.embedding_model,
            savedir=str(models_dir(cfg) / "speechbrain" / "ecapa"),
            run_opts={"device": "cuda"},
        )
        if c.speaker_change_threshold:
            job.progress(0.08, "checking for speaker changes inside utterances", force=True)
            before = len(segs)
            segs = split_speaker_changes(y, segs, enc, c.speaker_change_threshold)
            if len(segs) > before:
                job.log(f"split {len(segs) - before} utterances at a change of voice")
        if not job.p.language:
            detect_language(job, cfg, y, [(a, b) for a, b in segs if b - a >= c.min_line_seconds])
        spans = [(a, b, []) for a, b in segs]
        if crosstalk:
            job.progress(0.1, "looking for crosstalk", force=True)
            spans = overlap.split_crosstalk(job, cfg, y, spans, enc, lambda x: vad_segments(x, c),
                                            lambda clips: _check_asr(job, cfg, clips), 0.1, 0.2)
        del enc
        empty_torch_cache()
    else:
        spans = [(a, b, []) for a, b in segs]

    spans = [s for s in spans if s[1] - s[0] >= c.min_line_seconds]
    if not job.p.language:
        detect_language(job, cfg, y, [(a, b) for a, b, _ in spans])
    texts = transcribe_spans(job, cfg, y, [(a, b) for a, b, _ in spans], seps=[sep for _, _, sep in spans])
    phrases = lang_cfg(cfg, job_lang(job, cfg)).hallucination_phrases

    lines: list[Line] = []
    for (a, b, sep), (text, alt) in zip(spans, texts):
        if _usable(text, b - a, phrases):
            lines.append(Line(id=len(lines) + 1, start=round(a, 3), end=round(b, 3), ja=text, ja_asr=text, ja_alt=alt,
                              sep=sep))
    job.p.lines = lines
    job.log(f"transcribed {len(lines)} lines from {len(spans)} utterances")


def detect_language(job: Job, cfg: Cfg, y: np.ndarray, segs, sample_seconds=60.0):
    """Whisper language ID on speech sampled from across the video (an opening song or a single
    English line shouldn't decide it)."""
    from faster_whisper import WhisperModel

    job.progress(0.09, "detecting the spoken language", force=True)
    picks = sorted(segs, key=lambda s: s[0])[:: max(1, len(segs) // 40)]
    clips, total = [], 0.0
    for a, b in picks:
        if total >= sample_seconds:
            break
        clips.append(audio.segment(y, SR, a, b))
        total += b - a
    if not clips:
        raise RuntimeError("no speech found to detect the language from")
    model = WhisperModel(cfg.asr.fallback_model, device="cuda", compute_type=cfg.asr.compute_type,
                         download_root=str(models_dir(cfg) / "whisper"))
    _, _, probs = model.detect_language(np.concatenate(clips), language_detection_segments=4)
    del model
    empty_torch_cache()
    probs = dict(probs)
    top = max(probs, key=probs.get)
    best = max(LANGS, key=lambda c: probs.get(c, 0.0) + (probs.get("yue", 0.0) if c == "zh" else 0.0))
    p = probs.get(best, 0.0) + (probs.get("yue", 0.0) if best == "zh" else 0.0)
    if p < 0.3:
        raise RuntimeError(f"the dialogue doesn't sound like Japanese, Chinese or Korean (detected '{top}'); "
                           f"set the language explicitly if it is one of those")
    job.p.language = best
    job.log(f"detected spoken language: {LANGS[best].name} ({p:.0%})")


def _check_asr(job: Job, cfg: Cfg, clips: list[np.ndarray]) -> list[str]:
    """What each line a crosstalk split would make says ("" when neither recognizer heard words)."""
    return [best for best, _ in transcribe_clips(job, cfg, clips, 0.16, 0.2)] if clips else []


def transcribe_spans(job: Job, cfg: Cfg, y: np.ndarray, spans: list[tuple[float, float]], lo=0.2, hi=0.9,
                     seps=None) -> list[tuple[str, str]]:
    """Returns (best, alternative) per span. seps: per span, the separated voice to hear in place of
    crosstalk (overlap.clip)."""
    clips = [overlap.clip(job, y, SR, a, b, sep) for (a, b), sep in zip(spans, seps or [[]] * len(spans))]
    return transcribe_clips(job, cfg, clips, lo, hi)


def transcribe_clips(job: Job, cfg: Cfg, clips: list[np.ndarray], lo=0.2, hi=0.9) -> list[tuple[str, str]]:
    """Returns (best, alternative) per clip: large-v3 first when usable, the fine-tuned model otherwise."""
    c = cfg.asr
    lang = job_lang(job, cfg)
    lc = lang_cfg(cfg, lang)
    mid = (lo + hi) / 2
    alt = (_finetuned_whisper(job, cfg, lc.asr_model, lc.get("asr_generation_config", ""), lang.name, clips, lo, mid)
           if lc.asr_model else [""] * len(clips))
    v3 = _faster_whisper(job, cfg, lang.code, clips, mid, hi) if c.fallback_model else [""] * len(clips)
    out = []
    for x, t1, t2 in zip(clips, v3, alt):
        dur = len(x) / SR
        ok1, ok2 = _usable(t1, dur, lc.hallucination_phrases), _usable(t2, dur, lc.hallucination_phrases)
        best, alt = (t1, t2 if ok2 else "") if ok1 else (t2 if ok2 else "", "")
        out.append((best, alt if alt != best else ""))
    return out


def _finetuned_whisper(job: Job, cfg: Cfg, model: str, gen_config: str, language: str, clips: list[np.ndarray],
                       lo: float, hi: float) -> list[str]:
    import torch
    from transformers import pipeline

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        device="cuda",
        torch_dtype=torch.float16,
        model_kwargs={"cache_dir": str(models_dir(cfg) / "hf")},
    )
    if gen_config:  # some fine-tunes ship without the language tables generate() needs
        from transformers import GenerationConfig

        gc = GenerationConfig.from_pretrained(gen_config, cache_dir=str(models_dir(cfg) / "hf"))
        pipe.model.generation_config = gc
        pipe.generation_config = gc  # the pipeline keeps its own copy
    # cap the output: a fine-tune that loops on one clip would otherwise grow the whole batch's
    # KV cache until the GPU runs out (max_line_seconds of speech never needs more)
    kwargs = {"language": language.lower(), "no_repeat_ngram_size": 0, "repetition_penalty": 1.0,
              "max_new_tokens": 160}
    out, batch = [], 8
    for i in range(0, len(clips), batch):
        inputs = [{"raw": np.ascontiguousarray(x), "sampling_rate": SR} for x in clips[i : i + batch]]
        out += [(r.get("text") or "").strip() for r in pipe(inputs, generate_kwargs=kwargs, batch_size=batch)]
        job.progress(lo + (hi - lo) * min(1.0, (i + batch) / len(clips)), f"transcribing ({min(i + batch, len(clips))}/{len(clips)})")
    del pipe
    empty_torch_cache()
    return out


def _faster_whisper(job: Job, cfg: Cfg, code: str, clips: list[np.ndarray], lo: float, hi: float) -> list[str]:
    from faster_whisper import WhisperModel

    job.progress(lo, f"second transcription with whisper {cfg.asr.fallback_model}", force=True)
    model = WhisperModel(cfg.asr.fallback_model, device="cuda", compute_type=cfg.asr.compute_type,
                         download_root=str(models_dir(cfg) / "whisper"))
    out = []
    for n, x in enumerate(clips):
        if n % 10 == 0:
            job.progress(lo + (hi - lo) * n / len(clips), f"second transcription ({n}/{len(clips)})")
        segs, _ = model.transcribe(x, language=code, beam_size=cfg.asr.beam_size, vad_filter=False,
                                   condition_on_previous_text=False, without_timestamps=True)
        out.append("".join(s.text for s in segs).strip())
    del model
    empty_torch_cache()
    return out
