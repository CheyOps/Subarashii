"""Voice-cloned English synthesis, fitted to each line's time slot.

Per line, in order of preference:
 0. With tts.voice_candidates > 1, synthesize the natural line with that many seeds and keep the
    take closest to the original speaker's voice (VoiceMatch); the rest of the fitting uses its seed.
 1. Synthesize the chosen candidate at natural speed. Fits -> done (optionally slowed a touch
    if it's much shorter than the original).
 2. Too long -> re-synthesize the same take (same seed) with IndexTTS duration_factor down to
    timing.min_duration_factor.
 3. Still too long -> try the next shorter LLM candidate (concise, minimal), repeating 1-2.
 4. Nothing fits -> queue the line for an LLM rewrite at a measured syllable budget
    (timing.refit_rounds), then as a last resort rubberband-stretch and allow overflow.
"""

import time

import numpy as np

from .. import audio
from ..config import Cfg, models_dir
from ..gpu import TTSClient, empty_torch_cache, free_gpu
from ..llm import get_llm
from ..project import Job, Line, TTSTake
from ..timing import en_syllables, shorter_candidates
from .translate import clean_for_tts

REFIT_SYSTEM = """You shorten English dub lines so they can be spoken in the time available.
For each line, keep the meaning and the character's voice, and write a version with at most
max_syllables syllables. Natural spoken English; contractions encouraged. Return every id."""

REFIT_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["id", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


class VoiceMatch:
    """How close a take sounds to the original speaker: cosine similarity of ECAPA voice embeddings
    against the average of the speaker's own lines. Runs on the CPU: the translator LLM is loaded
    with every layer on the GPU for the refit rounds, and even a CUDA context here would starve it."""

    def __init__(self, job: Job, cfg: Cfg):
        self.job, self.cfg = job, cfg
        self.enc = None
        self.centroids: dict[str, np.ndarray] = {}

    def _embed(self, y: np.ndarray, sr: int) -> np.ndarray:
        import torch

        if self.enc is None:
            from speechbrain.inference.speaker import EncoderClassifier

            self.enc = EncoderClassifier.from_hparams(
                source=self.cfg.diarization.embedding_model,
                savedir=str(models_dir(self.cfg) / "speechbrain" / "ecapa"),
                run_opts={"device": "cpu"},
            )
        if sr != 16000:
            import librosa

            y = librosa.resample(y, orig_sr=sr, target_sr=16000, res_type="soxr_hq")
        with torch.no_grad():
            e = self.enc.encode_batch(torch.from_numpy(y).unsqueeze(0)).squeeze().float().cpu().numpy()
        return e / (np.linalg.norm(e) + 1e-9)

    def centroid(self, sid: str) -> np.ndarray:
        if sid not in self.centroids:
            lines = [l for l in self.job.p.lines if l.speaker == sid and l.ja_clip and self.job.path(l.ja_clip).is_file()]
            clean = [l for l in lines if not l.sep and l.duration >= 1.5]
            embs = []
            for l in (clean or lines)[:60]:
                y, sr = audio.load(self.job.path(l.ja_clip), mono=True)
                embs.append(self._embed(y, sr))
            c = np.mean(embs, axis=0)
            self.centroids[sid] = c / (np.linalg.norm(c) + 1e-9)
        return self.centroids[sid]

    def score(self, sid: str, y: np.ndarray, sr: int) -> float:
        return float(self._embed(y, sr) @ self.centroid(sid))

    def close(self):
        self.enc = None


class Fitter:
    def __init__(self, job: Job, cfg: Cfg):
        self.job, self.cfg = job, cfg
        self.voice = VoiceMatch(job, cfg) if int(cfg.tts.get("voice_candidates", 1)) > 1 else None
        self.picked: list[tuple[float, float]] = []  # (first seed's match, chosen take's match)
        self.t = cfg.timing
        self.tts = TTSClient(cfg)
        self.sr = cfg.mix.sample_rate
        self.rates: dict[str, float] = {}  # measured syllables/sec per speaker at factor 1.0

    def rate(self, sid: str) -> float:
        return self.rates.get(sid, self.t.tts_syllable_rate)

    def _synth(self, line: Line, text: str, factor: float, seed: int) -> tuple:
        spk = self.job.p.speakers[line.speaker]
        raw = self.job.path(f"lines/{line.id:04d}_en_raw.wav")
        emo = str(self.job.path(line.ja_clip)) if self.cfg.tts.emotion_from_original and line.ja_clip else None
        self.tts.synthesize(
            text=text,
            output=str(raw),
            spk_audio=str(self.job.path(spk.ref_audio)),
            emo_audio=emo,
            emo_alpha=self.cfg.tts.emo_alpha,
            lang="EN",
            duration_factor=factor,
            seed=seed,
        )
        y, _ = audio.load(raw, sr=self.sr, mono=True)
        y = audio.trim(y, self.sr)
        return y, len(y) / self.sr

    def _learn(self, sid: str, text: str, dur: float, factor: float):
        if dur <= 0.3:
            return
        r = en_syllables(text) / (dur / factor)
        self.rates[sid] = 0.7 * self.rate(sid) + 0.3 * r

    def _pick_take(self, line: Line, text: str, avail: float) -> tuple[int, tuple]:
        """(seed, (y, dur)) of the natural-speed take that sounds most like the original speaker. A take
        that overruns the slot loses a little, since it would have to be sped up."""
        base = int(self.cfg.tts.seed) + line.id
        best, first = None, None
        for k in range(int(self.cfg.tts.voice_candidates)):
            seed = base + 1000 * k
            y, d = self._synth(line, text, 1.0, seed)
            if k == 0 and d < 1.0:
                return seed, (y, d)  # too short for a voice comparison to mean much
            sim = self.voice.score(line.speaker, y, self.sr)
            first = sim if first is None else first
            s = sim - 0.5 * max(0.0, d / avail - 1.0)
            if best is None or s > best[0]:
                best = (s, seed, (y, d), sim)
        self.picked.append((first, best[3]))
        return best[1], best[2]

    def fit(self, line: Line, final: bool) -> bool:
        """Returns True when the line fits its slot. When final, always stores the best take."""
        target = max(line.duration, 0.2)
        avail = max(line.slot_end - line.start, target)
        seed = int(self.cfg.tts.seed) + line.id
        texts = [line.en.strip()] + ([] if line.en_edited else shorter_candidates(line, line.en))
        best = None  # (overshoot, y, dur, text, factor)
        take = None
        if self.voice:
            seed, take = self._pick_take(line, texts[0], avail)

        for n, text in enumerate(texts):
            y, d = take if take and n == 0 else self._synth(line, text, 1.0, seed)
            self._learn(line.speaker, text, d, 1.0)
            if d <= avail:
                factor, fit = 1.0, "ok"
                if d < target * self.t.slowdown_below and self.t.max_slowdown > 1.0:
                    factor = min(self.t.max_slowdown, target / d)
                    y2, d2 = self._synth(line, text, factor, seed)
                    if d2 <= avail:
                        y, d, fit = y2, d2, "slowed"
                    else:
                        factor = 1.0
                return self._store(line, y, d, text, factor, 1.0, fit, seed)

            factor = max(self.t.min_duration_factor, avail / d * 0.97)
            y2, d2 = self._synth(line, text, factor, seed)
            if d2 <= avail * 1.02:
                return self._store(line, y2, d2, text, factor, 1.0, "sped", seed)
            if best is None or d2 - avail < best[0]:
                best = (d2 - avail, y2, d2, text, factor)

        if not final:
            return False
        _, y, d, text, factor = best
        stretch = min(self.t.max_extra_stretch, d / avail)
        if stretch > 1.0:
            y = audio.time_stretch(y, self.sr, stretch)
            d = len(y) / self.sr
        fit = "stretched" if d <= avail * 1.02 else "overflow"
        return self._store(line, y, d, text, factor, stretch, fit, seed)

    def _store(self, line, y, d, text, factor, stretch, fit, seed) -> bool:
        rel = f"lines/{line.id:04d}_en.wav"
        audio.save(self.job.path(rel), audio.fade(y, self.sr, 0.005, 0.03), self.sr)
        line.tts = TTSTake(file=rel, text=text, duration=round(d, 3), duration_factor=round(factor, 3),
                           stretch=round(stretch, 3), fit=fit, seed=seed)  # gain_db is set by mix
        line.en = text  # a shorter candidate may have been used; show what is actually spoken
        line.dirty = False
        return fit not in ("overflow",)


def _needs(job: Job, line: Line) -> bool:
    if line.skip or line.lyric or not line.en.strip():
        return False
    return line.dirty or line.tts is None or line.tts.text != line.en or not job.path(line.tts.file).is_file()


def _refit(job: Job, cfg: Cfg, lines: list[Line], fitter: Fitter):
    """Ask the LLM for shorter rewrites using each speaker's measured TTS pace."""
    llm = get_llm(cfg)
    items = []
    for l in lines:
        avail = l.slot_end - l.start
        budget = max(1, int(avail * fitter.rate(l.speaker) / cfg.timing.min_duration_factor * 0.95))
        spk = job.p.speakers.get(l.speaker)
        items.append({"id": l.id, "speaker": spk.label if spk else l.speaker, "ja": l.ja,
                      "current_en": l.en, "max_syllables": budget})
    import json

    out = llm.json(REFIT_SYSTEM, json.dumps({"lines": items}, ensure_ascii=False, indent=1), REFIT_SCHEMA, max_tokens=4000)
    llm.release()
    got = {int(i["id"]): clean_for_tts(i["text"].strip(), cfg.translation.honorifics) for i in out.get("lines", []) if i.get("text")}
    for l in lines:
        if got.get(l.id):
            l.candidates["refit"] = got[l.id]
            l.en = got[l.id]
    job.log(f"LLM shortened {len(got)} overflowing lines")


def run(job: Job, cfg: Cfg):
    free_gpu(cfg, keep="tts")
    missing = [s for s in job.p.speakers.values() if not s.ref_audio or not job.path(s.ref_audio).is_file()]
    if missing:
        raise RuntimeError(f"speakers without reference audio: {[s.id for s in missing]} (re-run profile)")
    for l in job.p.lines:
        if l.speaker not in job.p.speakers:
            raise RuntimeError(f"line {l.id} has unknown speaker {l.speaker!r}")

    todo = sorted([l for l in job.p.lines if _needs(job, l)], key=lambda l: (l.speaker, l.start))
    if not todo:
        job.log("all lines already synthesized")
        return
    job.progress(0.0, "loading the voice model", force=True)
    fitter = Fitter(job, cfg)
    rounds = int(cfg.timing.refit_rounds)
    t0 = time.monotonic()
    pending = todo
    for rnd in range(rounds + 1):
        final = rnd == rounds
        failed = []
        for i, line in enumerate(pending):
            job.progress(i / len(pending), f"synthesizing line {line.id} ({i + 1}/{len(pending)})"
                         + (f" - refit round {rnd}" if rnd else ""))
            refitable = not final and not line.en_edited
            if not fitter.fit(line, final=final or not refitable):
                failed.append(line)
            if i % 10 == 0:
                job.save()
        job.save()
        if final or not [l for l in failed if not l.en_edited]:
            break
        fitter.tts.unload()
        empty_torch_cache()
        _refit(job, cfg, [l for l in failed if not l.en_edited], fitter)
        pending = [l for l in failed if not l.en_edited]

    fitter.tts.unload()
    if fitter.voice:
        fitter.voice.close()
    fits = {}
    for l in todo:
        if l.tts:
            fits[l.tts.fit] = fits.get(l.tts.fit, 0) + 1
    job.log(f"synthesized {len(todo)} lines in {time.monotonic() - t0:.0f}s: {fits}")
    if fitter.picked:
        first, chosen = (float(np.mean(x)) for x in zip(*fitter.picked))
        job.log(f"voice match to the original speaker: {chosen:.3f} with the best of "
                f"{cfg.tts.voice_candidates} takes per line ({first:.3f} with the first take)")
