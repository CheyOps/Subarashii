"""Crosstalk: find where two characters talk at once and pull their voices apart.

1. pyannote segmentation-3.0 marks where two voices are active at the same time. With
   overlap.scan, all speech is also separated, which finds a quiet second voice pyannote misses.
2. Each overlap, with some context either side, is split into two voices by MossFormer2. The split
   is kept only when it looks like two people: both voices speak, partly at the same time but not
   in lockstep, and they say different words. Otherwise the line stays with the dominant voice.
3. Existing lines in the window keep their times and hear only their own voice; speech of the
   other voice that has no line yet becomes a new line, overlapping the first in time.
The separated audio is saved under overlap/ and clip() splices it in wherever a stage reads a
line's audio.
"""

import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import audio
from .config import Cfg, models_dir
from .gpu import empty_torch_cache
from .project import Job, SepRef

SR = 16000
BUNDLED = Path(__file__).resolve().parent.parent / "models" / "pyannote-segmentation-3.0"
HOP = 0.02  # frame size for deciding where a separated voice is active (s)


# ------------------------------------------------------------------ reading a line's audio
@lru_cache(maxsize=64)
def _voice(path: str, sr: int, mtime: float) -> np.ndarray:
    return audio.load(Path(path), sr=sr, mono=True)[0]


def clip(job: Job, y: np.ndarray, sr: int, a: float, b: float, sep: list[SepRef]) -> np.ndarray:
    """y[a:b], with this line's own separated voice in place of the crosstalk."""
    out = audio.segment(y, sr, a, b)
    if not sep:
        return out
    out = out.copy()
    a = max(0.0, a)
    for ref in sep:
        s, e = max(a, ref.start), min(b, ref.end)
        if e <= s:
            continue
        p = job.path(ref.file)
        v = audio.segment(_voice(str(p), sr, p.stat().st_mtime), sr, s - ref.start, e - ref.start)
        i = int(round((s - a) * sr))
        n = min(len(v), len(out) - i)
        if n <= 0:
            continue
        w = np.ones(n, dtype=np.float32)  # short crossfade where the splice meets the original
        r = min(int(0.02 * sr), n // 2)
        if r and s > a:
            w[:r] = np.linspace(0, 1, r, dtype=np.float32)
        if r and e < b:
            w[n - r :] = np.linspace(1, 0, r, dtype=np.float32)
        out[i : i + n] = out[i : i + n] * (1 - w) + v[:n] * w
    return out


def separated_share(line) -> float:
    """Fraction of a line that comes from separated (crosstalk) audio."""
    s = sum(max(0.0, min(line.end, r.end) - max(line.start, r.start)) for r in line.sep)
    return s / max(line.duration, 1e-3)


# ------------------------------------------------------------------ detection and separation
def detect(y: np.ndarray, c) -> list[tuple[float, float]]:
    """Time ranges where two or more voices are active."""
    import torch
    from pyannote.audio import Model
    from pyannote.audio.core.task import Problem, Resolution, Specifications
    from pyannote.audio.pipelines import OverlappedSpeechDetection

    # the checkpoint stores these besides the weights; torch only unpickles allowlisted classes
    safe = [Problem, Resolution, Specifications, torch.torch_version.TorchVersion]
    with torch.serialization.safe_globals(safe):
        model = Model.from_pretrained(str(Path(c.segmentation_model or BUNDLED) / "pytorch_model.bin"))
    pipe = OverlappedSpeechDetection(segmentation=model)
    pipe.onset = pipe.offset = c.detect_threshold  # fixed at 0.5 for this model otherwise
    pipe.instantiate({"min_duration_on": c.min_overlap, "min_duration_off": 0.1})
    pipe.to(torch.device("cuda"))
    ann = pipe({"waveform": torch.from_numpy(y).unsqueeze(0), "sample_rate": SR})
    out = [(float(s.start), float(s.end)) for s in ann.get_timeline().support()]
    del pipe, model
    empty_torch_cache()
    return out


def _windows(regions, total: float, context: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for a, b in sorted(regions):
        a, b = max(0.0, a - context), min(total, b + context)
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


class _Separator:
    """MossFormer2 (ClearerVoice-Studio, vendored in vendor/mossformer2_ss)."""

    def __init__(self, cfg: Cfg):
        import torch
        from argparse import Namespace

        from huggingface_hub import hf_hub_download

        from .vendor.mossformer2_ss.mossformer2 import MossFormer2_SS_16K

        path = hf_hub_download(cfg.overlap.separation_model, "last_best_checkpoint.pt", cache_dir=str(models_dir(cfg) / "hf"))
        net = MossFormer2_SS_16K(Namespace(encoder_embedding_dim=512, mossformer_sequence_dim=512,
                                           num_mossformer_layer=24, encoder_kernel_size=16, num_spks=2))
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("model", state)
        net.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.net = net.cuda().eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """(2, n): the two voices, scaled so they add back up to x."""
        import torch

        with torch.no_grad():
            out = self.net(torch.from_numpy(x).unsqueeze(0).cuda())
        est = np.stack([np.pad(e[: len(x)], (0, max(0, len(x) - len(e)))) for e in (o[0].float().cpu().numpy() for o in out)])
        g, *_ = np.linalg.lstsq(est.T.astype(np.float64), x.astype(np.float64), rcond=None)
        return (est * g[:, None]).astype(np.float32)


def _frame_db(x: np.ndarray, n: int) -> np.ndarray:
    f = len(x) // n
    return 10 * np.log10(np.mean(x[: f * n].reshape(f, n).astype(np.float64) ** 2, axis=1) + 1e-10)


def _runs(on: np.ndarray, gap: float, min_len: float) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(np.int8), 0]))
    for a, b in zip(edges[::2] * HOP, edges[1::2] * HOP):
        if out and a - out[-1][1] < gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b - a >= min_len]


def _active(v: np.ndarray, mix: np.ndarray, vad, c) -> list[tuple[float, float]]:
    """Where a separated voice really speaks: VAD on the voice, minus what is only leakage of
    the other one (too quiet compared with the original at that moment)."""
    n = int(HOP * SR)
    ev, em = _frame_db(v, n), _frame_db(mix, n)
    on = (ev - em >= c.min_share_db) & (ev > em.max() - 45)
    return _runs(on & _mask(vad(v), len(on)), gap=0.3, min_len=0.15)


def _embed(enc, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        e = enc.encode_batch(torch.from_numpy(x).unsqueeze(0).cuda()).squeeze().float().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def _mask(iv, n: int) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for a, b in iv:
        m[int(a / HOP) : int(np.ceil(b / HOP))] = True
    return m


def _why_not_two(act, n: int, c, min_len: float) -> str:
    """Why the separated pair is not two people talking over each other ("" when it is): both
    voices must speak, at the same time for a while, but not start and stop together (one voice
    that the separator split in two does)."""
    if any(sum(b - a for a, b in iv) < min_len for iv in act):
        return "only one voice"
    m0, m1 = _mask(act[0], n), _mask(act[1], n)
    both = (m0 & m1).sum() * HOP
    if both < c.min_overlap:
        return "voices take turns"
    if both / max((m0 | m1).sum() * HOP, HOP) > c.max_sync:
        return "voices in lockstep"
    return ""


def _bare(text: str) -> str:
    return "".join(ch for ch in text if ch.isalnum())


def _repeats(t0: str, t1: str, c) -> bool:
    """The second "voice" only repeats the first: most of the words are shared, or the shorter
    transcript sits almost wholly inside the longer one (leakage). A side the recognizer heard
    nothing on proves nothing; it just won't become a line."""
    from difflib import SequenceMatcher

    a, b = _bare(t0), _bare(t1)
    if not a or not b:
        return False
    m = sum(x.size for x in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return m / max(len(a), len(b)) >= c.same_words or m / min(len(a), len(b)) >= 0.9


def _apply(spans, wa: float, wb: float, two, voices, merge_gap: float, min_len: float, max_len: float):
    """Give the speech segments in one crosstalk window to the voices in it.

    Each segment goes to its voice (the one speaking where it leaves the window, if it runs past an
    edge; else the louder one over it) and is stretched over that voice's burst of speech; segments of one voice that a single burst runs through (the voice-change check cut
    a line around an interjection) are joined again. A burst with no segment of its own becomes a
    new line. Returns (inside, owner, times, joined, new) or None when nothing would change.
    """
    inside = [i for i, s in enumerate(spans) if s[1] > wa and s[0] < wb]
    if not inside:
        return None

    def level(k, a, b):
        return audio.rms_db(audio.segment(two[k], SR, max(a, wa) - wa, min(b, wb) - wa))

    def owner_of(a, b, edge=0.5):
        if b > wb and a >= wa:
            a, b = wb - edge, wb
        elif a < wa and b <= wb:
            a, b = wa, wa + edge
        return int(np.argmax([level(k, a, b) for k in range(2)]))

    owner = {i: owner_of(spans[i][0], spans[i][1]) for i in inside}
    times = {i: [spans[i][0], spans[i][1]] for i in inside}
    joined: dict[int, int] = {}  # segment -> the segment it was joined into
    new = []
    for k, v in enumerate(voices):
        bursts: list[list[float]] = []
        for a, b in v:
            if bursts and a - bursts[-1][1] < merge_gap:
                bursts[-1][1] = b
            else:
                bursts.append([a, b])
        mine = [i for i in inside if owner[i] == k]
        for a, b in bursts:
            hit = [i for i in mine if i not in joined and times[i][0] < b and a < times[i][1]]
            span = [min([a] + [times[i][0] for i in hit]), max([b] + [times[i][1] for i in hit])]
            if len(hit) == 1 or (hit and span[1] - span[0] <= max_len):
                times[hit[0]] = span
                joined.update({i: hit[0] for i in hit[1:]})
            elif not hit and b - a >= min_len:
                new.append((a, b, k))
    if not new and len(set(owner.values())) < 2:
        return None  # one voice owns everything: nothing to gain from the separated audio
    return inside, owner, times, joined, new


def _scan(job: Job, y: np.ndarray, spans, sep, enc, vad, c, lo: float, hi: float) -> list[tuple[float, float]]:
    """Find crosstalk by separating all speech: where both separated voices are active at once.
    Catches overlaps pyannote misses (often a quiet second voice)."""
    chunks: list[list[float]] = []
    for a, b, _ in spans:  # continuous speech, in pieces the separator handles in one pass
        if chunks and a - chunks[-1][1] < 0.3 and b - chunks[-1][0] <= c.max_window:
            chunks[-1][1] = max(chunks[-1][1], b)
        else:
            chunks.append([a, b])
    total = len(y) / SR
    out = []
    for n, (a, b) in enumerate(chunks):
        if n % 10 == 0:
            job.progress(lo + (hi - lo) * n / len(chunks), f"listening for crosstalk ({n}/{len(chunks)})")
        a, b = max(0.0, a - 0.3), min(total, b + 0.3)
        mix = audio.segment(y, SR, a, b)
        if len(mix) < SR:
            continue
        two = sep(mix)
        act = [_active(v, mix, vad, c) for v in two]
        if any(not iv for iv in act):
            continue
        m = _mask(act[0], len(mix) // int(HOP * SR) + 1) & _mask(act[1], len(mix) // int(HOP * SR) + 1)
        regions = [(a + x, a + z) for x, z in _runs(m, gap=0.3, min_len=c.min_overlap)]
        if not regions:
            continue
        e = [_embed(enc, np.concatenate([audio.segment(v, SR, x, z) for x, z in iv])) for v, iv in zip(two, act)]
        if float(e[0] @ e[1]) <= c.same_voice:
            out += regions
    return out


def split_crosstalk(job: Job, cfg: Cfg, y: np.ndarray, spans, enc, vad, asr, lo: float, hi: float):
    """spans: (start, end, sep) speech segments in time order. Returns them with crosstalk resolved
    into one span per voice (which may overlap each other). asr(clips) -> texts transcribes the
    lines a split would make; it is dropped when two of them say the same words (one voice heard
    twice) or the new lines hold no words."""
    from collections import Counter

    c = cfg.overlap
    regions = detect(y, c)
    found = f"{len(regions)} overlaps found by pyannote"
    out_dir = job.path("overlap")
    shutil.rmtree(out_dir, ignore_errors=True)
    sep = _Separator(cfg)
    mid = lo + (hi - lo) * (0.6 if c.scan else 0.0)
    if c.scan:
        scanned = _scan(job, y, spans, sep, enc, vad, c, lo, mid)
        found += f", {len(scanned)} by separating all speech"
        regions += scanned
    merged = _windows(regions, len(y) / SR, c.context)
    windows = [(a, b) for a, b in merged if b - a <= c.max_window]
    min_len = cfg.asr.min_line_seconds
    why = Counter({"too long to separate": len(merged) - len(windows)})
    cands = []
    total, done_to = len(y) / SR, 0.0
    for w, (wa, wb) in enumerate(windows):
        job.progress(mid + (hi - mid) * 0.8 * w / len(windows), f"separating crosstalk ({w + 1}/{len(windows)})")
        # both voices still talking at an edge: the crosstalk goes on, so take in more of it
        # (up to max_window, never into a neighbouring window)
        lo_lim, hi_lim = done_to, windows[w + 1][0] if w + 1 < len(windows) else total
        while True:
            mix = audio.segment(y, SR, wa, wb)
            two = sep(mix)
            act = [_active(v, mix, vad, c) for v in two]
            left = wa - c.context >= lo_lim and all(iv and iv[0][0] <= 0.15 for iv in act)
            right = wb + c.context <= hi_lim and all(iv and iv[-1][1] >= wb - wa - 0.15 for iv in act)
            if not (left or right) or wb - wa + c.context > c.max_window:
                break
            wa, wb = wa - c.context * left, wb + c.context * right
        done_to = wb
        reason = _why_not_two(act, len(mix) // int(HOP * SR) + 1, c, min_len)
        if reason in ("", "voices in lockstep"):
            e = [_embed(enc, np.concatenate([audio.segment(v, SR, a, b) for a, b in iv])) for v, iv in zip(two, act)]
            similar = float(e[0] @ e[1])
            if similar > c.same_voice:
                reason = "same voice"
            elif similar < c.distinct_voice:
                reason = ""  # clearly two people, even if the split made them look in step
        if reason:
            why[reason] += 1
            continue
        cands.append((wa, wb, two, act))
    del sep
    empty_torch_cache()

    # what each split would make: the lines in its window, each heard through its own voice
    out_dir.mkdir()
    props = []
    for n, (wa, wb, two, act) in enumerate(cands):
        voices = [[(wa + a, wa + b) for a, b in iv] for iv in act]
        res = _apply(spans, wa, wb, two, voices, cfg.diarization.merge_gap, min_len, cfg.asr.max_line_seconds)
        if res is None:
            why["no line to give the second voice"] += 1
            continue
        refs = []
        for k, v in enumerate(two):
            rel = f"overlap/{n:04d}_{k}.wav"
            audio.save(job.path(rel), v, SR)
            refs.append(SepRef(file=rel, start=round(wa, 3), end=round(wb, 3)))
        props.append((wa, wb, two, voices, refs, *_commit(spans, res, refs)[1:]))
    job.progress(mid + (hi - mid) * 0.8, "checking separated voices", force=True)
    lines = [s for p in props for s in p[5] + p[6]]
    heard = dict(zip(map(id, lines), asr([clip(job, y, SR, a, b, sep) for a, b, sep in lines])))

    kept, added = 0, 0
    for wa, wb, two, voices, refs, changed, new in props:
        mine = [(s, heard[id(s)]) for s in changed + new]
        if not all(any(t for s, t in mine if r in s[2]) for r in refs):
            why["no words heard"] += 1  # one of the voices would have no line with words
        elif any(s[0] < u[1] and u[0] < s[1] and _repeats(t, v, c) for i, (s, t) in enumerate(mine) for u, v in mine[i + 1:]):
            why["same words"] += 1  # two overlapping lines saying the same thing: one voice heard twice
        else:
            res = _apply(spans, wa, wb, two, voices, cfg.diarization.merge_gap, min_len, cfg.asr.max_line_seconds)
            if res is not None:
                spans, _, new = _commit(spans, res, refs)
                kept += 1
                added += len(new)
                continue
        for r in refs:
            job.path(r.file).unlink(missing_ok=True)
    rejected = ", ".join(f"{n} {r}" for r, n in why.most_common() if n)
    job.log(f"crosstalk: {found}; {kept} of {len(windows)} places split into two voices ({added} lines added)"
            + (f"; not split: {rejected}" if rejected else ""))
    return spans


def _commit(spans, res, refs):
    """spans with one window's split applied; also returns the window's changed and new spans."""
    inside, owner, times, joined, new = res
    sep_of = {i: list(spans[i][2]) for i in inside if i not in joined}
    for i, j in joined.items():
        sep_of[j] += [r for r in spans[i][2] if r not in sep_of[j]]
    changed = [(times[i][0], times[i][1], sep_of[i] + [refs[owner[i]]]) for i in sep_of]
    added = [(a, b, [refs[k]]) for a, b, k in new]
    rest = [s for i, s in enumerate(spans) if i not in owner]
    return sorted(rest + changed + added, key=lambda s: (s[0], s[1])), changed, added
