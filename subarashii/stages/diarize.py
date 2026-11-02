"""Group lines by voice: ECAPA speaker embeddings + agglomerative clustering.

Short lines (< min_cluster_line_seconds) and lines mostly made of separated crosstalk give noisy
embeddings, so they don't vote in clustering; they're attached to the nearest speaker afterwards. Misassignments are fixed
in the review UI (reassign a line, merge two speakers).
"""

import numpy as np

from .. import audio, overlap
from ..config import Cfg, models_dir
from ..gpu import empty_torch_cache
from ..project import Job, Speaker


def _embed_all(job: Job, cfg: Cfg) -> np.ndarray:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams(
        source=cfg.diarization.embedding_model,
        savedir=str(models_dir(cfg) / "speechbrain" / "ecapa"),
        run_opts={"device": "cuda"},
    )
    y, sr = audio.load(job.path("vocals_16k.wav"), mono=True)
    embs = np.zeros((len(job.p.lines), 192), dtype=np.float32)
    with torch.no_grad():
        for i, line in enumerate(job.p.lines):
            seg = overlap.clip(job, y, sr, line.start, min(line.end, line.start + 10.0), line.sep)
            if seg.size < sr * 0.3:
                seg = np.pad(seg, (0, int(sr * 0.3) - seg.size))
            e = enc.encode_batch(torch.from_numpy(seg).unsqueeze(0).cuda()).squeeze().float().cpu().numpy()
            embs[i] = e / (np.linalg.norm(e) + 1e-9)
            if i % 20 == 0:
                job.progress(0.8 * i / len(job.p.lines), "speaker embeddings")
    del enc
    empty_torch_cache()
    return embs


def cluster(embs: np.ndarray, durs: list[float], c, votes: list[bool] | None = None) -> tuple[list[int], list[float]]:
    """Assign every line a speaker cluster; returns (label per line, similarity to its centroid).

    1. Agglomerative clustering on lines long enough for a reliable voice print.
    2. A few k-means style passes reassign those lines to the nearest centroid.
    3. Clusters whose centroids are near-identical are merged (same actor, different mood).
    4. Clusters with too little speech are folded into the nearest voice.
    5. Short lines join the nearest cluster.
    """
    from sklearn.cluster import AgglomerativeClustering

    n = len(embs)
    idx = [i for i in range(n) if durs[i] >= c.min_cluster_line_seconds and (votes is None or votes[i])]
    if len(idx) < 2:
        idx = list(range(n))
    if len(idx) == 1:
        return [0] * n, [1.0] * n
    kw = {"n_clusters": c.max_speakers} if c.max_speakers else {"n_clusters": None, "distance_threshold": c.distance_threshold}
    lab = dict(zip(idx, (int(x) for x in AgglomerativeClustering(metric="cosine", linkage="average", **kw).fit_predict(embs[idx]))))

    def centroids():
        out = {}
        for k in set(lab.values()):
            m = embs[[i for i in idx if lab[i] == k]].mean(axis=0)
            out[k] = m / (np.linalg.norm(m) + 1e-9)
        return out

    for _ in range(int(c.get("refine_passes", 3))):
        cents = centroids()
        keys = list(cents)
        mat = np.stack([cents[k] for k in keys])
        new = {i: keys[int(np.argmax(mat @ embs[i]))] for i in idx}
        if new == lab:
            break
        lab = new

    merge_at = float(c.get("centroid_merge", 1.01))
    while not c.max_speakers:
        cents = centroids()
        keys = list(cents)
        if len(keys) < 2:
            break
        sim, a, b = max((float(cents[a] @ cents[b]), a, b) for x, a in enumerate(keys) for b in keys[x + 1:])
        if sim < merge_at:
            break
        lab = {i: (a if k == b else k) for i, k in lab.items()}

    total = sum(durs[i] for i in idx)
    min_secs = max(c.min_speaker_seconds, float(c.get("min_speaker_share", 0.0)) * total)
    while True:
        cents = centroids()
        secs = {k: sum(durs[i] for i in idx if lab[i] == k) for k in cents}
        small = [k for k in cents if secs[k] < min_secs]
        if not small or len(cents) == 1:
            break
        k = min(small, key=lambda s: secs[s])
        target = max((o for o in cents if o != k), key=lambda o: float(cents[k] @ cents[o]))
        lab = {i: (target if x == k else x) for i, x in lab.items()}

    cents = centroids()
    keys = list(cents)
    mat = np.stack([cents[k] for k in keys])
    labels, sims = [], []
    for i in range(n):
        if i in lab:
            k = lab[i]
        else:
            k = keys[int(np.argmax(mat @ embs[i]))]
        labels.append(k)
        sims.append(float(cents[k] @ embs[i]))
    return labels, sims


def run(job: Job, cfg: Cfg):
    c = cfg.diarization
    lines = job.p.lines
    if not lines:
        job.p.speakers = {}
        return
    embs = _embed_all(job, cfg)
    np.save(job.path("embeddings.npy"), embs)

    labels, sims = cluster(embs, [l.duration for l in lines], c, [overlap.separated_share(l) < 0.5 for l in lines])
    assign = {i: (labels[i], sims[i]) for i in range(len(lines))}

    # stable ids, most talkative first
    totals = {}
    for i, (k, _) in assign.items():
        totals[k] = totals.get(k, 0.0) + lines[i].duration
    order = sorted(totals, key=lambda k: -totals[k])
    sid = {k: f"S{n + 1:02d}" for n, k in enumerate(order)}

    old = job.p.speakers
    job.p.speakers = {}
    for k in order:
        s = sid[k]
        job.p.speakers[s] = old.get(s) if old.get(s) and old[s].user_edited else Speaker(id=s)
    for i, line in enumerate(lines):
        k, sim = assign[i]
        line.speaker, line.speaker_conf = sid[k], round(sim, 3)
    _merge_fragments(job, cfg)
    for sid, spk in job.p.speakers.items():
        mine = [l for l in job.p.lines if l.speaker == sid]
        spk.line_count, spk.total_seconds = len(mine), round(sum(l.duration for l in mine), 2)
    job.log(f"found {len(job.p.speakers)} speakers")


_SENTENCE_END = tuple("。！？!?…♪.")


def _merge_fragments(job: Job, cfg: Cfg):
    """Re-join pieces of one sentence that VAD or the voice-change check cut apart.

    Consecutive lines merge when they have the same speaker, a short gap (no overlap), and the
    first piece doesn't end a sentence. Two overlapping lines from one crosstalk split that turn
    out to be the same speaker were one voice split in two: they merge back and are heard from
    the original audio again. Merged lines are re-transcribed as a whole, which fixes words that
    were cut at the boundary.
    """
    from .transcribe import transcribe_spans

    d = cfg.diarization
    lines = job.p.lines
    merged: list[list] = []
    for l in lines:
        prev = merged[-1][-1] if merged else None
        if prev is not None and prev.speaker == l.speaker and (
            (
                -0.01 <= l.start - prev.end <= d.merge_gap
                and l.end - merged[-1][0].start <= cfg.asr.max_line_seconds
                and not prev.ja.rstrip().endswith(_SENTENCE_END)
            )
            or (l.start < prev.end and _windows(prev) & _windows(l))
        ):
            merged[-1].append(l)
        else:
            merged.append([l])
    groups = [g for g in merged if len(g) > 1]
    if not groups:
        return
    y, _ = audio.load(job.path("vocals_16k.wav"), mono=True)
    job.progress(0.85, f"re-transcribing {len(groups)} re-joined lines", force=True)
    seps, rejoined = [], 0
    for g in groups:
        refs = []
        for x in g:
            refs += [r for r in x.sep if r not in refs]
        # a split whose both voices are in this group was wrong: hear that window as recorded
        split = {w for w in _windows(*g) if sum(1 for r in refs if (r.start, r.end) == w) > 1}
        rejoined += len(split)
        seps.append([r for r in refs if (r.start, r.end) not in split])
    texts = transcribe_spans(job, cfg, y, [(g[0].start, max(x.end for x in g)) for g in groups], lo=0.85, hi=0.98, seps=seps)
    for g, sep, (text, alt) in zip(groups, seps, texts):
        first = g[0]
        first.end = max(x.end for x in g)
        first.ja = first.ja_asr = text or "".join(x.ja for x in g)
        first.ja_alt = alt
        first.sep = sep
        first.speaker_conf = round(min(x.speaker_conf for x in g), 3)
    job.p.lines = [g[0] for g in merged]
    for i, l in enumerate(job.p.lines, start=1):
        l.id = i
    job.log(f"merged {sum(len(g) - 1 for g in groups)} sentence fragments"
            + (f"; undid {rejoined} crosstalk splits that were one voice" if rejoined else ""))


def _windows(*lines) -> set[tuple[float, float]]:
    """The crosstalk windows these lines' separated audio comes from."""
    return {(r.start, r.end) for l in lines for r in l.sep}
