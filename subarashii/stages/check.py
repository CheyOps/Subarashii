"""Script check: an LLM editor reads the translated script scene by scene and flags lines a viewer
would misunderstand (wrong meaning, misheard dialogue, a reply that doesn't follow, a name or term
rendered two ways). Flagged lines are re-translated with the problem explained, and a blind A/B
judge must prefer the new line before it replaces the old one (a weak critic otherwise makes
lines worse). Scenes that changed are read again, up to check.rounds passes. Lines the editor
still objects to after a rewrite was tried are marked for the viewer."""

import json
import random

from ..config import Cfg, deep_merge
from ..gpu import TTSClient
from ..lang import Lang, job_lang, name_in
from ..llm import LLMError, get_llm
from ..project import Job, Line
from .translate import translate_lines

SYSTEM = """You are the script editor for an English dub of a {lang} video. Read the scene as an
English-speaking viewer would and find lines that would mislead or confuse them.

The {lang} came from speech recognition, so it can contain misheard words (a similar-sounding
word, a garbled name). "original_other" is a second recognizer's hearing when it differs.

Problems:
- mistranslation: the English says something different from the original.
- misheard: the original itself is a likely mishearing; give the corrected text in original_fixed.
- doesnt_follow: the line contradicts or ignores what was just said.
- inconsistent: a name or term is rendered differently from the glossary or from other lines.
Severity "major" means a viewer would misunderstand what is happening, who is meant, or what
someone wants. Anything else is "minor": tone, nuance, wording, length, punctuation.
Fragments, mumbles, interrupted lines and short reactions are normal dialogue, not problems.
Judge only by the original text and the scene. Never "correct" names, numbers, titles or events
from your own knowledge: the video may be newer than what you know, or fiction. A detail the
original contains (a brand, a number, a place) must stay in the English.
Most lines are fine; return an empty list when the scene reads well. Only flag ids from "lines".

For each flag give a one-sentence explanation a viewer who doesn't speak {lang} understands,
original_fixed (empty unless the problem is a mishearing), and a better English line."""

SCHEMA = {
    "type": "object",
    "properties": {
        "flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "problem": {"type": "string", "enum": ["mistranslation", "misheard", "doesnt_follow", "inconsistent"]},
                    "severity": {"type": "string", "enum": ["major", "minor"]},
                    "explanation": {"type": "string"},
                    "original_fixed": {"type": "string"},
                    "en_suggestion": {"type": "string"},
                },
                "required": ["id", "problem", "severity", "explanation", "original_fixed", "en_suggestion"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["flags"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """Each line of an English dub has two candidate versions, "1" and "2". Using the
original {lang} line (it may contain speech-recognition errors), the conversation around it and
who is speaking, pick the version that says what the original says (keeping every detail it contains: names,
numbers, places) and that sounds like natural dialogue. Judge by the original, not by your own
knowledge of the world. Answer "same" if neither is clearly better. Return every id."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "integer"}, "better": {"type": "string", "enum": ["1", "2", "same"]}},
                "required": ["id", "better"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}

PROBLEM_TEXT = {
    "mistranslation": "meaning was off",
    "misheard": "the original was misheard",
    "doesnt_follow": "didn't follow the conversation",
    "inconsistent": "inconsistent name/term",
}


def _checkable(l: Line) -> bool:
    return not l.skip and not l.lyric and bool(l.en.strip())


def _speaker(job: Job, l: Line) -> str:
    s = job.p.speakers.get(l.speaker)
    return s.label if s else l.speaker


def _context(job: Job, window: list[Line], before: list[Line], after: list[Line], item) -> str:
    return json.dumps({
        "setting": job.p.setting,
        "characters": [
            {"speaker": s.label, "gender": s.gender, "age_group": s.age_group, "personality": s.personality}
            for s in job.p.speakers.values()
        ],
        "glossary": [{"original": g["ja"], "en": g["en"]} for g in job.p.glossary],
        "context_before": [{"speaker": _speaker(job, l), "en": l.en} for l in before],
        "lines": [item(l) for l in window],
        "context_after": [{"speaker": _speaker(job, l), "en": l.en} for l in after],
    }, ensure_ascii=False, indent=1)


def _read(job: Job, llm, lang: str, window, before, after) -> list[dict]:
    def item(l: Line) -> dict:
        d = {"id": l.id, "speaker": _speaker(job, l), "original": l.ja}
        if l.ja_alt and l.ja_alt != l.ja:
            d["original_other"] = l.ja_alt
        d["en"] = l.en
        return d

    try:
        out = llm.json(SYSTEM.format(lang=lang), _context(job, window, before, after, item), SCHEMA, max_tokens=12000)
    except LLMError as e:
        # usually a reply cut off on a scene of long lines: read it in halves; the check is an
        # improvement pass and must never fail the job
        if len(window) >= 8:
            h = len(window) // 2
            return (_read(job, llm, lang, window[:h], before, window[h:])
                    + _read(job, llm, lang, window[h:], window[:h], after))
        job.log(f"script check: skipped lines {window[0].id}-{window[-1].id} ({e})")
        return []
    ids = {l.id for l in window}
    return [f for f in out.get("flags", []) if int(f.get("id", -1)) in ids and f.get("severity") == "major"]


def _judge(job: Job, llm, lang: str, window, before, after, old: dict[int, str]) -> set[int]:
    """Ids whose new English the judge prefers over the old (order randomised per line)."""
    flip = {lid: random.Random(lid).random() < 0.5 for lid in old}

    def item(l: Line) -> dict:
        d = {"id": l.id, "speaker": _speaker(job, l), "original": l.ja}
        if l.id in old:
            a, b = (l.en, old[l.id]) if flip[l.id] else (old[l.id], l.en)
            d["1"], d["2"] = a, b
        else:
            d["en"] = l.en
        return d

    try:
        out = llm.json(JUDGE_SYSTEM.format(lang=lang), _context(job, window, before, after, item), JUDGE_SCHEMA,
                       max_tokens=6000)
    except LLMError as e:
        job.log(f"script check: kept the old wording for lines {sorted(old)} (judge failed: {e})")
        return set()
    won = set()
    for r in out.get("lines", []):
        lid = int(r.get("id", -1))
        if lid in old and r.get("better") == ("1" if flip[lid] else "2"):
            won.add(lid)
    return won


def _invented_name(job: Job, l: Line, old: str, source: Lang) -> str:
    """A name/term from the glossary that the rewrite brings in although the original line doesn't
    contain it (any 2+ character piece of its original spelling; for Russian and Ukrainian, the stem
    of any of its words). Weak models 'fix' unclear lines by inserting a character who isn't there."""
    src = l.ja + " " + l.ja_alt
    for g in job.p.glossary:
        words = [w for w in g["en"].split() if len(w) >= 3 and w[0].isupper()]
        if not any(w in l.en and w not in old for w in words):
            continue
        orig = g["ja"]
        if source.inflected:
            if not any(name_in(w, src, source) for w in orig.split()):
                return g["en"]
        elif not any(orig[i : i + 2] in src for i in range(max(1, len(orig) - 1))):
            return g["en"]
    return ""


def _windows(lines: list[Line], size: int) -> list[tuple[int, int]]:
    """Split into roughly equal windows of at most `size` lines."""
    if not lines:
        return []
    count = max(1, -(-len(lines) // size))
    step = -(-len(lines) // count)
    return [(i, min(len(lines), i + step)) for i in range(0, len(lines), step)]


def run(job: Job, cfg: Cfg):
    c = cfg.check
    if not c.enabled:
        job.log("script check skipped (check.enabled is off)")
        return
    TTSClient(cfg).unload()
    if cfg.llm.provider == "llamacpp" and c.get("llamacpp_model"):
        cfg = Cfg(deep_merge(cfg, {"llm": {"llamacpp": {"model": c.llamacpp_model}}}))
    llm = get_llm(cfg)
    source = job_lang(job, cfg)
    lang = source.name
    lines = [l for l in job.p.lines if _checkable(l)]
    for l in lines:
        l.check_note, l.check_unresolved = "", False
    spans = _windows(lines, c.window)
    todo = list(range(len(spans)))
    tried: set[int] = set()       # lines a rewrite was attempted for
    rewritten: set[int] = set()
    total_steps, step = len(todo) * c.rounds, 0

    for rnd in range(c.rounds):
        changed: set[int] = set()
        for w in todo:
            a, b = spans[w]
            window = lines[a:b]
            step += 1
            job.progress(min(0.99, step / total_steps), f"reading lines {window[0].id}-{window[-1].id} (pass {rnd + 1})")
            before = lines[max(0, a - c.context_lines) : a]
            after = lines[b : b + c.context_lines]
            by_id = {l.id: l for l in window}
            fix, notes, why = [], {}, {}
            for f in _read(job, llm, lang, window, before, after):
                l = by_id[int(f["id"])]
                if l.en_edited or l.id in why:
                    continue
                expl = (f.get("explanation") or "").strip()
                why[l.id] = PROBLEM_TEXT.get(f.get("problem"), "unclear") + (f": {expl}" if expl else "")
                if l.id in tried:  # still objected to after a rewrite was tried
                    l.check_unresolved = True
                    l.check_note = f"May still be unclear: {expl or why[l.id]}"
                fixed = (f.get("original_fixed") or "").strip()
                notes[l.id] = (f"An earlier translation was rejected ({why[l.id]}). "
                               f"Editor's suggestion: {(f.get('en_suggestion') or '').strip()}"
                               + (f" The original was probably misheard; it likely says: {fixed}" if fixed else ""))
                fix.append(l)
            if not fix:
                continue
            saved = {l.id: (l.en, dict(l.candidates)) for l in fix}
            fix.sort(key=lambda l: l.start)
            try:
                translate_lines(job, cfg, llm, fix, notes=notes)
            except LLMError as e:
                for l in fix:
                    l.en, l.candidates = saved[l.id]
                job.log(f"script check: couldn't re-translate lines {[l.id for l in fix]} ({e})")
                continue
            tried.update(l.id for l in fix)
            old = {l.id: saved[l.id][0] for l in fix if l.en.strip() != saved[l.id][0].strip()
                   and not _invented_name(job, l, saved[l.id][0], source)}
            won = _judge(job, llm, lang, window, before, after, old) if old else set()
            for l in fix:
                if l.id in won:
                    l.check_note = f"Rewritten by the script check ({why[l.id]})"
                    l.check_unresolved = False
                    rewritten.add(l.id)
                    changed.add(w)
                else:
                    l.en, l.candidates = saved[l.id]
            job.save()
        if not changed:
            break
        # re-read changed windows and their neighbours, whose context changed
        todo = sorted({n for w in changed for n in (w - 1, w, w + 1) if 0 <= n < len(spans)})
        total_steps = step + len(todo) * (c.rounds - rnd - 1)

    unresolved = sum(l.check_unresolved for l in lines)
    job.log(f"script check: {len(rewritten)} lines rewritten, {unresolved} may still be unclear "
            f"(marked for review) out of {len(lines)}")
