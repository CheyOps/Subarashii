"""Timing-aware dub adaptation: three English renderings per line at decreasing length."""

import json
import re

from ..config import Cfg
from ..gpu import TTSClient
from ..lang import job_lang
from ..llm import LLMError, get_llm
from ..project import Job, Line
from ..timing import assign_budgets, en_syllables, pick_candidate

SYSTEM = """You are a professional dub script adapter working {lang} -> English.
Your lines will be spoken by English voice actors over the original animation, so each line must
fit the time the character's mouth is moving.

For every line write three English versions:
- natural: the best dub line, idiomatic and in character, within syllables_max (aim near syllables_target).
- concise: same meaning and tone in about syllables_target syllables or fewer.
- minimal: the core meaning in roughly 60% of syllables_target; may drop nuance but must still sound natural.

Rules:
- Write speakable dialogue, not subtitles: contractions, natural rhythm, no brackets or notes.
- Keep each character's voice (personality, speech_style, gender, age). A rough boy talks rough;
  a polite girl stays polite; an excitable character keeps the energy (exclamations are fine).
- Keep interjections and reactions short and English-native ({interjections}).
- The {lang} comes from speech recognition and may contain a misheard word; translate what
  was most plausibly said in context.
- Everything you write is read aloud by a TTS voice: never write stage directions or action words
  like "Sigh", "*laughs*", "(gasps)". Write the sound itself if needed ("Haah...", "Hmph.", "Ha!").
- Use the glossary spellings for names and terms. {honorifics}
- Use previous lines for consistency; never translate the context lines themselves
  (previous_lines, lines_in_between, next_lines are context only).
- If a line has a "note", an earlier translation of it was rejected: follow the note.
- Each line is translated on its own (do not merge or move content between ids).
- Set is_lyric true only if the text is song lyrics (opening/ending/insert song, or any other sung lyrics), otherwise false.
- Numbers as words when spoken ("three", not "3")."""

SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "natural": {"type": "string"},
                    "concise": {"type": "string"},
                    "minimal": {"type": "string"},
                    "is_lyric": {"type": "boolean"},
                },
                "required": ["id", "natural", "concise", "minimal", "is_lyric"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}


def _characters(job: Job) -> list[dict]:
    return [
        {
            "speaker": sid,
            "name": s.name or sid,
            "gender": s.gender,
            "age_group": s.age_group,
            "personality": s.personality,
            "speech_style": s.speech_style,
        }
        for sid, s in job.p.speakers.items()
    ]


def _name(job: Job, sid: str) -> str:
    s = job.p.speakers.get(sid)
    return s.label if s else sid


def _request(job: Job, batch: list[Line], before: list[Line], after: list[Line],
             between: list[Line] = (), notes: dict[int, str] | None = None) -> str:
    def item(l: Line) -> dict:
        d = {
            "id": l.id,
            "speaker": _name(job, l.speaker),
            "duration_s": round(l.duration, 2),
            "syllables_target": l.syl_target,
            "syllables_max": l.syl_max,
            "original": l.ja,
        }
        if notes and notes.get(l.id):
            d["note"] = notes[l.id]
        return d

    payload = {
        "setting": job.p.setting,
        "characters": _characters(job),
        "glossary": [{"original": g["ja"], "en": g["en"]} for g in job.p.glossary],
        "previous_lines": [{"speaker": _name(job, l.speaker), "original": l.ja, "en": l.en} for l in before],
        "lines": [item(l) for l in batch],
        "next_lines": [{"speaker": _name(job, l.speaker), "original": l.ja} for l in after],
    }
    if between:
        payload["lines_in_between"] = [{"speaker": _name(job, l.speaker), "original": l.ja, "en": l.en} for l in between]
    return json.dumps(payload, ensure_ascii=False, indent=1)


_STAGE_DIRECTION = re.compile(r"\*[^*]*\*|\([^)]*\)|\[[^\]]*\]")
_SOUND_WORDS = {r"sighs?": "Haah", r"gasps?": "Ah", r"laughs?": "Haha", r"chuckles?": "Heh", r"giggles?": "Hehe",
                r"groans?": "Ugh", r"scoffs?": "Hmph", r"sniffs?": "Hn", r"huffs?": "Hmph"}
_SOUND_RE = re.compile(r"^\s*(" + "|".join(_SOUND_WORDS) + r")\b[.!…,]*", re.I)
_HONORIFIC_RE = re.compile(r"\b([A-Z][\w']+)-(san|kun|chan|sama|dono|tan|senpai|sensei|ssi|nim)\b")


def clean_for_tts(text: str, honorifics: str) -> str:
    """The voice reads everything literally: strip stage directions, turn action words into sounds."""
    text = _STAGE_DIRECTION.sub("", text)
    m = _SOUND_RE.match(text)
    if m:
        sound = next(v for k, v in _SOUND_WORDS.items() if re.fullmatch(k, m.group(1), re.I))
        text = sound + "..." + text[m.end():]
    if honorifics == "drop":
        text = _HONORIFIC_RE.sub(r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,")


def _apply(line: Line, item: dict, cfg: Cfg):
    hon = cfg.translation.honorifics
    line.candidates = {k: clean_for_tts((item.get(k) or "").strip(), hon) for k in ("natural", "concise", "minimal")}
    line.lyric = bool(item.get("is_lyric"))
    if not line.en_edited:
        line.en = pick_candidate(line, cfg.timing.tts_syllable_rate)


def translate_lines(job: Job, cfg: Cfg, llm, lines: list[Line], progress=None, notes: dict[int, str] | None = None):
    t = cfg.translation
    lang = job_lang(job, cfg)
    system = SYSTEM.format(lang=lang.name, interjections=lang.interjections,
                           honorifics=lang.honorifics.get(t.honorifics, lang.honorifics["drop"]))
    ordered = job.p.lines
    pos = {l.id: i for i, l in enumerate(ordered)}
    todo = list(lines)
    for b in range(0, len(todo), t.batch_size):
        batch = todo[b : b + t.batch_size]
        first, last = pos[batch[0].id], pos[batch[-1].id]
        before = ordered[max(0, first - t.context_lines) : first]
        after = ordered[last + 1 : last + 1 + t.context_lines]
        ids = {l.id for l in batch}
        between = [l for l in ordered[first:last] if l.id not in ids and not l.skip]  # batches may be scattered
        out = llm.json(system, _request(job, batch, before, after, between, notes), SCHEMA, max_tokens=8000)
        got = {int(i["id"]): i for i in out.get("lines", []) if "id" in i}
        missing = [l for l in batch if l.id not in got]
        for l in batch:
            if l.id in got:
                _apply(l, got[l.id], cfg)
        for l in missing:  # retry stragglers one at a time
            single = llm.json(system, _request(job, [l], before, after, between, notes), SCHEMA, max_tokens=2000)
            item = next((i for i in single.get("lines", []) if int(i.get("id", -1)) == l.id), None)
            if item is None:
                raise LLMError(f"LLM did not translate line {l.id}")
            _apply(l, item, cfg)
        if progress:
            progress(min(1.0, (b + len(batch)) / len(todo)))


def run(job: Job, cfg: Cfg):
    TTSClient(cfg).unload()
    assign_budgets(job.p.lines, cfg, job_lang(job, cfg).code)
    llm = get_llm(cfg)
    todo = [l for l in job.p.lines if not l.skip]
    job.progress(0.0, f"translating {len(todo)} lines with {llm.name}", force=True)
    translate_lines(job, cfg, llm, todo, progress=lambda f: job.progress(f, f"translating ({int(f * 100)}%)"))
    over = sum(1 for l in todo if en_syllables(l.en) > l.syl_max)
    lyr = sum(1 for l in todo if l.lyric)
    job.log(f"translated {len(todo)} lines ({lyr} flagged as lyrics, {over} still long for their slot)")
