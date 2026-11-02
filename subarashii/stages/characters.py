"""Who each speaker is: name, gender, age, personality, plus a glossary of recurring terms.

A local LLM guesses confidently and often wrongly, so its claims are checked in code:
- a name is kept only if the dialogue shows it (the character is addressed by that name and
  answers, or introduces themselves); otherwise the card gets a plain description;
- gender comes first from hard dialogue evidence (Japanese self-pronouns 俺/僕 vs あたし and
  gendered sentence endings; Korean kinship terms 오빠/언니 vs 형/누나), then the voice analysis,
  then the LLM.
"""

import json

from ..config import Cfg
from ..gpu import TTSClient
from ..lang import INTRO, gender_score, job_lang, romanize_name
from ..llm import get_llm
from ..project import Job, Speaker

SYSTEM = """You analyse {lang} dialogue transcripts for an English dubbing team.
Speakers were clustered automatically from voice, so a label may occasionally mix two people.
Using the dialogue (how characters address each other, how they refer to themselves, {cues}
politeness, topic) plus the acoustic hints, describe each speaker.

Names: only give a name when the transcript shows it. Either the character introduces themselves
({intro_examples}), or another speaker addresses them by name and they answer within
the next few lines. A name a speaker SAYS usually belongs to the person they are talking to, not
to the speaker. Put the name exactly as written in the transcript in name_original, its {romanization}
in name, and the id of the line that proves it in evidence_line. If there is no such
proof, leave name, name_original empty and evidence_line 0.

Always give label: a short English description a viewer would recognise, e.g. "Old monk",
"Mother", "Teacher", "Narrator", "Girl in pigtails", "Boy 2".

Also summarize in 2-3 sentences what the dialogue reveals about the situation and relationships
(do not invent visual details), and give a glossary of recurring names, places and special terms
with their English dub rendering."""

PROMPT_HINTS = {
    "ja": {"cues": "self-reference like 俺/僕/私/あたし/わし, sentence endings,", "intro_examples": "…と申します, 私は…です"},
    "zh": {"cues": "terms of address like 哥/姐/师兄/师父,", "intro_examples": "我叫…, 我是…"},
    "ko": {"cues": "kinship terms (a girl says 오빠/언니, a boy says 형/누나), speech level (반말/존댓말),",
           "intro_examples": "저는 …입니다, 제 이름은 …예요"},
}

SCHEMA = {
    "type": "object",
    "properties": {
        "setting": {"type": "string"},
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "name": {"type": "string"},
                    "name_original": {"type": "string"},
                    "evidence_line": {"type": "integer"},
                    "label": {"type": "string"},
                    "gender": {"type": "string", "enum": ["male", "female", "unknown"]},
                    "age_group": {"type": "string", "enum": ["child", "teen", "young adult", "adult", "elderly", "unknown"]},
                    "personality": {"type": "string"},
                    "speech_style": {"type": "string"},
                },
                "required": ["id", "name", "name_original", "evidence_line", "label", "gender", "age_group", "personality", "speech_style"],
                "additionalProperties": False,
            },
        },
        "glossary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"original": {"type": "string"}, "en": {"type": "string"}},
                "required": ["original", "en"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["setting", "characters", "glossary"],
    "additionalProperties": False,
}



def _transcript(job: Job, max_chars: int) -> str:
    rows = [f"{l.id}|{l.speaker}|{l.ja}" for l in job.p.lines]
    text = "\n".join(rows)
    if len(text) <= max_chars:
        return text
    step = len(text) / max_chars  # keep an even sample across the video
    keep = rows[:: max(2, int(step + 1))]
    return "\n".join(keep)


def dialogue_gender(job: Job, sid: str, code: str) -> str:
    male = female = 0
    for l in job.p.lines:
        if l.speaker != sid:
            continue
        m, f = gender_score(l.ja, code)
        male, female = male + m, female + f
    if male >= 2 and male >= 2 * female:
        return "male"
    if female >= 2 and female >= 2 * male:
        return "female"
    return ""


def name_is_proven(job: Job, sid: str, name_ja: str, evidence: int, code: str) -> bool:
    """The name must appear in the evidence line, which is either the speaker introducing themselves
    or someone else addressing them right before they speak."""
    if not name_ja or len(name_ja) > 12:
        return False
    lines = job.p.lines
    idx = next((i for i, l in enumerate(lines) if l.id == evidence), None)
    if idx is None or name_ja not in lines[idx].ja:
        return False
    line = lines[idx]
    if line.speaker == sid:
        return bool(INTRO[code].search(line.ja))
    return any(l.speaker == sid for l in lines[idx + 1 : idx + 4])


def _fallback_label(spk: Speaker) -> str:
    table = {
        ("child", "male"): "Boy", ("child", "female"): "Girl",
        ("teen", "male"): "Teenage boy", ("teen", "female"): "Teenage girl",
        ("young adult", "male"): "Young man", ("young adult", "female"): "Young woman",
        ("adult", "male"): "Man", ("adult", "female"): "Woman",
        ("elderly", "male"): "Old man", ("elderly", "female"): "Old woman",
    }
    return table.get((spk.age_group, spk.gender), "Voice")


def run(job: Job, cfg: Cfg):
    TTSClient(cfg).unload()
    llm = get_llm(cfg)
    lang = job_lang(job, cfg)
    speakers = {
        sid: {
            "acoustic_gender_probs": {k: s.acoustic.get(k) for k in ("female", "male", "child")},
            "acoustic_age": s.acoustic.get("age_estimate"),
            "pitch_hz": s.acoustic.get("pitch_hz"),
            "lines": s.line_count,
            **({"known_name": s.name} if s.user_edited and s.name else {}),
        }
        for sid, s in job.p.speakers.items()
    }
    user = (
        f"Speakers:\n{json.dumps(speakers, ensure_ascii=False, indent=1)}\n\n"
        f"Transcript (id|speaker|{lang.name.lower()}):\n{_transcript(job, cfg.translation.max_transcript_chars)}"
    )
    job.progress(0.1, f"analysing characters with {llm.name}", force=True)
    out = llm.json(SYSTEM.format(lang=lang.name, romanization=lang.romanization, **PROMPT_HINTS[lang.code]),
                   user, SCHEMA, max_tokens=6000)
    job.progress(0.9, "checking names and genders against the dialogue", force=True)

    job.p.setting = out.get("setting", "")
    job.p.glossary = [{"ja": g["original"], "en": g["en"]} for g in out.get("glossary", [])
                      if g.get("original") and g.get("en")]
    sheet = {ch.get("id", ""): ch for ch in out.get("characters", [])}
    proven = rejected = 0
    for sid, spk in job.p.speakers.items():
        ch = sheet.get(sid, {})
        a = spk.acoustic
        if not (spk.user_edited and spk.gender):
            voice = "female" if a.get("female", 0) >= 0.85 else "male" if a.get("male", 0) >= 0.85 else ""
            spk.gender = dialogue_gender(job, sid, lang.code) or voice or ch.get("gender") or "unknown"
        if not (spk.user_edited and spk.age_group):
            age = ch.get("age_group") or "unknown"
            if a.get("child", 0) >= 0.4 and age not in ("child", "teen"):
                age = "child"
            spk.age_group = age
        for field in ("personality", "speech_style"):
            if not (spk.user_edited and getattr(spk, field)):
                setattr(spk, field, ch.get(field) or "")
        if spk.user_edited and spk.name:
            continue
        name_src = ch.get("name_original", "")
        if ch.get("name") and name_is_proven(job, sid, name_src, int(ch.get("evidence_line") or 0), lang.code):
            spk.name, spk.name_confident = romanize_name(ch["name"].strip() or name_src, lang.code), True
            proven += 1
            # make sure the translator renders this name the same way everywhere
            job.p.glossary = [g for g in job.p.glossary if g.get("ja") != name_src]
            job.p.glossary.append({"ja": name_src, "en": spk.name})
        else:
            rejected += bool(ch.get("name"))
            spk.name, spk.name_confident = (ch.get("label") or "").strip() or _fallback_label(spk), False

    # no two cards may share a name: the bigger part keeps it, the others get numbered
    seen: dict[str, int] = {}
    for spk in sorted(job.p.speakers.values(), key=lambda s: -s.total_seconds):
        key = spk.name.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            spk.name, spk.name_confident = f"{spk.name} {seen[key]}", False
    job.log(f"character sheet: {', '.join(s.label for s in job.p.speakers.values())} "
            f"({proven} names confirmed by the dialogue, {rejected} unproven names replaced by descriptions)")
