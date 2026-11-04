"""LLM proofreading of the original-language transcript, so recognition errors don't become translation errors."""

import json
import re

from ..config import Cfg
from ..gpu import TTSClient
from ..lang import job_lang, lang_cfg
from ..llm import get_llm
from ..project import Job

SYSTEM = """Two independent {lang} speech recognizers transcribed each line of a video,
"a" and "b" (some lines only have "a"). Punctuation was removed from both. For every line decide
which version is the real dialogue, judging only by meaning: does it make sense on its own and
in the conversation around it? A garbled version contains nonsense words or an ungrammatical
string. Merge them if each is half right. Never add content that neither version supports.
Write the chosen text with natural {lang} punctuation in "text", and say which source you mainly
used in "pick". Return every id you were given."""

SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "pick": {"type": "string", "enum": ["a", "b", "merged"]},
                    "text": {"type": "string"},
                },
                "required": ["id", "pick", "text"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["lines"],
    "additionalProperties": False,
}

# punctuation is stripped before the model compares versions, otherwise it prefers whichever
# recognizer punctuates more, not the one that heard the words right. Apostrophes and hyphens
# inside a word are spelling (Ukrainian м'ясо, Russian что-нибудь), not punctuation.
_PUNCT = re.compile(r"[、。，,.!?！？…・「」『』“”„«»\"；;：:（）()~〜—–]|(?<!\w)['‘’ʼ-]|['‘’ʼ-](?!\w)")


def _bare(text: str) -> str:
    # Korean separates words with spaces: keep them (collapsed) so the text stays readable
    return re.sub(r"\s+", " ", _PUNCT.sub("", text)).strip()


BATCH = 40
CONTEXT = 5


def _hypotheses(l, phrases) -> tuple[str, str]:
    """(a, b) originals with punctuation; a known recognizer hallucination is dropped."""
    a, b = l.ja_asr or l.ja, l.ja_alt
    if b and any(p in a for p in phrases):
        a, b = b, ""
    return a, b


def run(job: Job, cfg: Cfg):
    TTSClient(cfg).unload()
    llm = get_llm(cfg)
    lang = job_lang(job, cfg)
    system = SYSTEM.format(lang=lang.name)
    phrases = lang_cfg(cfg, lang).hallucination_phrases
    lines = job.p.lines
    changed = 0
    for b in range(0, len(lines), BATCH):
        batch = lines[b : b + BATCH]
        hyp = {l.id: _hypotheses(l, phrases) for l in batch}
        payload = {
            "context_before": [{"speaker": l.speaker, "text": l.ja} for l in lines[max(0, b - CONTEXT) : b]],
            "lines": [{"id": l.id, "speaker": l.speaker, "a": _bare(hyp[l.id][0]),
                       **({"b": _bare(hyp[l.id][1])} if hyp[l.id][1] else {})} for l in batch],
            "context_after": [{"speaker": l.speaker, "text": l.ja} for l in lines[b + BATCH : b + BATCH + CONTEXT]],
        }
        out = llm.json(system, json.dumps(payload, ensure_ascii=False, indent=1), SCHEMA, max_tokens=8000)
        fixed = {int(i["id"]): (i.get("text") or "").strip() for i in out.get("lines", []) if "id" in i}
        for l in batch:
            new = fixed.get(l.id, "")
            a, alt = hyp[l.id]
            # a straight pick gets the recognizer's own punctuation back
            new = next((h for h in (a, alt) if h and _bare(h) == _bare(new)), new)
            # ignore empty or wildly different rewrites: that's not proofreading
            longest = max(len(_bare(a)), len(_bare(alt)))
            if new and 0.5 <= len(_bare(new)) / max(1, longest) <= 1.3:
                changed += new != l.ja
                l.ja = new
        job.progress(min(1.0, (b + BATCH) / len(lines)), f"proofread {min(b + BATCH, len(lines))}/{len(lines)} lines")
    job.log(f"proofreading corrected {changed} of {len(lines)} lines")
