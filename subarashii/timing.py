"""Speech-length estimation used to keep the English dub inside the original line timing."""

import re
from functools import lru_cache

from .config import Cfg
from .project import Line

_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_WORD_RE = re.compile(r"[A-Za-z']+|\d+")


@lru_cache(maxsize=1)
def _kakasi():
    import pykakasi

    return pykakasi.kakasi()


def ja_morae(text: str) -> int:
    """Approximate mora count: kana after kanji->reading conversion, small kana merged."""
    count = 0
    for item in _kakasi().convert(text):
        for ch in item["hira"]:
            if "ぁ" <= ch <= "ゟ" or "ァ" <= ch <= "ヿ":
                if ch not in _SMALL_KANA:
                    count += 1
            elif ch.isalnum():
                count += 1
    return count


def _word_syllables(word: str) -> int:
    w = word.lower().strip("'")
    if not w:
        return 0
    if w.isdigit():
        return max(1, int(len(w) * 1.6))
    if len(w) <= 3:
        return 1
    groups = re.findall(r"[aeiouy]+", w)
    n = len(groups)
    if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and n > 1:
        n -= 1
    if w.endswith(("ed", "es")) and not w.endswith(("ted", "ded", "ses", "zes", "ches", "shes")) and n > 1:
        n -= 1
    return max(1, n)


def en_syllables(text: str) -> int:
    return sum(_word_syllables(w) for w in _WORD_RE.findall(text))


def assign_budgets(lines: list[Line], cfg: Cfg, code: str = "ja"):
    """Give each line its English syllable target/max and the latest time it may end (slot_end)."""
    from .lang import speech_units

    t = cfg.timing
    ordered = sorted(lines, key=lambda l: l.start)
    for i, line in enumerate(ordered):
        later = [o for o in ordered[i + 1 :] if o.start > line.start]
        # a line that starts inside this one is crosstalk: it doesn't cut this line short, but
        # this one shouldn't run on past its own end into the rest of it either
        nxt = next((o.start for o in later if o.start >= line.end), None)
        slot = line.end + (0.0 if any(o.start < line.end < o.end for o in later) else t.max_overflow)
        if nxt is not None:
            slot = min(slot, max(line.end, nxt - t.min_gap))
        line.slot_end = round(slot, 3)

        dur = max(line.duration, 0.3)
        units = speech_units(line.ja, code) or 1
        rate = (units / dur) * getattr(cfg.languages, code).unit_to_en_syllable
        rate = min(max(rate, t.min_syllable_rate), t.max_syllable_rate)
        line.syl_target = max(1, round(dur * rate))
        # most the TTS can say in the slot at its fastest allowed speed-up, with a little headroom
        avail = line.slot_end - line.start
        fit_rate = min(t.max_syllable_rate, t.tts_syllable_rate / t.min_duration_factor * 0.9)
        line.syl_max = max(line.syl_target, round(avail * fit_rate))


def pick_candidate(line: Line, syl_rate: float) -> str:
    """Longest-quality candidate whose estimated spoken length fits the slot."""
    avail = line.slot_end - line.start
    for key in ("natural", "concise", "minimal"):
        text = (line.candidates.get(key) or "").strip()
        if text and en_syllables(text) / syl_rate <= avail * 1.05:
            return text
    for key in ("minimal", "concise", "natural"):
        if line.candidates.get(key):
            return line.candidates[key].strip()
    return ""


def shorter_candidates(line: Line, current: str) -> list[str]:
    cur = en_syllables(current)
    seen, out = {current.strip()}, []
    for key in ("concise", "minimal"):
        text = (line.candidates.get(key) or "").strip()
        if text and text not in seen and en_syllables(text) < cur:
            out.append(text)
            seen.add(text)
    return sorted(out, key=en_syllables, reverse=True)
