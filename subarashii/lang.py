"""Source languages. Everything that differs between Japanese, Chinese and Korean input lives here
or under `languages.<code>` in the config; the pipeline asks `job_lang()` which one a job uses."""

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Lang:
    code: str           # ISO 639-1, also Whisper's language code
    name: str
    iso3: str           # ISO 639-2/T tag for the original track (mp4)
    iso3b: str          # ISO 639-2/B tag (mkv)
    romanization: str   # how names are written for English viewers
    interjections: str  # examples for the translator
    honorifics: dict    # translation.honorifics -> instruction


LANGS = {
    "ja": Lang(
        "ja", "Japanese", "jpn", "jpn", "Hepburn romanization",
        'え？ -> "Huh?", うそ -> "No way!"',
        {
            "drop": "Drop honorifics (-san, -kun, -chan, senpai, sensei) unless meaningful; use names or English equivalents.",
            "keep": "Keep honorifics attached to names as a dub would (Tanaka-san, Senpai).",
        },
    ),
    "zh": Lang(
        "zh", "Chinese", "zho", "chi", "Hanyu Pinyin without tone marks (family name first, e.g. Li Xiaolong)",
        '啊？ -> "Huh?", 不会吧 -> "No way!"',
        {
            "drop": "Render forms of address (师父, 师兄, 哥, 姐, 老师, 大人) as natural English (Master, the name, Sir), not romanized titles.",
            "keep": "Keep forms of address romanized where a dub would (Shifu, Shixiong, Gege, Jiejie).",
        },
    ),
    "ko": Lang(
        "ko", "Korean", "kor", "kor", "Revised Romanization (family name first, e.g. Kim Minjun)",
        '어? -> "Huh?", 말도 안 돼 -> "No way!"',
        {
            "drop": "Drop honorifics and address terms (-ssi, -nim, oppa, hyung, unnie, noona, sunbae) unless meaningful; use names or English equivalents.",
            "keep": "Keep Korean address terms as a dub would (Oppa, Hyung, Unnie, Noona, Sunbae, Minjun-ssi).",
        },
    ),
}
SUPPORTED = tuple(LANGS)


def job_lang(job, cfg) -> Lang:
    """The job's detected/chosen language; jobs from before multi-language support are Japanese."""
    code = job.p.language or cfg.input.language
    return LANGS.get(code, LANGS["ja"])


def lang_cfg(cfg, lang: Lang):
    return getattr(cfg.languages, lang.code)


# ------------------------------------------------------------------ speech length
_CJK = re.compile(r"[㐀-鿿豈-﫿]")
_HANGUL = re.compile(r"[가-힣]")
_LATIN_OR_DIGIT = re.compile(r"[A-Za-z]+|\d")


def speech_units(text: str, code: str) -> int:
    """Spoken length of a line: morae for Japanese, syllables for Chinese (one per character)
    and Korean (one per Hangul block). Latin words count one, digits one each."""
    if code == "ja":
        from .timing import ja_morae

        return ja_morae(text)
    script = _CJK if code == "zh" else _HANGUL
    return len(script.findall(text)) + len(_LATIN_OR_DIGIT.findall(text))


# ------------------------------------------------------------------ names
_KO_SURNAMES = {"김": "Kim", "이": "Lee", "박": "Park", "최": "Choi", "정": "Jung", "강": "Kang", "조": "Cho",
                "윤": "Yoon", "장": "Jang", "임": "Lim", "한": "Han", "오": "Oh", "서": "Seo", "신": "Shin",
                "권": "Kwon", "황": "Hwang", "안": "Ahn", "송": "Song", "류": "Ryu", "유": "Yoo", "전": "Jeon",
                "홍": "Hong", "고": "Ko", "문": "Moon", "양": "Yang", "손": "Son", "배": "Bae", "백": "Baek"}
_KO_INITIAL = ["g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h"]
_KO_MEDIAL = ["a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi",
              "yu", "eu", "ui", "i"]
_KO_FINAL = ["", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "l", "l", "l", "p", "l", "m", "p", "p", "t", "t",
             "ng", "t", "t", "k", "t", "p", "t"]


def _hangul_rr(text: str) -> str:
    out = []
    for ch in text:
        n = ord(ch) - 0xAC00
        if 0 <= n < 11172:
            out.append(_KO_INITIAL[n // 588] + _KO_MEDIAL[(n % 588) // 28] + _KO_FINAL[n % 28])
        else:
            out.append(ch)
    return "".join(out)


def romanize_name(name: str, code: str) -> str:
    """The LLM sometimes returns a name in the original script; viewers need Latin letters."""
    name = name.strip()
    if name.isascii():
        return name
    if code == "zh" and _CJK.search(name):
        from pypinyin import lazy_pinyin

        syl = [s for s in lazy_pinyin(name) if s.strip()]
        if len(syl) >= 2:
            return f"{syl[0].capitalize()} {''.join(syl[1:]).capitalize()}"
        return "".join(syl).capitalize()
    if code == "ko" and _HANGUL.search(name):
        if len(name) == 3 and _HANGUL.fullmatch(name[0]):
            family = _KO_SURNAMES.get(name[0], _hangul_rr(name[0]).capitalize())
            return f"{family} {_hangul_rr(name[1:]).capitalize()}"
        return _hangul_rr(name).capitalize()
    from .timing import _kakasi

    return "".join(item["hepburn"] for item in _kakasi().convert(name)).strip().capitalize()


# ------------------------------------------------------------------ dialogue evidence
# Japanese: self-pronouns and sentence endings. Korean: kinship address terms depend on the
# speaker's gender (a girl says 오빠/언니, a boy 형/누나). Spoken Chinese has no reliable marker.
_GENDER_CUES = {
    "ja": {
        "male": [(re.compile(r"俺|おれ|オレ|僕|ぼく|ボク|わし|儂"), 2), (re.compile(r"(だぜ|ぜ|だぞ)[。！？!?…]*$"), 1)],
        "female": [(re.compile(r"あたし|アタシ"), 2), (re.compile(r"(わよ|のよ|かしら|だわ|わね)[。！？!?…]*$"), 1)],
    },
    "ko": {
        "male": [(re.compile(r"(?<![가-힣])누나|(?<![가-힣])형(?=$|[아야님이은는도한!?,.…\s])"), 1)],
        "female": [(re.compile(r"(?<![가-힣])(오빠|언니)"), 1)],
    },
    "zh": {"male": [], "female": []},
}

# a speaker naming themselves
INTRO = {
    "ja": re.compile(r"申します|と言います|といいます|です"),
    "zh": re.compile(r"我叫|我是|我的名字|名叫|在下|本人"),
    "ko": re.compile(r"저는|나는|제 이름|내 이름|입니다|이에요|예요|라고 합니다|라고 해"),
}


def gender_score(text: str, code: str) -> tuple[int, int]:
    cues = _GENDER_CUES.get(code, _GENDER_CUES["zh"])
    male = sum(w * len(rx.findall(text)) for rx, w in cues["male"])
    female = sum(w * len(rx.findall(text)) for rx, w in cues["female"])
    return male, female
