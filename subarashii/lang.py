"""Source languages. Everything that differs between Japanese, Chinese, Korean, Russian and Ukrainian
input lives here or under `languages.<code>` in the config; the pipeline asks `job_lang()` which one a job uses."""

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
    inflected: bool = False  # names change their endings with grammatical case (Маша -> Машу)


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
    "ru": Lang(
        "ru", "Russian", "rus", "rus", "common English spelling (BGN/PCGN, e.g. Aleksandr, Natasha, Yevgeny)",
        'А? -> "Huh?", Да ладно! -> "No way!"',
        {
            "drop": "Render each character's name one consistent way: name-and-patronymic address (Ivan Petrovich) and diminutives (Sasha, Sashenka for Aleksandr) become the plain name or Sir/Ma'am; show formal вы vs familiar ты through tone, not words.",
            "keep": "Keep name-and-patronymic address (Ivan Petrovich) and familiar diminutives (Sasha) as a dub would.",
        },
        inflected=True,
    ),
    "uk": Lang(
        "uk", "Ukrainian", "ukr", "ukr", "Ukrainian national transliteration (e.g. Oleksandr, Kyiv, Yevhen; not Russian spellings)",
        'Га? -> "Huh?", Та ну! -> "No way!"',
        {
            "drop": "Render each character's name one consistent way: name-and-patronymic address (Ivan Petrovych), пане/пані and diminutives (Sashko, Oksanka) become the plain name or Sir/Ma'am; show formal ви vs familiar ти through tone, not words.",
            "keep": "Keep name-and-patronymic address (Ivan Petrovych) and familiar diminutives (Sashko) as a dub would; пане/пані -> Mr./Mrs. or Sir/Ma'am.",
        },
        inflected=True,
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
# one syllable per character (Chinese), Hangul block (Korean) or vowel letter (Russian, Ukrainian)
_SYLLABLE = {
    "zh": _CJK,
    "ko": _HANGUL,
    "ru": re.compile(r"[аеёиоуыэюя]", re.I),
    "uk": re.compile(r"[аеєиіїоуюя]", re.I),
}


def speech_units(text: str, code: str) -> int:
    """Spoken length of a line: morae for Japanese, syllables for the others. Latin words count
    one, digits one each."""
    if code == "ja":
        from .timing import ja_morae

        return ja_morae(text)
    return len(_SYLLABLE[code].findall(text)) + len(_LATIN_OR_DIGIT.findall(text))


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


# Russian: BGN/PCGN without diacritics. Ukrainian: the national system (2010), whose letters
# є ї й ю я are spelled ye yi y yu ya at the start of a name and ie i i iu ia inside it.
_RU_LATIN = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
                     ["a", "b", "v", "g", "d", "e", "yo", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p", "r",
                      "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"]))
_UK_LATIN = dict(zip("абвгґдеєжзиіїйклмнопрстуфхцчшщьюя'’ʼ",
                     ["a", "b", "v", "h", "g", "d", "e", "ie", "zh", "z", "y", "i", "i", "i", "k", "l", "m", "n",
                      "o", "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "iu", "ia", "", "", ""]))
_UK_INITIAL = {"є": "ye", "ї": "yi", "й": "y", "ю": "yu", "я": "ya"}
_RU_VOWELS = set("аеёиоуыэюяъь")


def _cyrillic_latin(word: str, code: str) -> str:
    out, prev = [], ""
    for ch in word.lower():
        if code == "uk":
            if ch == "г" and prev == "з":
                out.append("gh")  # зг -> zgh, so it doesn't read as zh
            else:
                out.append(_UK_INITIAL[ch] if not prev and ch in _UK_INITIAL else _UK_LATIN.get(ch, ch))
        else:
            out.append("ye" if ch == "е" and (not prev or prev in _RU_VOWELS) else _RU_LATIN.get(ch, ch))
        prev = ch
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
    if code in ("ru", "uk"):
        return " ".join(_cyrillic_latin(w, code).capitalize() for w in name.split())
    from .timing import _kakasi

    return "".join(item["hepburn"] for item in _kakasi().convert(name)).strip().capitalize()


def name_in(name: str, text: str, lang: Lang) -> bool:
    """Whether the line contains the name. Russian and Ukrainian names change their endings
    (Маша, Машу, Машей; Олександр, Олександре), so there each word is matched by its stem."""
    if not lang.inflected:
        return name in text
    text = text.lower()
    return all(w[: max(3, len(w) - 2)] in text for w in name.lower().split())


# ------------------------------------------------------------------ dialogue evidence
# Japanese: self-pronouns and sentence endings. Korean: kinship address terms depend on the
# speaker's gender (a girl says 오빠/언니, a boy 형/누나). Russian and Ukrainian: the past tense
# and short adjectives agree with я ("I"): я сказал/сказала, я рад/рада. Spoken Chinese has no
# reliable marker.
# "I", then only filler words (not, would, already, him, you...) before the verb or adjective: a
# noun there (я люблю футбол, я боюся котів) would look like a masculine past tense
_RU_I = (r"\bя\s+(?:(?:не|ни|бы|б|уже|ведь|же|ж|тоже|так|только|просто|всё|все|ещё|еще|его|её|ее|их|тебя|"
         r"вас|ему|ей|им|тебе|вам|это|там|тут|сразу|даже|вообще|никогда|давно|сегодня|вчера)\s+){0,3}")
_UK_I = (r"\bя\s+(?:(?:не|ні|б|би|вже|ж|же|теж|також|так|тільки|просто|все|ще|його|її|їх|тебе|вас|йому|їй|"
         r"їм|тобі|вам|це|там|тут|відразу|навіть|взагалі|ніколи|давно|сьогодні|вчора)\s+){0,3}")


def _cue(prefix: str, words: str):
    return re.compile(prefix + r"(?:" + words + r")\b", re.I), 1


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
    "ru": {
        "male": [_cue(_RU_I, r"\w+[аеёиоуыя]лс?я?|мог|смог|помог"),
                 _cue(_RU_I, r"сам|рад|готов|должен|уверен|согласен|один|прав|виноват|занят")],
        "female": [_cue(_RU_I, r"\w+[аеёиоуыя]лас?ь?"),
                   _cue(_RU_I, r"сама|рада|готова|должна|уверена|согласна|одна|права|виновата|занята")],
    },
    "uk": {
        "male": [_cue(_UK_I, r"\w+[аеиіоуя]вс?я?|міг|зміг|допоміг"),
                 _cue(_UK_I, r"сам|радий|готовий|винен|певен|згоден|один|правий|винуватий|зайнятий")],
        "female": [_cue(_UK_I, r"\w+[аеиіоуя]лас?я?"),
                   _cue(_UK_I, r"сама|рада|готова|винна|певна|згодна|одна|права|винувата|зайнята")],
    },
}

# a speaker naming themselves
INTRO = {
    "ja": re.compile(r"申します|と言います|といいます|です"),
    "zh": re.compile(r"我叫|我是|我的名字|名叫|在下|本人"),
    "ko": re.compile(r"저는|나는|제 이름|내 이름|입니다|이에요|예요|라고 합니다|라고 해"),
    "ru": re.compile(r"меня зовут|мо[её] имя|зовите меня|\bя\s*[—–-]\s", re.I),
    "uk": re.compile(r"мене зв(?:ати|уть)|моє ім['’ʼ]?я|кличте мене|\bя\s*[—–-]\s", re.I),
}


def gender_score(text: str, code: str) -> tuple[int, int]:
    cues = _GENDER_CUES.get(code, _GENDER_CUES["zh"])
    male = sum(w * len(rx.findall(text)) for rx, w in cues["male"])
    female = sum(w * len(rx.findall(text)) for rx, w in cues["female"])
    return male, female
