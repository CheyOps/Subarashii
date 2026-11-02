import json
import re


class LLMError(RuntimeError):
    pass


class LLM:
    name = "llm"

    def json(self, system: str, user: str, schema: dict, max_tokens: int = 8000) -> dict:
        raise NotImplementedError

    def release(self):
        """Free any GPU memory the backend holds (no-op for remote APIs)."""


_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def parse_json(text: str) -> dict:
    text = _THINK_RE.sub("", text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        a, b = text.find("{"), text.rfind("}")
        if a != -1 and b > a:
            return json.loads(text[a : b + 1])
        raise LLMError(f"model did not return JSON: {text[:300]!r}")
