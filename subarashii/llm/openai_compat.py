"""OpenAI-compatible chat endpoint (llama.cpp server/router, vLLM, LM Studio, OpenRouter...)."""

import time

import httpx

from .base import LLM, LLMError, parse_json


class OpenAICompatLLM(LLM):
    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        temperature: float = 0.3,
        extra_body: dict | None = None,
        router: bool = False,
        unload_all: bool = False,
        timeout: float = 900,
    ):
        self.root = url.rstrip("/").removesuffix("/v1")
        self.model = model
        self.temperature = temperature
        self.extra_body = extra_body or {}
        self.router = router
        self.unload_all = unload_all
        self.name = f"{'llamacpp' if router else 'openai'}:{model}"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.Client(base_url=self.root, headers=headers, timeout=timeout)

    def _chat(self, body: dict) -> httpx.Response:
        for attempt in range(4):
            try:
                r = self.http.post("/v1/chat/completions", json=body)
            except httpx.TransportError as e:
                if attempt == 3:
                    raise LLMError(f"cannot reach LLM at {self.root}: {e}") from e
                time.sleep(3 * (attempt + 1))
                continue
            # router answers 503 while a model is still loading
            if r.status_code in (429, 502, 503) and attempt < 3:
                time.sleep(5 * (attempt + 1))
                continue
            return r
        return r

    def json(self, system: str, user: str, schema: dict, max_tokens: int = 8000) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature,
            "max_tokens": max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema, "strict": True},
            },
            **self.extra_body,
        }
        r = self._chat(body)
        for attempt in range(2):
            if not (self.router and r.status_code == 500 and "failed to load" in r.text):
                break
            # usually VRAM held by another router model; clear everything and try again
            self.unload_all, keep = True, self.unload_all
            self.release()
            self.unload_all = keep
            time.sleep(5 * (attempt + 1))
            r = self._chat(body)
        if r.status_code == 400 and "response_format" in r.text:
            body["response_format"] = {"type": "json_object"}
            r = self._chat(body)
        if r.status_code >= 400:
            raise LLMError(f"LLM HTTP {r.status_code}: {r.text[:500]}")
        choice = r.json()["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMError("LLM output truncated (max_tokens); lower translation.batch_size")
        return parse_json(choice["message"].get("content") or "")

    def release(self):
        if not self.router:
            return
        try:
            targets = [self.model]
            if self.unload_all:
                r = self.http.get("/models")
                if r.status_code == 200:
                    targets = [
                        m["id"]
                        for m in r.json().get("data", [])
                        if (m.get("status") or {}).get("value") in ("loaded", "loading")
                    ]
            for model in targets:
                self.http.post("/models/unload", json={"model": model})
        except httpx.HTTPError:
            pass
