"""Claude via the Anthropic API (optional; set an API key and llm.provider=anthropic)."""

import anthropic

from .base import LLM, LLMError, parse_json


class AnthropicLLM(LLM):
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", api_key: str | None = None):
        self.model = model
        self.effort = effort
        self.name = f"anthropic:{model}"
        self.client = anthropic.Anthropic(api_key=api_key)

    def json(self, system: str, user: str, schema: dict, max_tokens: int = 16000) -> dict:
        try:
            # Server-side fallback: if a safety classifier declines, the API retries on a fallback model.
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as e:
            raise LLMError(f"Anthropic rate limited: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic API error {e.status_code}: {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"cannot reach Anthropic API: {e}") from e
        if response.stop_reason == "refusal":
            raise LLMError("Claude declined this request")
        if response.stop_reason == "max_tokens":
            raise LLMError("Claude output truncated (max_tokens); lower translation.batch_size")
        text = next((b.text for b in response.content if b.type == "text"), "")
        return parse_json(text)
