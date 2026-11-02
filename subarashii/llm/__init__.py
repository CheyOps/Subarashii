"""LLM backends. All expose json(system, user, schema) -> dict and release() to free GPU memory."""

from ..config import Cfg
from .base import LLM, LLMError


def get_llm(cfg: Cfg) -> LLM:
    provider = cfg.llm.provider
    if provider == "llamacpp":
        from .openai_compat import OpenAICompatLLM

        c = cfg.llm.llamacpp
        return OpenAICompatLLM(
            url=c.url,
            api_key=c.api_key,
            model=c.model,
            temperature=cfg.llm.temperature,
            extra_body=dict(c.get("extra_body") or {}),
            router=True,
            unload_all=bool(c.get("unload_all_before_gpu", True)),
        )
    if provider == "openai":
        from .openai_compat import OpenAICompatLLM

        c = cfg.llm.openai
        if not c.url or not c.model:
            raise LLMError("llm.openai.url and llm.openai.model must be set")
        return OpenAICompatLLM(url=c.url, api_key=c.api_key, model=c.model, temperature=cfg.llm.temperature)
    if provider == "anthropic":
        from .anthropic_llm import AnthropicLLM

        c = cfg.llm.anthropic
        return AnthropicLLM(model=c.model, effort=c.get("effort", "medium"), api_key=c.get("api_key") or None)
    raise LLMError(f"unknown llm.provider: {provider}")


__all__ = ["LLM", "LLMError", "get_llm"]
