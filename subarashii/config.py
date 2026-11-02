"""Layered configuration: built-in defaults < /config/config.yaml < env vars < web UI settings < per-job overrides."""

import copy
import json
import os
from pathlib import Path

import yaml

DEFAULTS = Path(__file__).with_name("default_config.yaml")
USER_CONFIG = Path(os.environ.get("GS_CONFIG", "/config/config.yaml"))

# env var -> dotted config key
ENV_MAP = {
    "GS_ASSETS": "paths.assets",
    "GS_DATA": "paths.data",
    "GS_TTS_URL": "tts.url",
    "GS_LLM_PROVIDER": "llm.provider",
    "GS_LLAMACPP_URL": "llm.llamacpp.url",
    "GS_LLAMACPP_API_KEY": "llm.llamacpp.api_key",
    "GS_LLAMACPP_MODEL": "llm.llamacpp.model",
    "GS_OPENAI_URL": "llm.openai.url",
    "GS_OPENAI_API_KEY": "llm.openai.api_key",
    "GS_OPENAI_MODEL": "llm.openai.model",
    "GS_ANTHROPIC_MODEL": "llm.anthropic.model",
    "ANTHROPIC_API_KEY": "llm.anthropic.api_key",
}

# translator settings editable from the web UI (provider -> fields); saved in <data>/settings.json
LLM_FIELDS = {
    "llamacpp": ("url", "api_key", "model"),
    "openai": ("url", "api_key", "model"),
    "anthropic": ("api_key", "model", "effort"),
}
SECRET_FIELDS = {"api_key"}


class Cfg(dict):
    """dict with attribute access for nested keys (cfg.timing.max_overflow)."""

    def __getattr__(self, key):
        try:
            v = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(v) if isinstance(v, dict) and not isinstance(v, Cfg) else v

    def get_path(self, dotted: str, default=None):
        node = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_dotted(d: dict, dotted: str, value):
    parts = dotted.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def parse_value(raw: str):
    """CLI --set values: parse as YAML so numbers/bools/lists work."""
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def settings_path(cfg: dict) -> Path:
    return Path(cfg["paths"]["data"]) / "settings.json"


def load_settings(cfg: dict) -> dict:
    try:
        return json.loads(settings_path(cfg).read_text())
    except (OSError, ValueError):
        return {}


def save_settings(cfg: dict, settings: dict):
    path = settings_path(cfg)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # holds API keys
    with os.fdopen(fd, "w") as fh:
        json.dump(settings, fh, indent=2)
    tmp.replace(path)


def load_config(overrides: dict | None = None) -> Cfg:
    cfg = yaml.safe_load(DEFAULTS.read_text())
    if USER_CONFIG.is_file():
        cfg = deep_merge(cfg, yaml.safe_load(USER_CONFIG.read_text()) or {})
    for env, key in ENV_MAP.items():
        val = os.environ.get(env)
        if val:
            set_dotted(cfg, key, val)
    cfg = deep_merge(cfg, load_settings(cfg))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return Cfg(cfg)


def overrides_from_pairs(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        set_dotted(out, key.strip(), parse_value(raw))
    return out


def models_dir(cfg: Cfg) -> Path:
    p = Path(cfg.paths.assets) / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(cfg: Cfg, sub: str) -> Path:
    p = Path(cfg.paths.data) / sub
    p.mkdir(parents=True, exist_ok=True)
    return p
