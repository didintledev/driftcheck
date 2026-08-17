"""Provider registry.

A provider is one module in this package exposing:

    ENV_KEY          name of the environment variable holding the API key
    DEFAULT_MODEL    model used when --model is not given
    RPM              requests per minute the batch runner paces to (None = no cap)
    PRICING          {model: (input $/MTok, output $/MTok)}
    FREE_NOTE        one line about free-tier availability, or None
    count_input_tokens(model, prompt) -> int
    generate(model, prompt, max_tokens, effort, temperature) -> str

Adding Groq, or anything else, is a new file here plus one line in `_MODULES`.
Nothing outside this package knows which provider is in use.
"""

import importlib
import os
from pathlib import Path

__all__ = ["DEFAULT", "get", "names", "require_key"]

DEFAULT = "gemini"

# ponytail: a dict of module names, not a plugin system with entry points.
_MODULES = {"gemini": ".gemini", "anthropic": ".anthropic"}


def names() -> list[str]:
    return list(_MODULES)


def get(name: str):
    """Import a provider module by name. Raises SystemExit on an unknown name."""
    if name not in _MODULES:
        raise SystemExit(f"unknown provider {name!r}; choose from {', '.join(_MODULES)}")
    # ponytail: imported on demand, so the default path never needs the
    # anthropic SDK installed and vice versa.
    return importlib.import_module(_MODULES[name], __name__)


def _load_dotenv() -> None:
    """Read KEY=value lines from ./.env into the environment, if present.

    ponytail: six lines of stdlib instead of a python-dotenv dependency.
    Real environment variables always win, so `.env` never shadows a real key.
    """
    path = Path.cwd() / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def require_key(env_key: str) -> str:
    """Return the API key from the environment (or ./.env), or exit saying which."""
    _load_dotenv()
    value = os.environ.get(env_key)
    if not value:
        raise SystemExit(
            f"{env_key} is not set. Copy .env.example to .env and add your key."
        )
    return value
