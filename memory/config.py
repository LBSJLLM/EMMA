"""
Memory pipeline configuration.

Values are read (in priority order):
  1. Real environment variables already set in the shell
  2. A .env file in this directory (memory/.env) — never commit this file

See .env.example for all available options.
"""
import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — supports KEY=value, # comments, and quoted values."""
    if not path.exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(Path(__file__).parent / ".env")

# ── Local vLLM server (Qwen3-VL-8B-Instruct, handles vision) ─────────────────
VLLM_BASE_URL: str = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY: str = os.environ.get("VLLM_API_KEY", "dummy")
# Model name as registered by vllm serve (use --served-model-name qwen-vl to set this)
LLM_MODEL: str = os.environ.get("LLM_MODEL", "qwen-vl")

# ── External OpenAI-compatible APIs ────────────────────────────────────────────
# Qwen-VL stays local; structured memory text calls use this separate model.
MEMORY_TEXT_MODEL: str = os.environ.get("MEMORY_TEXT_MODEL", "deepseek-v4-flash")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str = os.environ.get("OPENAI_BASE_URL", "")

# ── Local model paths ──────────────────────────────────────────────────────────
QWEN_CHECKPOINT: str = os.environ.get("QWEN_CHECKPOINT", "/root/models/Qwen3-VL-8B-Instruct")
WHISPER_MODEL_DIR: str = os.environ.get("WHISPER_MODEL_DIR", "/root/models/whisper-large-v3-turbo")

# ── Output ─────────────────────────────────────────────────────────────────────
# Default: <MMM_root>/results  (one level above this file's directory)
_MMM_ROOT: Path = Path(__file__).parent.parent
OUTPUT_ROOT: Path = Path(os.environ.get("OUTPUT_ROOT", str(_MMM_ROOT / "results")))
