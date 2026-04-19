"""Agent configuration — API keys, model settings, paths."""

import os
import sys
from pathlib import Path

# Agent root (directory containing agent/ package)
AGENT_ROOT = Path(__file__).resolve().parent.parent

# ── Framework discovery (4-level fallback) ─────────────────────────────────
_env_root = os.environ.get("CAT411_FRAMEWORK_ROOT")
if _env_root and Path(_env_root).exists():
    FRAMEWORK_ROOT = Path(_env_root)
elif (AGENT_ROOT / "src").exists() and (AGENT_ROOT / "data").exists():
    # Integrated layout: agent is inside the framework itself
    FRAMEWORK_ROOT = AGENT_ROOT
elif (AGENT_ROOT / "CAT411_framework").exists():
    # Self-contained repo layout: agent lives alongside CAT411_framework/
    FRAMEWORK_ROOT = AGENT_ROOT / "CAT411_framework"
elif (AGENT_ROOT.parent / "CAT411_framework").exists():
    # Google Drive layout: sibling directory
    FRAMEWORK_ROOT = AGENT_ROOT.parent / "CAT411_framework"
else:
    raise RuntimeError(
        "CAT411_framework not found. Either:\n"
        "  1. Place agent/ inside CAT411_framework/ directly\n"
        "  2. Place CAT411_framework/ in the same directory as agent/\n"
        "  3. Set CAT411_FRAMEWORK_ROOT environment variable\n"
    )

DATA_DIR = FRAMEWORK_ROOT / "data"
OUTPUT_DIR = AGENT_ROOT / "output"

# Add framework to sys.path so `from src.xxx import ...` works
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))


def ensure_dirs():
    """Create output directories (call at startup, not import time)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Secret/key helper ─────────────────────────────────────────────────────

def _get_secret(key: str, default: str = "") -> str:
    """Get secret from env var, .env file, or Streamlit secrets."""
    val = os.environ.get(key, "")
    if not val:
        try:
            import streamlit as st
            val = st.secrets.get(key, "")
        except Exception:
            pass
    return val or default


# ── LLM provider ──────────────────────────────────────────────────────────
# Supported: "openai", "anthropic", "ollama", "nvidia"

LLM_PROVIDER = _get_secret("CAT411_LLM_PROVIDER", "openai")

# API keys
OPENAI_API_KEY = _get_secret("OPENAI_API_KEY")
ANTHROPIC_API_KEY = _get_secret("ANTHROPIC_API_KEY")
NVIDIA_API_KEY = _get_secret("NVIDIA_API_KEY")

# Model defaults
OPENAI_MODEL = _get_secret("CAT411_OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = _get_secret("CAT411_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Ollama settings (local LLM, free)
OLLAMA_BASE_URL = _get_secret("CAT411_OLLAMA_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = _get_secret("CAT411_OLLAMA_MODEL", "mistral-small:24b")

# NVIDIA NIM settings (free tier, cloud-hosted)
NVIDIA_BASE_URL = _get_secret("CAT411_NVIDIA_URL", "https://integrate.api.nvidia.com/v1")
NVIDIA_MODEL = _get_secret("CAT411_NVIDIA_MODEL", "nvidia/llama-3.3-nemotron-super-49b-v1.5")

# Agent behavior
MAX_TOOL_ROUNDS = 10
TEMPERATURE = 0.1
