"""Agent configuration — API keys, model settings, paths."""

import os
from pathlib import Path

# Project root (CAT411_framework/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output" / "agent"


def ensure_dirs():
    """Create output directories (call at startup, not import time)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LLM provider: "openai" or "anthropic"
LLM_PROVIDER = os.environ.get("CAT411_LLM_PROVIDER", "openai")

# API keys (from environment)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# Model defaults
OPENAI_MODEL = os.environ.get("CAT411_OPENAI_MODEL", "gpt-4o")
ANTHROPIC_MODEL = os.environ.get("CAT411_ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

# Agent behavior
MAX_TOOL_ROUNDS = 10  # max consecutive tool-use rounds per user message
TEMPERATURE = 0.1
