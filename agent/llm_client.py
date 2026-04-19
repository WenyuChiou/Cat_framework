"""
LLM client — handles API calls with tool-use loop.

Supports OpenAI, Anthropic, and Ollama (local) backends. Automatically
processes tool calls in a loop until the model produces a final text
response or hits the max rounds limit.
"""

import json
import os
from typing import Generator

import agent.config as _cfg
from agent.tools.registry import (
    all_tool_schemas,
    all_tool_schemas_anthropic,
    call_tool,
)
from agent.system_prompt import SYSTEM_PROMPT


def _get_config():
    """Read LLM config at call time so sidebar changes take effect."""
    return {
        "provider": os.environ.get("CAT411_LLM_PROVIDER", _cfg.LLM_PROVIDER),
        "openai_key": os.environ.get("OPENAI_API_KEY", _cfg.OPENAI_API_KEY),
        "anthropic_key": os.environ.get("ANTHROPIC_API_KEY", _cfg.ANTHROPIC_API_KEY),
        "nvidia_key": os.environ.get("NVIDIA_API_KEY", _cfg.NVIDIA_API_KEY),
        "openai_model": os.environ.get("CAT411_OPENAI_MODEL", _cfg.OPENAI_MODEL),
        "anthropic_model": os.environ.get("CAT411_ANTHROPIC_MODEL", _cfg.ANTHROPIC_MODEL),
        "ollama_url": os.environ.get("CAT411_OLLAMA_URL", _cfg.OLLAMA_BASE_URL),
        "ollama_model": os.environ.get("CAT411_OLLAMA_MODEL", _cfg.OLLAMA_MODEL),
        "nvidia_url": os.environ.get("CAT411_NVIDIA_URL", _cfg.NVIDIA_BASE_URL),
        "nvidia_model": os.environ.get("CAT411_NVIDIA_MODEL", _cfg.NVIDIA_MODEL),
        "max_rounds": _cfg.MAX_TOOL_ROUNDS,
        "temperature": _cfg.TEMPERATURE,
    }


class ToolCallEvent:
    """Represents a tool call for display purposes."""
    def __init__(self, name: str, arguments: dict, result: str):
        self.name = name
        self.arguments = arguments
        self.result = result


def _openai_tool_loop(
    client, model: str, messages: list[dict], tools: list[dict],
    max_rounds: int, temperature: float,
) -> Generator[str | ToolCallEvent, None, None]:
    """Shared OpenAI-compatible tool-use loop (works for OpenAI and Ollama)."""
    working = list(messages)

    for _round in range(max_rounds):
        kwargs = dict(
            model=model,
            messages=working,
            temperature=temperature,
        )
        if tools:
            kwargs["tools"] = tools

        response = client.chat.completions.create(**kwargs)

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            yield msg.content or ""
            return

        # Build explicit assistant message to avoid SDK serialization issues
        working.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                result_str = f"Error: could not parse tool arguments: {exc}"
                yield ToolCallEvent(fn_name, {}, result_str)
                working.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
                continue

            result_str = call_tool(fn_name, fn_args)
            yield ToolCallEvent(fn_name, fn_args, result_str)

            working.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # Exhausted rounds — final response without tools
    response = client.chat.completions.create(
        model=model,
        messages=working,
        temperature=temperature,
    )
    yield response.choices[0].message.content or ""


def chat_openai(messages: list[dict], cfg: dict) -> Generator[str | ToolCallEvent, None, None]:
    """Run a chat completion with OpenAI."""
    if not cfg["openai_key"]:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    from openai import OpenAI

    client = OpenAI(api_key=cfg["openai_key"], timeout=120.0)
    tools = all_tool_schemas()
    yield from _openai_tool_loop(
        client, cfg["openai_model"], messages, tools,
        cfg["max_rounds"], cfg["temperature"],
    )


def chat_ollama(messages: list[dict], cfg: dict) -> Generator[str | ToolCallEvent, None, None]:
    """Run a chat completion with Ollama (local LLM)."""
    from openai import OpenAI

    client = OpenAI(base_url=cfg["ollama_url"], api_key="ollama")
    tools = all_tool_schemas()
    yield from _openai_tool_loop(
        client, cfg["ollama_model"], messages, tools,
        cfg["max_rounds"], cfg["temperature"],
    )


def chat_nvidia(messages: list[dict], cfg: dict) -> Generator[str | ToolCallEvent, None, None]:
    """Run a chat completion with NVIDIA NIM (free tier cloud)."""
    if not cfg["nvidia_key"]:
        raise EnvironmentError(
            "NVIDIA_API_KEY is not set. Get a free key at https://build.nvidia.com/"
        )
    from openai import OpenAI

    client = OpenAI(
        base_url=cfg["nvidia_url"],
        api_key=cfg["nvidia_key"],
        timeout=120.0,
    )
    tools = all_tool_schemas()
    yield from _openai_tool_loop(
        client, cfg["nvidia_model"], messages, tools,
        cfg["max_rounds"], cfg["temperature"],
    )


def chat_anthropic(messages: list[dict], cfg: dict) -> Generator[str | ToolCallEvent, None, None]:
    """Run a chat completion with Anthropic."""
    if not cfg["anthropic_key"]:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set.")
    import anthropic

    client = anthropic.Anthropic(api_key=cfg["anthropic_key"])
    tools = all_tool_schemas_anthropic()

    anthropic_messages = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            anthropic_messages.append({
                "role": m["role"],
                "content": m["content"],
            })

    for _round in range(cfg["max_rounds"]):
        response = client.messages.create(
            model=cfg["anthropic_model"],
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=anthropic_messages,
            tools=tools if tools else None,
            temperature=cfg["temperature"],
        )

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_use_blocks:
            yield "".join(b.text for b in text_blocks)
            return

        anthropic_messages.append({
            "role": "assistant",
            "content": [b.model_dump() for b in response.content],
        })

        tool_results = []
        for tb in tool_use_blocks:
            fn_name = tb.name
            fn_args = tb.input if isinstance(tb.input, dict) else {}

            result_str = call_tool(fn_name, fn_args)
            yield ToolCallEvent(fn_name, fn_args, result_str)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": result_str,
            })

        anthropic_messages.append({
            "role": "user",
            "content": tool_results,
        })

    response = client.messages.create(
        model=cfg["anthropic_model"],
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=anthropic_messages,
        temperature=cfg["temperature"],
    )
    yield "".join(b.text for b in response.content if b.type == "text")


def chat(messages: list[dict]) -> Generator[str | ToolCallEvent, None, None]:
    """Route to the configured LLM provider. Reads config at call time."""
    cfg = _get_config()
    provider = cfg["provider"]

    if provider == "anthropic":
        yield from chat_anthropic(messages, cfg)
    elif provider == "ollama":
        yield from chat_ollama(messages, cfg)
    elif provider == "nvidia":
        yield from chat_nvidia(messages, cfg)
    else:
        yield from chat_openai(messages, cfg)
