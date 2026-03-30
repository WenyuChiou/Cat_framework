"""
LLM client — handles API calls with tool-use loop.

Supports OpenAI and Anthropic backends. Automatically processes
tool calls in a loop until the model produces a final text response
or hits the max rounds limit.
"""

import json
from typing import Generator

from agent.config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    OPENAI_MODEL,
    ANTHROPIC_MODEL,
    MAX_TOOL_ROUNDS,
    TEMPERATURE,
)
from agent.tools.registry import (
    all_tool_schemas,
    all_tool_schemas_anthropic,
    call_tool,
)
from agent.system_prompt import SYSTEM_PROMPT


class ToolCallEvent:
    """Represents a tool call for display purposes."""
    def __init__(self, name: str, arguments: dict, result: str):
        self.name = name
        self.arguments = arguments
        self.result = result


def _validate_api_key(provider: str) -> None:
    """Raise early if the API key for the chosen provider is missing."""
    if provider == "openai" and not OPENAI_API_KEY:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Export it: export OPENAI_API_KEY='sk-...'"
        )
    if provider == "anthropic" and not ANTHROPIC_API_KEY:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY is not set. "
            "Export it: export ANTHROPIC_API_KEY='sk-ant-...'"
        )


def chat_openai(messages: list[dict]) -> Generator[str | ToolCallEvent, None, None]:
    """Run a chat completion with OpenAI, handling tool calls in a loop."""
    _validate_api_key("openai")
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)
    tools = all_tool_schemas()

    # Work on a copy so we don't pollute caller's message history
    # with internal tool-round messages
    working = list(messages)

    for _round in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=working,
            tools=tools if tools else None,
            temperature=TEMPERATURE,
            timeout=120,
        )

        choice = response.choices[0]
        msg = choice.message

        # If no tool calls, return the text
        if not msg.tool_calls:
            yield msg.content or ""
            return

        # Process tool calls
        working.append(msg.model_dump())

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

            event = ToolCallEvent(fn_name, fn_args, result_str)
            yield event

            working.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    # If we exhausted rounds, get a final response without tools
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=working,
        temperature=TEMPERATURE,
        timeout=120,
    )
    yield response.choices[0].message.content or ""


def chat_anthropic(messages: list[dict]) -> Generator[str | ToolCallEvent, None, None]:
    """Run a chat completion with Anthropic, handling tool calls in a loop."""
    _validate_api_key("anthropic")
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    tools = all_tool_schemas_anthropic()

    # Convert caller's messages to Anthropic format (user/assistant only)
    anthropic_messages = []
    for m in messages:
        if m["role"] in ("user", "assistant"):
            anthropic_messages.append({
                "role": m["role"],
                "content": m["content"],
            })

    for _round in range(MAX_TOOL_ROUNDS):
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=anthropic_messages,
            tools=tools if tools else None,
            temperature=TEMPERATURE,
        )

        # Check for tool use
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        text_blocks = [b for b in response.content if b.type == "text"]

        if not tool_use_blocks:
            # Pure text response
            yield "".join(b.text for b in text_blocks)
            return

        # Add assistant message with all content blocks
        anthropic_messages.append({
            "role": "assistant",
            "content": [b.model_dump() for b in response.content],
        })

        # Process each tool call
        tool_results = []
        for tb in tool_use_blocks:
            fn_name = tb.name
            fn_args = tb.input if isinstance(tb.input, dict) else {}

            result_str = call_tool(fn_name, fn_args)
            event = ToolCallEvent(fn_name, fn_args, result_str)
            yield event

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": result_str,
            })

        anthropic_messages.append({
            "role": "user",
            "content": tool_results,
        })

    # Exhausted rounds — get final text
    response = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=anthropic_messages,
        temperature=TEMPERATURE,
    )
    yield "".join(b.text for b in response.content if b.type == "text")


def chat(messages: list[dict]) -> Generator[str | ToolCallEvent, None, None]:
    """Route to the configured LLM provider."""
    if LLM_PROVIDER == "anthropic":
        yield from chat_anthropic(messages)
    else:
        yield from chat_openai(messages)
