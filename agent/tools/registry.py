"""
Tool registry — central place to register, look up, and export tool schemas.

Each tool is a dict with:
  - "name": str
  - "description": str
  - "parameters": JSON Schema dict
  - "function": callable(**kwargs) -> str   (returns text for LLM)
"""

from typing import Callable

_TOOLS: dict[str, dict] = {}


def register_tool(
    name: str,
    description: str,
    parameters: dict,
    function: Callable,
) -> None:
    """Register a tool for the agent."""
    _TOOLS[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "function": function,
    }


def get_tool(name: str) -> dict | None:
    """Look up a registered tool by name."""
    return _TOOLS.get(name)


def call_tool(name: str, arguments: dict) -> str:
    """Execute a tool and return its string result."""
    tool = _TOOLS.get(name)
    if tool is None:
        return f"Error: unknown tool '{name}'"
    try:
        return tool["function"](**arguments)
    except Exception as e:
        return f"Error executing {name}: {type(e).__name__}: {e}"


def all_tool_schemas() -> list[dict]:
    """Return OpenAI-compatible function schemas for all registered tools."""
    schemas = []
    for t in _TOOLS.values():
        schemas.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        })
    return schemas


def all_tool_schemas_anthropic() -> list[dict]:
    """Return Anthropic-compatible tool schemas for all registered tools."""
    schemas = []
    for t in _TOOLS.values():
        schemas.append({
            "name": t["name"],
            "description": t["description"],
            "input_schema": t["parameters"],
        })
    return schemas
