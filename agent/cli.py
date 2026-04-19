"""
CAT411 Agent CLI — interactive command-line interface.

Usage:
    python -m agent.cli

Environment variables:
    OPENAI_API_KEY       — OpenAI API key (default provider)
    ANTHROPIC_API_KEY    — Anthropic API key
    CAT411_LLM_PROVIDER  — "openai" (default) or "anthropic"
    CAT411_OPENAI_MODEL  — OpenAI model (default: gpt-4o)
    CAT411_ANTHROPIC_MODEL — Anthropic model (default: claude-sonnet-4-20250514)
"""

# Import tools to trigger registration (order doesn't matter)
import agent.tools.query_bridges      # noqa: F401
import agent.tools.get_fragility      # noqa: F401
import agent.tools.compute_loss       # noqa: F401
import agent.tools.run_scenario       # noqa: F401
import agent.tools.plot_results       # noqa: F401
import agent.tools.summarize_portfolio  # noqa: F401
import agent.tools.plot_map            # noqa: F401
import agent.tools.run_scenario_uncertainty  # noqa: F401
import agent.tools.export_report       # noqa: F401

from agent.system_prompt import SYSTEM_PROMPT
from agent.llm_client import chat, ToolCallEvent
from agent.config import LLM_PROVIDER, OPENAI_MODEL, ANTHROPIC_MODEL, OLLAMA_MODEL, ensure_dirs


def _print_banner():
    """Print the agent welcome banner."""
    model = {"anthropic": ANTHROPIC_MODEL, "ollama": OLLAMA_MODEL}.get(LLM_PROVIDER, OPENAI_MODEL)
    print()
    print("=" * 62)
    print("  CAT411 Earthquake Risk Agent")
    print("  Seismic bridge loss estimation powered by Hazus 6.1")
    print(f"  LLM: {LLM_PROVIDER} / {model}")
    print("=" * 62)
    print()
    print("  Available commands:")
    print("    Type your question in natural language")
    print("    'quit' or 'exit' to end the session")
    print("    'clear' to reset conversation history")
    print()
    print("  Example queries:")
    print("    > Show me fragility parameters for HWB5")
    print("    > Run a M6.7 earthquake scenario near LA")
    print("    > How many bridges are in the Northridge area?")
    print("    > Plot fragility curves for HWB3")
    print()


def _format_tool_call(event: ToolCallEvent) -> str:
    """Format a tool call event for display."""
    args_str = ", ".join(f"{k}={v!r}" for k, v in event.arguments.items())
    return f"  [{event.name}({args_str})]"


def main():
    """Run the interactive CLI loop."""
    ensure_dirs()
    _print_banner()

    # Conversation history
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break

        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("  [Conversation cleared]")
            continue

        messages.append({"role": "user", "content": user_input})

        print()
        try:
            final_text = ""
            for event in chat(messages):
                if isinstance(event, ToolCallEvent):
                    # Show tool call to user
                    try:
                        print(_format_tool_call(event))
                    except UnicodeEncodeError:
                        print(f"  [{event.name}(...)]")
                else:
                    # Final text response
                    final_text = event

            if final_text:
                print(f"\nAgent: {final_text}")
                messages.append({"role": "assistant", "content": final_text})

        except Exception as e:
            # Roll back the user message on failure to keep history balanced
            messages.pop()
            error_msg = f"Error: {type(e).__name__}: {e}"
            print(f"\n  {error_msg}")


if __name__ == "__main__":
    main()
