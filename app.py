"""
CAT411 Earthquake Risk Agent — Streamlit Chat Interface

Usage:
    cd "Wenyu Chiou"
    streamlit run app.py

Environment variables:
    OPENAI_API_KEY or ANTHROPIC_API_KEY
    CAT411_LLM_PROVIDER  — "openai" (default) or "anthropic"
"""

import sys
import os
import re
from pathlib import Path

# Load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Ensure this directory is on sys.path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import streamlit as st

# ── Page config (must be first st call) ────────────────────────────────────
st.set_page_config(
    page_title="CAT411 Earthquake Risk Agent",
    page_icon="🌉",
    layout="wide",
)

# ── Import agent components ────────────────────────────────────────────────
from agent.config import (
    LLM_PROVIDER, OPENAI_MODEL, ANTHROPIC_MODEL, OLLAMA_MODEL, NVIDIA_MODEL,
    OUTPUT_DIR, ensure_dirs,
)
from agent.system_prompt import SYSTEM_PROMPT

# Register tools
import agent.tools.query_bridges       # noqa: F401
import agent.tools.get_fragility       # noqa: F401
import agent.tools.compute_loss        # noqa: F401
import agent.tools.run_scenario        # noqa: F401
import agent.tools.plot_results        # noqa: F401
import agent.tools.summarize_portfolio # noqa: F401
import agent.tools.plot_map            # noqa: F401
import agent.tools.run_scenario_uncertainty  # noqa: F401
import agent.tools.export_report       # noqa: F401

from agent.llm_client import chat, ToolCallEvent

ensure_dirs()


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    providers = ["openai", "anthropic", "ollama", "nvidia"]
    default_idx = providers.index(LLM_PROVIDER) if LLM_PROVIDER in providers else 0
    provider = st.selectbox("LLM Provider", providers, index=default_idx)
    os.environ["CAT411_LLM_PROVIDER"] = provider

    if provider == "ollama":
        st.success("Ollama (local) — no API key needed")
    elif provider == "nvidia":
        st.info("NVIDIA NIM — free API at [build.nvidia.com](https://build.nvidia.com/)")
        api_key = st.text_input(
            "NVIDIA API Key",
            type="password",
            value=os.environ.get("NVIDIA_API_KEY", ""),
            help="Get a free key at https://build.nvidia.com/",
        )
        os.environ["NVIDIA_API_KEY"] = api_key
    else:
        api_key = st.text_input(
            f"{'OpenAI' if provider == 'openai' else 'Anthropic'} API Key",
            type="password",
            value=os.environ.get(
                "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY", ""
            ),
        )
        if provider == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
        else:
            os.environ["ANTHROPIC_API_KEY"] = api_key

    st.divider()
    st.markdown("### 📋 Available Tools (11)")
    st.markdown("""
    - **query_bridges** — Search bridge inventory
    - **run_scenario** — Earthquake scenario analysis
    - **run_scenario_with_uncertainty** — Scenario + confidence intervals
    - **compute_bridge_loss** — Single bridge loss
    - **get_fragility** — Fragility parameters
    - **plot_fragility_curves** — Fragility curve plots
    - **plot_damage_distribution** — Damage bar charts
    - **plot_class_comparison** — Cross-class comparison
    - **plot_bridge_map** — Interactive damage map
    - **summarize_portfolio** — Portfolio statistics
    - **export_report** — Word report generation
    """)

    st.divider()
    st.markdown("### 💡 Example Queries")
    example_queries = [
        "Show fragility parameters for HWB5 at Sa=0.5g",
        "Run a M6.7 earthquake near Northridge with uncertainty analysis",
        "Show me a damage map for M6.7 at Northridge with 100 bridges",
        "How many bridges are near downtown LA?",
        "Compare fragility of HWB3, HWB5, and HWB17",
        "Generate a risk assessment report for M6.7 Northridge scenario",
    ]
    for q in example_queries:
        if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
            st.session_state["pending_query"] = q

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.session_state["api_messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        st.rerun()


# ── Session state init ─────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []  # display messages

if "api_messages" not in st.session_state:
    st.session_state["api_messages"] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]


# ── Header ─────────────────────────────────────────────────────────────────

st.title("🌉 CAT411 Earthquake Risk Agent")
_model_name = {
    "openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL,
    "ollama": OLLAMA_MODEL, "nvidia": NVIDIA_MODEL,
}
st.caption(
    "Seismic bridge loss estimation powered by Hazus 6.1 · "
    f"LLM: {provider} / {_model_name.get(provider, 'unknown')}"
)


# ── Display chat history ───────────────────────────────────────────────────

for msg in st.session_state["messages"]:
    role = msg["role"]
    with st.chat_message(role):
        st.markdown(msg["content"])
        # Show images if any
        if "images" in msg:
            for img_path in msg["images"]:
                if Path(img_path).exists():
                    st.image(img_path, use_container_width=True)


# ── Helper: find generated files in output directory ──────────────────────

def find_new_files(extensions: tuple, since_time: float) -> list[str]:
    """Find files in OUTPUT_DIR modified after since_time."""
    results = []
    if OUTPUT_DIR.exists():
        for f in OUTPUT_DIR.iterdir():
            if f.suffix.lower() in extensions and f.stat().st_mtime >= since_time:
                results.append(str(f))
    return results


# ── Process user input ─────────────────────────────────────────────────────

def process_query(user_input: str):
    """Send user query to LLM agent and display results."""
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.session_state["api_messages"].append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Process with LLM
    with st.chat_message("assistant"):
        tool_container = st.container()
        collected_images = []

        try:
            import time
            query_start = time.time()

            final_text = ""

            for event in chat(st.session_state["api_messages"]):
                if isinstance(event, ToolCallEvent):
                    args_str = ", ".join(
                        f"{k}={v!r}" for k, v in event.arguments.items()
                    )
                    with tool_container.expander(
                        f"🔧 {event.name}({args_str})", expanded=False
                    ):
                        st.code(event.result, language="text")
                else:
                    final_text = event

            if final_text:
                st.markdown(final_text)

                # Find all files generated during this query
                new_images = find_new_files((".png", ".jpg"), query_start)
                new_maps = find_new_files((".html",), query_start)
                new_docs = find_new_files((".docx",), query_start)

                # Show generated plots inline
                for img_path in new_images:
                    st.image(img_path, use_container_width=True)

                # Show interactive maps
                for map_path in new_maps:
                    try:
                        with open(map_path, "r", encoding="utf-8") as f:
                            st.components.v1.html(f.read(), height=500, scrolling=True)
                    except Exception:
                        st.info(f"Map saved to: {map_path}")

                # Show download buttons for documents
                for doc_path in new_docs:
                    with open(doc_path, "rb") as f:
                        st.download_button(
                            label=f"📄 Download {Path(doc_path).name}",
                            data=f.read(),
                            file_name=Path(doc_path).name,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )

                # Save to history
                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": final_text,
                    "images": new_images,
                })
                st.session_state["api_messages"].append({
                    "role": "assistant",
                    "content": final_text,
                })
            else:
                st.warning("Agent returned no response.")

        except EnvironmentError as e:
            st.error(f"⚠️ {e}")
            # Roll back
            st.session_state["messages"].pop()
            st.session_state["api_messages"].pop()

        except Exception as e:
            st.error(f"Error: {type(e).__name__}: {e}")
            st.session_state["messages"].pop()
            st.session_state["api_messages"].pop()


# ── Chat input ─────────────────────────────────────────────────────────────

# Check for pending query from sidebar examples
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")
    process_query(query)

# Regular chat input
if user_input := st.chat_input("Ask about earthquake bridge risk..."):
    process_query(user_input)
