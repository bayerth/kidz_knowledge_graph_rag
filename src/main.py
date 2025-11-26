"""
Application entry point for the GUI RAG app.

Responsibilities:
- Load environment (.env if present) and initialize LLM connectors/clients.
- Load the base ontology from data/ontology_base.json.
- Build a model dictionary keyed by menu labels used in the GUI.
- Start the Tkinter GUI (`App`).

Notes:
- Connectors are implemented in `src.connectors.llmconnector`.
- We initialize one representative model per provider to match the GUI menu items:
  - "gpt-4-o"  -> OpenAI
  - "grok"     -> xAI (Grok)
  - "gemini"   -> Google Gemini (placeholder until `GeminiClient` is implemented)
"""

from __future__ import annotations

import os
from pathlib import Path

# Underlying SDK clients
from openai import OpenAI  # OpenAI SDK (also used for xAI with base_url override)
from xai_sdk import Client as Client
from google import genai

# Optional: load environment variables from a .env file if present
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from src.gui.gui_rag import main_gui, Logger
from src.rag import ontology as ontology_mod
from src.connectors.llmconnector import OpenAIClient, XAIClient, GeminiClient


def load_env() -> None:
    """Load environment variables from .env if available."""
    if load_dotenv is not None:
        # Look for .env in project root
        project_root = Path(__file__).resolve().parents[1]
        dotenv_path = project_root / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)


def build_model_dict(logger: Logger) -> dict[str, object]:
    """Initialize model clients/connectors based on available environment variables.

    Returns a dictionary mapping menu labels to initialized client wrappers.
    Missing providers are skipped with a log message.
    """
    model_dict: dict[str, object] = {}
    openai_client = None
    xai_client = None
    gemini_client = None
    # OpenAI (gpt-4-o)
    openai_api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_KEY")
    if openai_api_key:
        try:
            # Organization is optional
            org = os.environ.get("OPENAI_ORG_KEY") or os.environ.get("OPENAI_ORGANIZATION")
            openai_client = OpenAI(api_key=openai_api_key, organization=org)
            model_dict["gpt-4o-2024-11-20"] = OpenAIClient(client=openai_client, model="gpt-4o-2024-11-20")
            model_dict["gpt-4o-mini"] = OpenAIClient(client=openai_client, model="gpt-4o-mini")
            model_dict["gpt-4-o"] = OpenAIClient(client=openai_client, model="gpt-4o")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
    else:
        logger.info("OPENAI_API_KEY not set — skipping OpenAI model initialization.")

    # xAI (Grok)
    xai_api_key = (
            os.environ.get("XAI_API_KEY")
            or os.environ.get("GROK_API_KEY")
            or os.environ.get("XAI_KEY")
    )
    if xai_api_key:
        try:
            # xAI uses OpenAI SDK with a custom base_url
            xai_client = Client(
                    api_key=xai_api_key,
                    timeout=60,  # Override default timeout with longer timeout for reasoning models
                    )
            # Choose a sensible default model name for Grok
            model_dict["grok-4"] = XAIClient(client=xai_client, model=os.environ.get("GROK_MODEL", "grok-4"))
            model_dict["grok-3-mini"] = XAIClient(client=xai_client, model=os.environ.get("GROK_MODEL", "grok-3-mini"))
        except Exception as e:
            logger.error(f"Failed to initialize xAI (Grok) client: {e}")
    else:
        logger.info("XAI_API_KEY/GROK_API_KEY not set — skipping Grok model initialization.")

    # Gemini placeholder — GeminiClient is not implemented in llmconnector yet
    # We still expose a menu entry to keep UI consistent; value is currently None.
    # If you implement `GeminiClient`, replace this section with actual initialization
    # using `google.generativeai` similar to notebooks/test_llmclient.ipynb.
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        gemini_client = genai.Client(api_key=gemini_api_key)
        model_dict["gemini-2.5-flash-lite"] = GeminiClient(client=gemini_client, model="gemini-2.5-flash-lite")
    else:
        # Still add Gemini to the menu for consistency; user can configure later
        model_dict.setdefault("gemini", None)

    return model_dict


def load_ontology(logger: Logger) -> ontology_mod.Ontology:
    """Load ontology from the data directory."""
    project_root = Path(__file__).resolve().parents[1]
    ontology_path = project_root / "data" / "ontology_base.json"
    owl = ontology_mod.Ontology()
    try:
        owl.deserialize(str(ontology_path))
    except Exception as e:
        logger.error(f"Failed to load ontology from {ontology_path}: {e}")
        # Continue with an empty ontology to allow the GUI to start
    return owl


from src.rag.retriever import read_records_by_unique_id


def main() -> None:
    logger = Logger()
    load_env()
    owl = load_ontology(logger)
    models = build_model_dict(logger)
    question_dict = read_records_by_unique_id("../data/question_evaluations_processed.json")
    question_list = [e['question'] for e in question_dict.values()]
    if not models:
        logger.info("No models initialized. You can still use the GUI; configure API keys in .env to enable models.")
    main_gui(ontology=owl, model_dict=models, question_list=question_list, logger=None)


if __name__ == "__main__":
    main()
