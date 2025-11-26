KI BASIERTER DIGITALER ZWILLING (KIDZ) Retrieval Augmneted Generation Prototype

https://forschung.rwu.de/forschungsprojekte/ki-basierter-digitaler-zwilling-kidz

A small Retrieval-Augmented Generation (RAG) prototype with a Tkinter GUI. It allows you to:

- Load a domain ontology (JSON) and ask questions.
- Select an LLM provider/model from a menu (OpenAI, xAI/Grok, Gemini placeholder).
- View outputs and logs directly in the GUI.

This repository also contains utilities for working with evaluation question files and for visualizing retrieved
ontology nodes as a graph (HTML) using PyVis.

Project layout

- src/main.py — Application entry point. Loads environment, ontology, initializes LLM clients, and starts the GUI.
- src/gui/gui_rag.py — Tkinter GUI application:
    - Model menu (radiobuttons) to pick the provider/model.
    - Dropdown with example questions loaded from data files.
    - Input text, one-line model info, Output area, and Logs area.
    - Buttons: Start, Clear output, Clear log_text.
    - Help menu with Documentation and About dialogs (placeholders).
- src/connectors/llmconnector.py — LLM client wrappers:
    - OpenAIClient (OpenAI SDK),
    - XAIClient (xAI/Grok using OpenAI-compatible API),
    - GeminiClient (uses google-genai; basic implementation; history handling is simplified).
- src/rag/ontology.py — Ontology model + serialization/deserialization and traversal helpers.
- src/rag/retriever.py — RAG pipeline utilities:
    - High-level steps to identify relevant classes and instance starting points.
    - Iterative traversal to collect relevant nodes.
    - Graph generation via PyVis.
    - Utility read_records_by_unique_id(...) to read evaluation question files (JSON/JSONL/TXT) keyed by unique_id.
- data/ — Ontology and evaluation inputs.
- notebooks/ — Experimental notebooks (dev/testing of clients and RAG behavior).

Requirements

Python 3.10+ is recommended.

Install Python dependencies (no version pins):

```
pip install -r requirements.txt
```

Required/optional packages:

- openai — OpenAI API client.
- xai-sdk — xAI (Grok) SDK (OpenAI-compatible chat interface).
- google-genai — Google Gemini API client (new SDK), used by GeminiClient.
- pyvis — HTML graph visualization for retrieved ontology nodes.
- python-dotenv — Optional, loads .env automatically from project root.
- pydantic — Used by GUI module for simple modeling.

Note: Tkinter must be available on your system. On some Linux distributions you may need to install tk/tk-dev packages.
On macOS and Windows, Tk typically ships with the Python installer from python.org.

Environment variables

Configure providers via environment variables or a .env file at the repository root:

OpenAI

- OPENAI_API_KEY (or OPENAI_KEY)
- Optional: OPENAI_ORG_KEY (or OPENAI_ORGANIZATION)

xAI / Grok

- One of: XAI_API_KEY, GROK_API_KEY, or XAI_KEY
- Optional: GROK_MODEL (overrides default model name used in the client wrappers)

Google Gemini

- GEMINI_API_KEY (if set, a Gemini client is created; basic flow wired; see notes below)

.env example (put this at project root):

```
OPENAI_API_KEY=sk-...
OPENAI_ORG_KEY=org_...
XAI_API_KEY=...
GEMINI_API_KEY=...
```

Running the app

From the project root:

```
python -m src.main
```

- Use the Model menu at the top to select an LLM.
- Use the dropdown to pick a sample question or type your own in the Input area.
- The one-line Model field reflects the current selection.
- Click Start to run the RAG query; see results in Output and events in Logs.
- Use Clear output / Clear log_text as needed.
- Help menu provides placeholder dialogs.

Data and evaluation utilities

- Ontology file: data/ontology_base.json is loaded at startup.
- Evaluation questions: files like data/question_evaluations.json and data/question_evaluations_processed.json.
- Utility: src/rag/retriever.py provides read_records_by_unique_id(path_or_stem) which:
    - Accepts a full path or a stem (e.g., data/question_evaluations).
    - Supports JSON array or JSON Lines (JSONL/TXT) formats.
    - Returns a dict keyed by unique_id with the remaining fields as values.

In src/main.py, we load processed questions and map them into the GUI dropdown.

How the RAG flow works (brief)

retriever.prepare_objects(...)

- Step 1: Ask the LLM to identify relevant ontology classes given the user query. Uses the ontology class-level
  structure as context.
- Step 2: Ask the LLM to select relevant starting instances from the chosen classes.

retriever.iterate_ontology(...)

- Step 3: Iteratively query and traverse related instances, accumulating a set of relevant nodes until a stop condition
  is met.
- Step 4: Returns collected node structures; can render a graph (PyVis) showing the discovered subgraph.

gui_rag.App

- Builds the UI, connects Start button to a high-level RAG function, and displays results.

Notes and limitations

- Gemini client: The included GeminiClient in llmconnector.py uses google-genai and a prompt template. History handling
  is simplified.
- Legacy code in retriever.py: There are older functions referencing gpt_request_with_history (commented or kept for
  reference). The GUI path uses higher-level helpers that rely on the client wrappers in llmconnector.py.
- API rates and costs: Keep an eye on token usage and rates when testing models.

Development

- Run formatting and linters per your preferences; follow the existing code style.
- Tests: basic tests under src/rag/tests. Extend as needed.
- Notebooks under notebooks/ are for experimentation and are not required to run the GUI.

License

Proprietary / internal project. Update this section if a specific license applies.