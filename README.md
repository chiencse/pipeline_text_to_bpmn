# LangGraph BPMN Generator

This project builds BPMN flows from natural language using LangGraph. It includes two scenarios:

- Scenario A: LLM NER + Hybrid retrieval + LLM BPMN with mapping
- Scenario B: HF NER + rule-based relations + LLM BPMN + post-retrieval mapping

It also provides detailed state logging and pretty console visualization to help you debug each graph step.

## Requirements

- Python 3.10–3.12 recommended
- Windows PowerShell (this README uses PowerShell commands)
- A Google Generative AI API key (Gemini) for `google-generativeai`

Install dependencies from `requirements.txt`:

```powershell
# (Optional) create virtual environment
python -m venv venv
./venv/Scripts/activate

# Install packages
pip install -r requirements.txt
```

## Configure API keys

Edit `app/llm.py` and set your Gemini API key:

```python
genai.configure(api_key="<YOUR_GEMINI_API_KEY>")
```

Alternatively, export it as an environment variable and update the code to read from env.

## Build the Vector DB (Chroma)

The hybrid retriever uses a Chroma vector store populated from activity templates.

Option A: Index from provided JSON (`tools/activity_tpl_docs.json`)

```powershell
cd tools
python index_activity.py
cd ..
```

Option B: Parse `activity.ts` then index (if you maintain templates in TS)

```powershell
cd tools
python python_only_build_and_index.py
cd ..
```

This will create a local Chroma DB in `chroma_activity/` used by `app/retrieval.py`.

## Run the FastAPI service

You can call compiled graphs programmatically via the FastAPI endpoints defined in `app/main.py`.

```powershell
# From repo root
uvicorn app.main:app --reload --port 8000
```

Endpoints:

- POST `/pipeline/a` — run Scenario A
- POST `/pipeline/b` — run Scenario B

Example request body:

```json
{
  "text": "Send email with Gmail after creating a Google Sheet",
  "options": {}
}
```

## Pretty run (local script mode)

`app/main.py` also contains a `__main__` block that streams the graph and prints rich logs per node.

```powershell
python -m app.main
```

You’ll see per-node panels with keys and timings, plus a summary table.

## Dev experience with LangGraph Dev Server

This repo includes a `langgraph.json` config so you can run the LangGraph Dev server and interact with graphs.

```json
{
  "entry": "app/main.py",
  "venv": "venv",
  "description": "LangGraph workflow config. Adjust as needed.",
  "dependencies": ["langgraph", "fastapi", "pydantic", "rich", "scikit-learn"],
  "graphs": {
    "scenario_a": "app.scenario_a:compiled_graph_a",
    "scenario_b": "app.scenario_b:compiled_graph_b"
  }
}
```

Start the dev server:

```powershell
langgraph dev
```

- If you see an error about missing `langgraph.json`, ensure the file exists in repo root.
- If you see an error "Invalid path '<file>' for graph", the value must be in `module.path:variable_name` format; this repo exposes `compiled_graph_a` and `compiled_graph_b`.
- If you see an error about missing dependencies, ensure `dependencies` in `langgraph.json` includes what you need, or install via pip.

## State logging & visualization

We added hooks to track state after each graph run:

- `app/state.py:log_state(state, step)` — dumps JSON snapshots to `app/viz/logs/`
- `app/tools/run_with_pretty_log.py` — pretty event panels (`print_event`, `print_summary`)
- `app/tools/viz_graphs.py` — utilities to render/preview graphs; writes Mermaid/PNG if supported

When calling the API endpoints, `main.py` logs and visualizes the final state for the pipeline call. The `__main__` block shows streaming logs while running locally.

## Troubleshooting

- Chroma errors (missing embeddings or DB path): Ensure you ran the indexing step and that `app/retrieval.py` points to `./chroma_activity`.
- Google Generative AI auth: Verify your API key is valid and not rate-limited. Consider moving the key to an environment variable.
- Windows PowerShell execution policy: If activating venv fails, try running PowerShell as Administrator or adjust the execution policy.
- GPU vs CPU for embeddings: The `HuggingFaceEmbeddings` model `intfloat/multilingual-e5-base` works on CPU; for better performance you can use a GPU-enabled environment.

## Project structure (key files)

- `app/scenario_a.py` — Scenario A graph, exported as `compiled_graph_a`
- `app/scenario_b.py` — Scenario B graph, exported as `compiled_graph_b`
- `app/main.py` — FastAPI app and local runner with pretty logs
- `app/retrieval.py` — Hybrid retriever (BM25 + embeddings + rules)
- `tools/index_activity.py` — Index templates into Chroma from JSON
- `tools/python_only_build_and_index.py` — Parse TS + index into Chroma
- `langgraph.json` — Config for `langgraph dev`

## Try it quickly

1. Install and index

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
cd tools; python index_activity.py; cd ..
```

2. Run Scenario A via API

```powershell
uvicorn app.main:app --reload --port 8000
```

POST http://localhost:8000/pipeline/a with JSON body:

```json
{ "text": "Send email with Gmail after creating a sheet." }
```

3. Run LangGraph Dev

```powershell
langgraph dev
```

4. Local pretty stream

```powershell
python -m app.main
```

Happy building! If you want deeper visualizations (Mermaid/Graphviz, state diffs per node, or timeline charts), we can wire those in next.
