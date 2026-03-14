# Experiment Value Predictor (EVP)

EVP is an **AI Research Decision Engine** that helps teams choose **which experiment to run first** by balancing expected impact against resource cost.

Instead of just automating experiments, EVP ranks candidates and returns a recommendation you can act on.

## What You Can Do

- Generate and rank experiment ideas from a topic + goal.
- Estimate compute cost and expected impact with agent heuristics or LLMs.
- Run a Streamlit lab for PDF audit, knowledge bridging, and extracted data.
- Use mock or static modes for fast demos and tests.

## Core Workflow

`literature -> hypotheses -> resource estimate + impact estimate -> value scoring -> recommendation`

## Quickstart

```bash
git clone <your-repo-url>
cd nexus
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### CrewAI note

`crewai` installs only on Python `<3.14` (per `requirements.txt`). If you are on Python 3.14+, it will be skipped. Use Python 3.11 or 3.12 if you need CrewAI specifically.

## Run the Streamlit Lab

```bash
streamlit run app.py
```

The UI includes:

- Paper Audit Lab (PDF upload, extraction, audit report)
- Knowledge Bridge graph
- Syntropy cross-domain bridge
- Extracted JSON views

## Run the Pipeline (CLI)

### Mock mode (fast, no external calls)

```bash
export EVP_LLM_MODE=mock
python - <<'PY'
import asyncio
from evp.orchestration.pipeline import run_pipeline

result = asyncio.run(run_pipeline(topic="Brain MRI", goal="Improve classification accuracy"))
print(result["recommended_experiment_id"])
for e in result["experiments"]:
    print(e["id"], e["compute_units"], e["novelty_score"], round(e["value"], 2))
PY
```

### Static mock mode (deterministic JSON for UI/demo snapshots)

```bash
export EVP_LLM_MODE=mock_static
python - <<'PY'
import asyncio
from evp.orchestration.pipeline import run_pipeline

result = asyncio.run(run_pipeline(topic="Brain MRI", goal="Improve classification accuracy"))
print(result["recommended_experiment_id"])
for e in result["experiments"]:
    print(e["id"], e["compute_units"], e["novelty_score"], round(e["value"], 2))
PY
```

### Local ACPX CLI mode (Gemini/Qwen via ACPX)

```bash
export EVP_LLM_MODE=local
export EVP_LOCAL_MODEL=gemini
export EVP_ACPX_CMD='acpx run --model {model}'
export EVP_ACPX_TIMEOUT_SECONDS=120
python - <<'PY'
import asyncio
from evp.orchestration.pipeline import run_pipeline

result = asyncio.run(run_pipeline(topic="Brain MRI", goal="Improve classification accuracy"))
print(result["recommended_experiment_id"])
PY
```

### OpenAI mode (LangChain)

```bash
export EVP_LLM_MODE=openai
export EVP_OPENAI_MODEL=gpt-4o
export OPENAI_API_KEY=...
python - <<'PY'
import asyncio
from evp.orchestration.pipeline import run_pipeline

result = asyncio.run(run_pipeline(topic="Brain MRI", goal="Improve classification accuracy"))
print(result["recommended_experiment_id"])
PY
```

## Syntropy (Cross-Domain Research Bridge)

Syntropy builds a concept graph across two domains and surfaces candidate bridges.
In the UI, open the **Syntropy** section and provide two files/datasets.

### Syntropy Environment Controls

- `SYNTROPY_LLM_MODE=mock|openai` (default `openai`)
- `SYNTROPY_MODEL=gpt-4o`
- `OPENAI_API_KEY=...` (required for `openai`)
- `SYNTROPY_EMBEDDINGS_MODEL=all-MiniLM-L6-v2`
- `SYNTROPY_LOCAL_PAPERS_DIR=data/papers`
- `SYNTROPY_USE_PUBMED=true|false`
- `SYNTROPY_USE_SCHOLAR=true|false`
- `SYNTROPY_PUBMED_EMAIL` / `NCBI_EMAIL`
- `SYNTROPY_PUBMED_API_KEY` / `NCBI_API_KEY`
- `SERPER_API_KEY` (for Google Scholar via Serper)

## Data Sources

EVP and Syntropy can draw from:

- Local papers (`data/papers`) - takes priority when present
- arXiv (default for EVP)
- PubMed (optional for Syntropy)
- Google Scholar via Serper (optional for Syntropy)

### Local Papers Format

```text
Title: My Paper Title
Authors: Jane Doe, John Smith

Main abstract or summary text goes here.
```

If you omit `Title:` and `Authors:`, the filename becomes the title and the full file body becomes the abstract.

To customize the folder:

```bash
export EVP_LOCAL_PAPERS_DIR=data/papers
```

## Pipeline Contract

`run_pipeline(topic, goal, constraints=None)` returns JSON with:

- `topic`, `goal`
- `literature`
- `papers`
- `hypotheses`
- `experiments` (ranked by `value`)
- `recommended_experiment_id`
- `memory`

## Project Structure

- `app.py`: Streamlit UI
- `evp/agents/`: agent definitions
- `evp/orchestration/pipeline.py`: async orchestration
- `evp/scoring/scoring.py`: value scoring
- `evp/lab/`: paper audit + knowledge bridge
- `evp/syntropy/`: cross-domain bridge
- `evp/data/`: arXiv, PubMed, Serper, and local loaders
- `evp/utils/`: LLM clients, schemas, heuristics, logging
- `tests/`: unit + integration tests

## Testing

```bash
pytest -q
```

If `pytest` is not found:

```bash
python -m pytest -q
```

## Architecture (Text)

1. User inputs topic + goal (+ optional constraints).
2. Pipeline fetches paper context (local, mock, or arXiv).
3. LiteratureAgent summarizes and writes memory.
4. HypothesisAgent generates experiment candidates.
5. ResourceEstimator and ImpactPredictor run in parallel for each experiment.
6. Scoring engine computes value and ranks experiments.
7. Frontend consumes ranked JSON and highlights recommendation.
