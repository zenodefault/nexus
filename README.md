# Experiment Value Predictor (EVP)

EVP is an **AI Research Decision Engine** that helps teams decide **which experiment is worth running before spending compute**.

Instead of only automating experiment execution, EVP prioritizes experiments by expected impact vs. resource cost.

## Core Workflow

`literature -> hypotheses -> resource estimate + impact estimate -> value scoring -> recommendation`

## Architecture (Hackathon)

- `LiteratureAgent`: summarizes findings/limitations from paper context.
- `HypothesisAgent`: generates candidate experiments.
- `ResourceEstimatorAgent`: estimates compute units (`Low|Medium|High`) with heuristic fallback.
- `ImpactPredictorAgent`: predicts novelty and expected gain.
- `Scoring Engine`: ranks experiments by value and returns the best next experiment.

## Project Structure

- `evp/agents/`: agent definitions.
- `evp/data/arxiv.py`: arXiv ingestion and parsing.
- `evp/orchestration/pipeline.py`: async orchestration and end-to-end pipeline.
- `evp/scoring/scoring.py`: value score computation and ranking.
- `evp/utils/llm.py`: mock/local (ACPX CLI) LLM clients.
- `tests/`: scoring, pipeline, and integration tests.

## Setup and Installation

```bash
git clone <your-repo-url>
cd nexus
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### CrewAI note

`crewai` is not in base requirements because it needs Python `<3.14`.
If you need CrewAI, create a Python 3.11 or 3.12 virtualenv and install it separately:

```bash
python3.11 -m venv .venv311
source .venv311/bin/activate
pip install -U pip
pip install crewai
```

## Run the Project

### 1) Activate environment

```bash
cd nexus
source .venv/bin/activate
```

### 2) Launch the Laboratory UI (PDF Audit + Bridge)

```bash
streamlit run app.py
```

This app now implements:

- PDF upload ingestion (`.pdf`) using Streamlit uploader
- PDF-to-text conversion via PyMuPDF (`fitz`)
- Structured deconstruction with Pydantic schema:
  - `abstract_summary`
  - `methodology_description`
  - `results_metrics`
  - `conclusion`
- Ghost Inspector consistency report JSON:
  - `is_consistent`
  - `discrepancies`
  - `verdict`
- Knowledge bridge graph generation with NetworkX + sentence-transformers
- Graph visualization via `streamlit-agraph`
- Tabs:
  - `📊 Paper Audit`
  - `🕸️ Knowledge Bridge`
  - `🧩 Extracted Data`

### 3) Mock mode (fast, no credits)

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

### 4) Static mock mode (frontend/demo snapshots)

Returns hardcoded deterministic JSON and bypasses agent calls.

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

### 5) Local ACPX CLI mode (Gemini/Qwen via ACPX)

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

`LocalLLMClient` sends prompts via stdin to the configured ACPX command and parses JSON from CLI output.

## Syntropy (Cross-Domain Research Catalyst)

Syntropy discovers bridges between two scientific domains by building a concept graph
and finding shortest paths across it. Launch the Streamlit UI and open the **Syntropy**
tab.

```bash
streamlit run app.py
```

### Syntropy Environment Controls

- `SYNTROPY_LLM_MODE=mock|openai` (default `openai`)
- `SYNTROPY_MODEL=gpt-4o` (used when `openai` mode is active)
- `OPENAI_API_KEY=...` (required for `openai` mode)
- `SYNTROPY_EMBEDDINGS_MODEL=all-MiniLM-L6-v2`
- `SYNTROPY_LOCAL_PAPERS_DIR=data/papers`

## arXiv Integration

`evp/data/arxiv.py` exposes:

- `fetch_papers(query)`: retrieves papers via `arxiv` Python library.
- `extract_abstracts(papers)`: prepares bounded abstract snippets.
- `build_literature_digest(papers)`: compact context string for agent prompts.

Reliability features:

- retry loop for transient failures/rate-limit style errors,
- graceful fallback to empty list if arXiv is unavailable,
- text normalization before passing to agents.

## Local Papers (Optional)

If you want to use your own paper summaries instead of arXiv, drop `.txt` or `.md` files into `data/papers/`.
Local files take priority over arXiv.

Format options:

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
- `experiments` (ranked with `value`)
- `recommended_experiment_id`
- `memory`

This shape is intended for Streamlit/API consumers.

## Testing

```bash
pytest -q
```

If `pytest` is not found:

```bash
python -m pytest -q
```

## Architecture Diagram (Text)

1. User inputs topic + goal (+ optional budget constraints).
2. Pipeline fetches paper context (mock or arXiv).
3. LiteratureAgent summarizes and writes memory.
4. HypothesisAgent generates experiment candidates.
5. For each experiment, ResourceEstimator and ImpactPredictor run in parallel.
6. Scoring engine computes value and ranks experiments.
7. Frontend consumes ranked JSON and highlights recommendation.

## Technical Complexity (for Presentation)

- Multi-agent orchestration with shared memory context.
- Asynchronous enrichment across per-experiment agent calls.
- Hybrid estimation strategy (LLM + deterministic heuristics fallback).
- External research ingestion and preprocessing through arXiv.
- Deterministic mock/static modes for fast demo reliability.
- ACPX CLI integration path for local model/provider switching.
