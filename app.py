import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

from evp.lab.paper_audit import (
    BridgeResult,
    build_knowledge_bridge,
    deconstruct_paper,
    extract_pdf_text,
    inspect_consistency,
)
from evp.orchestration.pipeline import run_pipeline
from evp.utils.llm import LocalLLMClient, MockLLMClient


st.set_page_config(page_title="Nexus", page_icon="EVP", layout="wide")

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@500;600;700&display=swap');

:root {
  --bg-1: #f7f1e6;
  --bg-2: #efe4d1;
  --bg-3: #e4d2b8;
  --glass: rgba(255, 255, 255, 0.62);
  --glass-2: rgba(255, 255, 255, 0.38);
  --ink: #2c2118;
  --muted: #6d5b4d;
  --line: rgba(120, 93, 66, 0.25);
  --accent-1: #b98b5e;
  --accent-2: #8a6a4b;
  --accent-3: #d4b48d;
}

html, body, [class*="stApp"] {
  font-family: "Inter", "Poppins", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(1.5px 1.5px at 12% 18%, rgba(158,126,95,0.18), transparent 60%),
    radial-gradient(1.8px 1.8px at 70% 22%, rgba(142,112,84,0.14), transparent 62%),
    radial-gradient(1.2px 1.2px at 85% 68%, rgba(161,129,98,0.16), transparent 60%),
    radial-gradient(1.4px 1.4px at 42% 82%, rgba(145,115,88,0.14), transparent 62%),
    radial-gradient(circle at 16% 14%, rgba(185, 139, 94, 0.18), transparent 34%),
    radial-gradient(circle at 86% 10%, rgba(164, 127, 93, 0.14), transparent 42%),
    radial-gradient(circle at 82% 88%, rgba(212, 180, 141, 0.18), transparent 40%),
    linear-gradient(145deg, var(--bg-1), var(--bg-2) 50%, var(--bg-3));
}

[data-testid="stHeader"] {
  background: transparent;
}

h1, h2, h3 {
  letter-spacing: 0.2px;
  font-family: "Poppins", "Inter", sans-serif;
}

.hero {
  border: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(255, 255, 255, 0.18), rgba(255, 255, 255, 0.08));
  border-radius: 24px;
  padding: 28px 26px;
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.38), inset 0 1px 0 rgba(255, 255, 255, 0.25);
  margin-bottom: 18px;
  backdrop-filter: blur(12px) saturate(140%);
  -webkit-backdrop-filter: blur(12px) saturate(140%);
  text-align: center;
}

.hero h1 {
  margin: 0 0 8px 0;
  font-size: 3rem;
  font-weight: 800;
}

.hero p {
  margin: 0;
  color: var(--muted);
  font-size: 1.04rem;
  font-family: "Inter", sans-serif;
}

.feature-card {
  border: 1px solid var(--line);
  background: linear-gradient(160deg, var(--glass), var(--glass-2));
  border-radius: 18px;
  padding: 18px;
  min-height: 175px;
  box-shadow: 0 18px 42px rgba(0, 0, 0, 0.32), inset 0 1px 0 rgba(255, 255, 255, 0.24);
  backdrop-filter: blur(14px) saturate(140%);
  -webkit-backdrop-filter: blur(14px) saturate(140%);
}

.feature-card h3 {
  margin-top: 0;
  margin-bottom: 8px;
}

.feature-card p {
  color: var(--muted);
  margin-bottom: 0;
}

.feature-card-gap {
  margin-top: 12px;
}

.section-shell {
  border: 1px solid var(--line);
  background: linear-gradient(160deg, var(--glass), var(--glass-2));
  border-radius: 18px;
  padding: 14px 16px 8px 16px;
  margin-bottom: 12px;
  box-shadow: 0 16px 38px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.18);
  backdrop-filter: blur(12px) saturate(130%);
  -webkit-backdrop-filter: blur(12px) saturate(130%);
}

.mono {
  font-family: "Inter", sans-serif;
  color: #7a634f;
  font-size: 0.82rem;
  text-align: center;
  margin-bottom: 14px;
}

div[data-testid="stMetric"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.08);
  padding: 8px 10px;
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
}

.stButton button {
  border-radius: 999px !important;
  border: 1px solid rgba(120, 93, 66, 0.45) !important;
  background: linear-gradient(90deg, rgba(185, 139, 94, 0.95), rgba(138, 106, 75, 0.95)) !important;
  color: #fffaf2 !important;
  font-weight: 700 !important;
  letter-spacing: 0.2px;
  width: auto !important;
  min-width: 170px;
  padding-left: 18px !important;
  padding-right: 18px !important;
}

.stButton button:hover {
  border-color: rgba(120, 93, 66, 0.72) !important;
  transform: translateY(-1px);
  box-shadow: 0 12px 26px rgba(126, 92, 59, 0.28);
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
  background: rgba(7, 13, 24, 0.82) !important;
  border-color: rgba(255, 255, 255, 0.14) !important;
}

[data-testid="stFileUploader"] {
  border: 1px dashed rgba(140, 112, 86, 0.55);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.06);
  padding: 8px;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--line);
  border-radius: 14px;
  overflow: hidden;
}

@media (max-width: 900px) {
  .hero h1 {
    font-size: 2.2rem;
  }
  .hero p {
    font-size: 0.95rem;
  }
  .feature-card {
    min-height: auto;
  }
  .stButton button {
    min-width: 140px;
  }
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _safe_filename(name: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return clean or fallback


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _uploaded_file_to_text(uploaded_file) -> str:
    name = (uploaded_file.name or "file").lower()
    raw = uploaded_file.read()
    if name.endswith(".pdf"):
        return extract_pdf_text(raw)

    try:
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return f"Binary file uploaded: {uploaded_file.name} (size={len(raw)} bytes)"


def _create_local_paper_from_upload(uploaded_file, target_dir: str) -> str:
    text = _uploaded_file_to_text(uploaded_file).strip()
    filename = _safe_filename(uploaded_file.name, "dataset.txt")
    stem = Path(filename).stem
    out_path = os.path.join(target_dir, f"{stem}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {stem}\n")
        f.write("Authors: User Upload\n\n")
        f.write(text[:12000] if text else "No readable text extracted from uploaded file.")

    return out_path


def _uploaded_file_to_paper(uploaded_file) -> Dict[str, Any]:
    text = _uploaded_file_to_text(uploaded_file).strip()
    title = Path(uploaded_file.name or "uploaded_paper").stem
    return {
        "paper_id": _safe_filename(title, "uploaded_paper"),
        "title": title,
        "abstract": text[:8000] if text else "No readable text extracted.",
        "authors": ["User Upload"],
        "published": None,
        "updated": None,
        "url": None,
        "categories": ["uploaded"],
        "source": "upload",
    }


def _get_lab_llm(mode: str):
    return LocalLLMClient() if mode == "local" else MockLLMClient()


def _render_lab_bridge_graph(bridge: BridgeResult):
    try:
        from streamlit_agraph import Config, Edge, Node, agraph
    except ImportError:
        st.warning("Install `streamlit-agraph` to render graph visualization.")
        st.json(
            {
                "nodes": bridge.nodes,
                "edges": [
                    {"source": s, "target": t, "weight": round(w, 3)}
                    for s, t, w in bridge.edges
                ],
            }
        )
        return

    nodes = [Node(id=name, label=name, size=16, color="#3ca1ff") for name in bridge.nodes]
    edges = [Edge(source=src, target=dst, label=f"{weight:.2f}") for src, dst, weight in bridge.edges]
    config = Config(width="100%", height=420, directed=False, physics=True, nodeHighlightBehavior=True)
    agraph(nodes=nodes, edges=edges, config=config)


def _home_view():
    st.markdown(
        """
<div class="hero">
  <h1>Nexus</h1>
  <p>AI research workspace for experiment intelligence, cross-domain discovery, and paper audit.</p>
</div>
<div class="mono">EVP • SYNTROPY • PAPER AUDIT LAB</div>
""",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
<div class="feature-card">
  <h3>EVP</h3>
  <p>Select the next best experiment before spending compute. Upload one dataset and rank options by value.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="feature-card-gap"></div>', unsafe_allow_html=True)
        if st.button("Open EVP"):
            st.session_state.active_feature = "evp"

    with c2:
        st.markdown(
            """
<div class="feature-card">
  <h3>Syntropy</h3>
  <p>Discover cross-domain bridges from two uploaded sources and generate an opportunity narrative.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="feature-card-gap"></div>', unsafe_allow_html=True)
        if st.button("Open Syntropy"):
            st.session_state.active_feature = "syntropy"

    with c3:
        st.markdown(
            """
<div class="feature-card">
  <h3>Paper Audit Lab</h3>
  <p>Deconstruct a PDF, audit claim-vs-result consistency, and visualize concept connections.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="feature-card-gap"></div>', unsafe_allow_html=True)
        if st.button("Open Paper Lab"):
            st.session_state.active_feature = "lab"


def _feature_header(title: str, subtitle: str):
    top_left, top_right = st.columns([5, 1])
    with top_left:
        st.markdown(
            f"""
<div class="hero">
  <h1>{title}</h1>
  <p>{subtitle}</p>
</div>
""",
            unsafe_allow_html=True,
        )
    with top_right:
        if st.button("Back to Home"):
            st.session_state.active_feature = "home"
            st.rerun()


def _evp_view():
    _feature_header(
        "EVP - Experiment Value Predictor",
        "Upload one dataset/file and choose budget scale to get a recommended experiment.",
    )

    if "evp_result" not in st.session_state:
        st.session_state.evp_result = None

    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        dataset_file = st.file_uploader(
            "Upload dataset/file for EVP",
            type=["csv", "json", "txt", "md", "pdf"],
            key="evp_dataset_upload",
        )
        budget = st.select_slider("Budget Scale", options=["Low", "Medium", "High"], value="Medium")
        llm_mode = st.radio("LLM Mode", options=["mock", "local", "mock_static"], index=0)
        goal = st.text_area(
            "Goal",
            value="Improve outcome quality while minimizing compute cost.",
            height=90,
        )

        run = st.button("Run EVP", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if dataset_file is None:
            st.error("Please upload one dataset/file for EVP.")
        else:
            os.environ["EVP_LLM_MODE"] = llm_mode
            upload_dir = _ensure_dir("data/evp_uploads")
            _create_local_paper_from_upload(dataset_file, upload_dir)
            os.environ["EVP_LOCAL_PAPERS_DIR"] = upload_dir

            dataset_name = Path(dataset_file.name).stem
            constraints = {
                "budget": budget,
                "dataset_uploaded": dataset_file.name,
            }
            with st.spinner("Running EVP..."):
                st.session_state.evp_result = run_async(
                    run_pipeline(topic=dataset_name, goal=goal, constraints=constraints)
                )

    with right:
        result = st.session_state.evp_result
        if not result:
            st.info("Run EVP to see recommendation, value chart, and ranked experiments.")
            return

        experiments = result.get("experiments", [])
        rec_id = result.get("recommended_experiment_id")
        rec = next((e for e in experiments if e.get("id") == rec_id), None)

        top = st.columns(3)
        top[0].metric("Experiments", len(experiments))
        top[1].metric("Recommended", rec.get("id", "-") if rec else "-")
        top[2].metric("Best Value", f"{rec.get('value', 0):.2f}" if rec else "-")

        if rec:
            st.success(f"Recommended: {rec.get('title', 'Experiment')}")
            st.caption(rec.get("impact_rationale", ""))

        if experiments:
            x_vals = [e.get("resource_cost", 0) for e in experiments]
            y_vals = [e.get("impact_score", 0) for e in experiments]
            labels = [e.get("id", "exp") for e in experiments]
            colors = ["#22c55e" if e.get("id") == rec_id else "#3b82f6" for e in experiments]

            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=x_vals,
                        y=y_vals,
                        mode="markers+text",
                        text=labels,
                        textposition="top center",
                        marker=dict(size=14, color=colors),
                    )
                ]
            )
            fig.update_layout(
                title="Value vs Cost",
                height=320,
                margin=dict(l=20, r=20, t=45, b=20),
                xaxis_title="Resource Cost",
                yaxis_title="Impact Score",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                [
                    {
                        "id": e.get("id"),
                        "title": e.get("title"),
                        "model": e.get("model"),
                        "compute_units": e.get("compute_units"),
                        "novelty_score": e.get("novelty_score"),
                        "value": round(float(e.get("value", 0)), 3),
                    }
                    for e in experiments
                ],
                use_container_width=True,
            )


def _syntropy_view():
    _feature_header(
        "Syntropy - Cross-Domain Bridge",
        "Upload two files/datasets. Syntropy will infer a conceptual bridge between them.",
    )

    if "syntropy_result" not in st.session_state:
        st.session_state.syntropy_result = None

    left, right = st.columns([1, 2])

    with left:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        file_a = st.file_uploader(
            "Upload File/Dataset A",
            type=["csv", "json", "txt", "md", "pdf"],
            key="syn_upload_a",
        )
        file_b = st.file_uploader(
            "Upload File/Dataset B",
            type=["csv", "json", "txt", "md", "pdf"],
            key="syn_upload_b",
        )

        topic_a = st.text_input("Domain A label", value="Domain A")
        topic_b = st.text_input("Domain B label", value="Domain B")
        threshold = st.slider("Similarity threshold", min_value=0.2, max_value=0.9, value=0.5, step=0.05)
        syntropy_mode = st.radio("Syntropy LLM Mode", options=["mock", "openai"], index=0)

        run = st.button("Run Syntropy", type="primary")
        st.markdown("</div>", unsafe_allow_html=True)

    if run:
        if file_a is None or file_b is None:
            st.error("Please upload both files/datasets for Syntropy.")
        else:
            os.environ["SYNTROPY_LLM_MODE"] = syntropy_mode

            paper_a = _uploaded_file_to_paper(file_a)
            paper_b = _uploaded_file_to_paper(file_b)

            state = {
                "topic_a": topic_a,
                "topic_b": topic_b,
                "papers_a": [paper_a],
                "papers_b": [paper_b],
                "similarity_threshold": float(threshold),
                "max_results": 1,
                "use_local_papers": False,
            }

            try:
                from evp.syntropy.graph import build_syntropy_app

                app = build_syntropy_app()
                with st.spinner("Running Syntropy..."):
                    st.session_state.syntropy_result = app.invoke(state)
            except Exception as exc:
                st.session_state.syntropy_result = {"error": str(exc)}

    with right:
        result = st.session_state.syntropy_result
        if not result:
            st.info("Run Syntropy to generate bridge path and opportunity report.")
            return
        if result.get("error"):
            st.error(f"Syntropy failed: {result['error']}")
            return

        path = result.get("connection_path", [])
        report = result.get("final_report", "")
        summary = result.get("graph_summary", {})

        st.markdown("**Connection Path**")
        st.write(" -> ".join(path) if path else "No path generated")

        if summary:
            row = st.columns(3)
            row[0].metric("Nodes", summary.get("nodes", 0))
            row[1].metric("Edges", summary.get("edges", 0))
            row[2].metric("Threshold", summary.get("threshold", "-"))

        st.markdown("**Future Opportunity Report**")
        st.write(report or "No report generated")

        with st.expander("Trace"):
            st.json(result.get("trace", []))


def _paper_lab_view():
    _feature_header(
        "Paper Audit Lab",
        "Upload one PDF to run deconstruction, ghost audit, and concept bridge.",
    )

    if "lab_deconstruction" not in st.session_state:
        st.session_state.lab_deconstruction = None
    if "lab_audit" not in st.session_state:
        st.session_state.lab_audit = None
    if "lab_bridge" not in st.session_state:
        st.session_state.lab_bridge = None
    if "lab_raw_text" not in st.session_state:
        st.session_state.lab_raw_text = ""

    c1, c2, c3 = st.columns([1, 1.2, 1])

    with c1:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        pdf = st.file_uploader("Upload a research paper (.pdf)", type=["pdf"], key="lab_pdf_upload")
        llm_mode = st.radio("Lab LLM", options=["mock", "local"], index=0)
        run = st.button("Run Paper Audit", type="primary")

        if run:
            if pdf is None:
                st.error("Please upload a PDF.")
            else:
                with st.spinner("Running Paper Audit Lab..."):
                    raw_text = extract_pdf_text(pdf.read())
                    llm_client = _get_lab_llm(llm_mode)
                    deconstruction = deconstruct_paper(raw_text, llm_client=llm_client)
                    audit = inspect_consistency(
                        deconstruction.abstract_summary,
                        deconstruction.results_metrics,
                        llm_client=llm_client,
                    )
                    bridge = build_knowledge_bridge(deconstruction)

                    st.session_state.lab_raw_text = raw_text
                    st.session_state.lab_deconstruction = deconstruction
                    st.session_state.lab_audit = audit
                    st.session_state.lab_bridge = bridge

        audit = st.session_state.lab_audit
        if audit is None:
            st.info("Audit output appears here.")
        else:
            report = audit.model_dump()
            st.metric("Consistency", "Consistent" if report["is_consistent"] else "Flagged")
            st.write(report["verdict"])
            if report["discrepancies"]:
                for item in report["discrepancies"]:
                    st.write(f"- {item}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        st.markdown("**Knowledge Bridge**")
        bridge = st.session_state.lab_bridge
        if bridge is None:
            st.info("Graph appears here.")
        else:
            _render_lab_bridge_graph(bridge)
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="section-shell">', unsafe_allow_html=True)
        st.markdown("**Extracted Data JSON**")
        data = st.session_state.lab_deconstruction
        if data is None:
            st.info("Structured output appears here.")
        else:
            payload = {
                "deconstruction": data.model_dump(),
                "audit": st.session_state.lab_audit.model_dump() if st.session_state.lab_audit else {},
                "text_preview": st.session_state.lab_raw_text[:2000],
            }
            raw = json.dumps(payload, indent=2)
            st.code(raw, language="json")
            st.download_button(
                "Download JSON",
                data=raw,
                file_name="paper_audit_report.json",
                mime="application/json",
                use_container_width=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


if "active_feature" not in st.session_state:
    st.session_state.active_feature = "home"

page = st.session_state.active_feature
if page == "home":
    _home_view()
elif page == "evp":
    _evp_view()
elif page == "syntropy":
    _syntropy_view()
else:
    _paper_lab_view()
