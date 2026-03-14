import asyncio
import os
import re
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_agraph import Config, Edge, Node, agraph
except ImportError:  # pragma: no cover - optional dependency for Syntropy graph
    Config = None
    Edge = None
    Node = None
    agraph = None

from evp.orchestration.pipeline import run_pipeline
from evp.syntropy.graph import build_syntropy_app


st.set_page_config(
    page_title="EVP | AI Research Decision Engine",
    page_icon="EVP",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
  --bg-1: #0a0f1f;
  --bg-2: #0c1426;
  --bg-3: #101b2e;
  --panel: rgba(13, 20, 38, 0.88);
  --panel-2: rgba(9, 16, 31, 0.92);
  --ink: #eef5ff;
  --muted: #a3b6d3;
  --accent: #29f1c3;
  --accent-2: #3bb1ff;
  --accent-3: #ffd166;
  --warn: #ffb347;
}

html, body, [class*="stApp"] {
  font-family: "Space Grotesk", sans-serif;
  color: var(--ink);
  background: radial-gradient(circle at 10% 10%, rgba(41, 241, 195, 0.18) 0%, transparent 40%),
              radial-gradient(circle at 90% 0%, rgba(59, 177, 255, 0.22) 0%, transparent 45%),
              radial-gradient(circle at 80% 90%, rgba(255, 209, 102, 0.14) 0%, transparent 45%),
              linear-gradient(135deg, var(--bg-1), var(--bg-2) 55%, var(--bg-3));
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, rgba(8, 14, 30, 0.96), rgba(10, 18, 36, 0.98));
  border-right: 1px solid rgba(255, 255, 255, 0.08);
}

.evp-hero {
  padding: 18px 22px;
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(41, 241, 195, 0.16), rgba(59, 177, 255, 0.14));
  border: 1px solid rgba(41, 241, 195, 0.28);
  box-shadow: 0 18px 45px rgba(0, 0, 0, 0.4);
}

.evp-hero h1 {
  font-size: 38px;
  margin: 0 0 6px 0;
}

.evp-hero p {
  margin: 0;
  color: var(--muted);
  font-size: 16px;
}

.evp-card {
  padding: 18px;
  border-radius: 16px;
  background: linear-gradient(160deg, rgba(13, 20, 38, 0.92), rgba(8, 12, 24, 0.96));
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 14px 32px rgba(0, 0, 0, 0.42);
  backdrop-filter: blur(6px);
  animation: fadeUp 0.45s ease;
}

.evp-card h3 {
  margin: 0 0 8px 0;
}

.evp-muted {
  color: var(--muted);
  font-size: 14px;
}

.evp-mono {
  font-family: "JetBrains Mono", monospace;
}

.evp-badge {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  background: linear-gradient(90deg, #27f4c0, #3ca1ff);
  color: #04121e;
  font-weight: 700;
  letter-spacing: 0.5px;
  animation: pulse 1.6s infinite;
  text-transform: uppercase;
  font-size: 12px;
}

.evp-outline {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.evp-thought {
  background: rgba(14, 22, 38, 0.72);
  border-radius: 14px;
  padding: 12px 14px;
  border: 1px solid rgba(255, 255, 255, 0.1);
  margin-bottom: 10px;
  animation: fadeUp 0.4s ease;
}

@keyframes pulse {
  0% { box-shadow: 0 0 0 0 rgba(39, 244, 192, 0.4); }
  70% { box-shadow: 0 0 0 14px rgba(39, 244, 192, 0); }
  100% { box-shadow: 0 0 0 0 rgba(39, 244, 192, 0); }
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
"""


st.markdown(CSS, unsafe_allow_html=True)


st.markdown(
    """
<div class="evp-hero">
  <h1>Research Ops Console</h1>
  <p>Upload papers, set a goal, and let the agents build the next step.</p>
</div>
""",
    unsafe_allow_html=True,
)


if "results" not in st.session_state:
    st.session_state.results = None
if "syntropy_results" not in st.session_state:
    st.session_state.syntropy_results = None


status_placeholder = st.empty()
syntropy_status_placeholder = st.empty()


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


@st.cache_resource
def get_syntropy_app():
    return build_syntropy_app()


def _safe_filename(name: str, fallback: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return clean or fallback


def save_uploaded_papers(files, env_var: str = "EVP_LOCAL_PAPERS_DIR") -> List[str]:
    saved: List[str] = []
    if not files:
        return saved
    target_dir = os.path.join("data", "papers")
    os.makedirs(target_dir, exist_ok=True)
    for idx, file in enumerate(files, start=1):
        name = _safe_filename(file.name, f"paper_{idx}.txt")
        path = os.path.join(target_dir, name)
        data = file.getvalue()
        with open(path, "wb") as f:
            f.write(data)
        saved.append(path)
    os.environ[env_var] = target_dir
    return saved


def parse_uploaded_paper(file, fallback_id: str) -> Dict[str, Any]:
    raw = file.getvalue().decode("utf-8", errors="ignore").strip()
    if not raw:
        return {
            "paper_id": fallback_id,
            "title": file.name or fallback_id,
            "abstract": "",
            "authors": [],
        }

    lines = raw.splitlines()
    title = file.name or fallback_id
    authors: List[str] = []
    body_start = 0

    if lines and lines[0].lower().startswith("title:"):
        title = lines[0].split(":", 1)[1].strip() or title
        body_start = 1
        if len(lines) > 1 and lines[1].lower().startswith("authors:"):
            authors_line = lines[1].split(":", 1)[1]
            authors = [a.strip() for a in authors_line.split(",") if a.strip()]
            body_start = 2

    abstract = "\n".join(lines[body_start:]).strip()
    return {
        "paper_id": fallback_id,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "source": "upload",
    }


def infer_topic_from_papers(papers: List[Dict[str, Any]], fallback: str) -> str:
    for paper in papers:
        title = str(paper.get("title", "")).strip()
        if title:
            return title
    return fallback


def summarize_payload(payload: Dict[str, Any]) -> str:
    if "summary" in payload:
        return payload["summary"]
    if "hypotheses" in payload:
        return f"Generated {len(payload.get('hypotheses', []))} hypotheses"
    if "compute_units" in payload:
        return f"Compute units: {payload.get('compute_units', 'unknown')}"
    if "novelty_score" in payload:
        return f"Novelty score: {payload.get('novelty_score', 'unknown')}"
    return "Captured context"


tab_evp, tab_syntropy = st.tabs(["EVP", "Syntropy"])

with tab_evp:
    st.markdown("**EVP: Experiment Value Predictor**")
    st.caption("Upload papers, define the goal and budget, then run the pipeline.")

    with st.form("evp_form"):
        uploads = st.file_uploader(
            "Upload papers",
            type=["txt", "md"],
            accept_multiple_files=True,
            key="evp_uploads",
        )
        goal = st.text_area(
            "Goal",
            value="Improve classification accuracy with minimal compute waste.",
            height=90,
            key="evp_goal",
        )
        budget = st.select_slider(
            "Budget Constraint",
            options=["Low", "Medium", "High"],
            value="Medium",
            key="evp_budget",
        )
        with st.expander("Advanced", expanded=False):
            llm_mode = st.radio(
                "LLM Mode",
                options=["mock", "local"],
                index=0,
                horizontal=True,
                key="evp_llm_mode",
            )
        evp_submit = st.form_submit_button("Run EVP")

    if evp_submit:
        if not uploads:
            st.warning("Upload at least one paper to run EVP.")
        else:
            parsed = [
                parse_uploaded_paper(file, f"evp_{idx}")
                for idx, file in enumerate(uploads, start=1)
            ]
            topic = infer_topic_from_papers(parsed, "Uploaded papers")
            constraints = {"budget": budget, "dataset_ready": True}
            os.environ["EVP_LLM_MODE"] = llm_mode
            save_uploaded_papers(uploads)
            with status_placeholder:
                with st.status("Running EVP pipeline", expanded=True) as status:
                    status.write("LiteratureAgent scanning papers")
                    status.write("HypothesisAgent drafting experiments")
                    status.write("ResourceEstimatorAgent sizing compute")
                    status.write("ImpactPredictorAgent scoring novelty")
                    status.write("Scoring Engine ranking value")
                    with st.spinner("Synthesizing experiments..."):
                        st.session_state.results = run_async(
                            run_pipeline(topic, goal, constraints)
                        )
                    status.update(state="complete", label="Pipeline complete")

    results = st.session_state.results

    left, right = st.columns([0.68, 0.32])

    with right:
        st.markdown("**Live Thought Stream**")
        if results and results.get("memory"):
            for item in results["memory"]:
                agent = item.get("agent", "Agent")
                payload = item.get("payload", {})
                summary = summarize_payload(payload)
                st.markdown(
                    f"""
<div class="evp-thought">
  <div class="evp-mono" style="font-size:12px; color:#8bdcff;">{agent}</div>
  <div style="font-size:14px;">{summary}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                """
<div class="evp-thought">
  <div class="evp-mono" style="font-size:12px; color:#8bdcff;">LiteratureAgent</div>
  <div style="font-size:14px;">Awaiting input. Ready to scan the field.</div>
</div>
<div class="evp-thought">
  <div class="evp-mono" style="font-size:12px; color:#8bdcff;">ImpactPredictorAgent</div>
  <div style="font-size:14px;">Scorecards will appear here after a run.</div>
</div>
""",
                unsafe_allow_html=True,
            )

    with left:
        st.markdown("**Experiment Intelligence**")

        if results:
            experiments: List[Dict[str, Any]] = results.get("experiments", [])
            recommended_id = results.get("recommended_experiment_id")
            recommended = next(
                (e for e in experiments if e.get("id") == recommended_id), None
            )

            top_row = st.columns(3)
            top_row[0].metric("Experiments Generated", len(experiments))
            if recommended:
                top_row[1].metric("Recommended", recommended.get("title", "Experiment"))
                top_row[2].metric("Best Value", f"{recommended.get('value', 0):.2f}")
            else:
                top_row[1].metric("Recommended", "Pending")
                top_row[2].metric("Best Value", "-")

            st.markdown("**Recommendation**")
            if recommended:
                st.markdown(
                    f"""
<div class="evp-card">
  <div class="evp-badge">Recommended</div>
  <h3>{recommended.get('title', 'Experiment')}</h3>
  <div class="evp-muted">Model: {recommended.get('model', 'unknown')}</div>
  <p>{recommended.get('impact_rationale', 'No rationale returned.')}</p>
  <div class="evp-mono" style="font-size:13px;">Value score: {recommended.get('value', 0):.2f}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
            else:
                st.info("Run the pipeline to see the best experiment recommendation.")

            st.markdown("**Experiment Cards**")
            for exp in experiments:
                is_recommended = exp.get("id") == recommended_id
                badge = (
                    "<span class=\\\"evp-badge\\\">Recommended</span>"
                    if is_recommended
                    else "<span class=\\\"evp-outline\\\">Candidate</span>"
                )
                st.markdown(
                    f"""
<div class="evp-card">
  {badge}
  <h3>{exp.get('title', 'Experiment')}</h3>
  <div class="evp-muted">Model: {exp.get('model', 'unknown')}</div>
  <div class="evp-muted">Compute units: {exp.get('compute_units', 'unknown')} | Expected gain: {exp.get('expected_gain', '0')}%</div>
  <p>{exp.get('resource_rationale', 'No compute rationale provided.')}</p>
  <div class="evp-mono" style="font-size:13px;">Novelty: {exp.get('impact_score', 0)} | Value: {exp.get('value', 0):.2f}</div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown("**Value vs. Cost Quadrant**")
            if experiments:
                cost_values = [exp.get("resource_cost", 0) for exp in experiments]
                impact_values = [exp.get("impact_score", 0) for exp in experiments]
                labels = [exp.get("title", "Experiment") for exp in experiments]
                colors = [
                    "#27f4c0" if exp.get("id") == recommended_id else "#3ca1ff"
                    for exp in experiments
                ]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=cost_values,
                        y=impact_values,
                        mode="markers+text",
                        text=labels,
                        textposition="top center",
                        marker=dict(size=16, color=colors, line=dict(color="#07111f", width=2)),
                        hovertemplate="Cost: %{x}<br>Impact: %{y}<extra></extra>",
                    )
                )

                if cost_values and impact_values:
                    fig.add_shape(
                        type="line",
                        x0=sum(cost_values) / len(cost_values),
                        x1=sum(cost_values) / len(cost_values),
                        y0=min(impact_values) - 1,
                        y1=max(impact_values) + 1,
                        line=dict(color="rgba(255,255,255,0.2)", dash="dash"),
                    )
                    fig.add_shape(
                        type="line",
                        x0=min(cost_values) - 0.5,
                        x1=max(cost_values) + 0.5,
                        y0=sum(impact_values) / len(impact_values),
                        y1=sum(impact_values) / len(impact_values),
                        line=dict(color="rgba(255,255,255,0.2)", dash="dash"),
                    )

                fig.update_layout(
                    height=360,
                    margin=dict(l=20, r=20, t=30, b=30),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(
                        title="Resource Cost (Low to High)",
                        tickmode="linear",
                        dtick=1,
                        range=[0.5, 3.5],
                    ),
                    yaxis=dict(title="Impact Score", range=[0, 11]),
                    font=dict(color="#e6eefc", family="Space Grotesk"),
                )

                st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        else:
            st.info("Enter a topic and run EVP to see ranked experiments, cost, and impact.")

with tab_syntropy:
    st.markdown("**Syntropy: Cross-Domain Research Catalyst**")
    st.caption("Upload two papers from different fields to discover a bridge.")

    with st.form("syntropy_form"):
        upload_a = st.file_uploader(
            "Upload paper for Domain A",
            type=["txt", "md"],
            accept_multiple_files=False,
            key="syntropy_upload_a",
        )
        upload_b = st.file_uploader(
            "Upload paper for Domain B",
            type=["txt", "md"],
            accept_multiple_files=False,
            key="syntropy_upload_b",
        )
        similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.3,
            max_value=0.8,
            value=0.5,
            step=0.05,
            key="syntropy_threshold",
        )
        with st.expander("Advanced", expanded=False):
            syntropy_llm_mode = st.radio(
                "Syntropy LLM Mode",
                options=["openai", "mock"],
                index=0,
                horizontal=True,
                key="syntropy_llm_mode",
            )
        syntropy_submit = st.form_submit_button("Run Syntropy")

    if syntropy_submit:
        if not upload_a or not upload_b:
            st.warning("Upload one paper for each domain to run Syntropy.")
        else:
            papers_a = [parse_uploaded_paper(upload_a, "domain_a")]
            papers_b = [parse_uploaded_paper(upload_b, "domain_b")]
            topic_a = infer_topic_from_papers(papers_a, "Domain A")
            topic_b = infer_topic_from_papers(papers_b, "Domain B")
            os.environ["SYNTROPY_LLM_MODE"] = syntropy_llm_mode
            inputs = {
                "topic_a": topic_a,
                "topic_b": topic_b,
                "papers_a": papers_a,
                "papers_b": papers_b,
                "similarity_threshold": similarity_threshold,
                "trace": [],
            }
            with syntropy_status_placeholder:
                with st.status("Running Syntropy pipeline", expanded=True) as status:
                    status.write("Archivist parsing uploads")
                    status.write("Deconstructor extracting concepts")
                    status.write("Connector mapping graph similarity")
                    status.write("GrantWriter drafting proposal")
                    with st.spinner("Synthesizing cross-domain bridge..."):
                        st.session_state.syntropy_results = get_syntropy_app().invoke(
                            inputs
                        )
                    status.update(state="complete", label="Syntropy complete")

    syntropy_results = st.session_state.syntropy_results
    st.markdown(
        "Discover a shortest conceptual bridge between two scientific domains using graph theory."
    )

    if syntropy_results:
        bridge_col, report_col = st.columns([0.42, 0.58])

        with bridge_col:
            st.markdown("**The Bridge**")
            connection_path = syntropy_results.get("connection_path", []) or []
            for idx, step in enumerate(connection_path, start=1):
                st.markdown(f"**Step {idx}:** {step}")

            summary = syntropy_results.get("graph_summary", {})
            if summary:
                st.caption(
                    f"Graph nodes: {summary.get('nodes', 0)} | "
                    f"edges: {summary.get('edges', 0)} | "
                    f"threshold: {summary.get('threshold', '-')}"
                )

        with report_col:
            st.markdown("**Research Proposal**")
            st.markdown(syntropy_results.get("final_report", ""))

        st.divider()
        st.markdown("**Knowledge Graph Visualization**")
        if agraph is None:
            st.warning("Install streamlit-agraph to see the interactive graph.")
        else:
            nodes = []
            edges = []
            for i, step in enumerate(connection_path):
                if i == 0:
                    color = "#90EE90"
                elif i == len(connection_path) - 1:
                    color = "#87CEEB"
                else:
                    color = "#FFD700"
                nodes.append(Node(id=str(i), label=step, size=25, color=color))
                if i > 0:
                    edges.append(Edge(source=str(i - 1), target=str(i), label="relates to"))

            config = Config(width=800, height=400, directed=True)
            agraph(nodes=nodes, edges=edges, config=config)

        st.divider()
        st.markdown("**Agent Trace**")
        trace = syntropy_results.get("trace", []) or []
        if trace:
            for item in trace:
                agent = item.get("agent", "Agent")
                payload = item.get("payload", {})
                summary = payload.get("note") or payload.get("source") or "Step complete"
                st.markdown(
                    f"""
<div class="evp-thought">
  <div class="evp-mono" style="font-size:12px; color:#8bdcff;">{agent}</div>
  <div style="font-size:14px;">{summary}</div>
</div>
""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Trace will appear here after a Syntropy run.")
    else:
        st.info("Enter two domains and run Syntropy to see the bridge and proposal.")
