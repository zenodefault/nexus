import asyncio
import os
import re
from typing import Any, Dict, List

import plotly.graph_objects as go
import streamlit as st

from evp.orchestration.pipeline import run_pipeline


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
  <h1>AI Research Decision Engine</h1>
  <p>Predicts which experiment is worth running before you burn compute.</p>
</div>
""",
    unsafe_allow_html=True,
)


with st.sidebar:
    st.subheader("Experiment Setup")
    topic = st.text_input("Research Topic", value="Brain MRI classification")
    goal = st.text_area(
        "Goal",
        value="Improve classification accuracy with minimal compute waste.",
        height=90,
    )
    budget = st.select_slider("Budget Constraint", options=["Low", "Medium", "High"], value="Medium")
    data_ready = st.toggle("Dataset Ready", value=True)
    llm_mode = st.radio("LLM Mode", options=["mock", "local"], index=0, horizontal=True)
    st.divider()
    st.caption("Local paper summaries (.txt/.md)")
    uploads = st.file_uploader(
        "Upload papers",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    save_uploads = st.button("Save uploads", width="stretch")
    run_button = st.button("Run EVP", type="primary")


constraints = {"budget": budget, "dataset_ready": data_ready}


if "results" not in st.session_state:
    st.session_state.results = None


status_placeholder = st.empty()


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


def save_uploaded_papers(files) -> List[str]:
    saved: List[str] = []
    if not files:
        return saved
    target_dir = os.path.join("data", "papers")
    os.makedirs(target_dir, exist_ok=True)
    for idx, file in enumerate(files, start=1):
        name = _safe_filename(file.name, f"paper_{idx}.txt")
        path = os.path.join(target_dir, name)
        data = file.read()
        with open(path, "wb") as f:
            f.write(data)
        saved.append(path)
    os.environ["EVP_LOCAL_PAPERS_DIR"] = target_dir
    return saved


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


if run_button:
    os.environ["EVP_LLM_MODE"] = llm_mode
    with status_placeholder:
        with st.status("Running EVP pipeline", expanded=True) as status:
            status.write("LiteratureAgent scanning papers")
            status.write("HypothesisAgent drafting experiments")
            status.write("ResourceEstimatorAgent sizing compute")
            status.write("ImpactPredictorAgent scoring novelty")
            status.write("Scoring Engine ranking value")
            with st.spinner("Synthesizing experiments..."):
                st.session_state.results = run_async(run_pipeline(topic, goal, constraints))
            status.update(state="complete", label="Pipeline complete")


if save_uploads:
    saved_paths = save_uploaded_papers(uploads)
    if saved_paths:
        st.sidebar.success(f"Saved {len(saved_paths)} file(s) to data/papers.")
    else:
        st.sidebar.warning("No files uploaded.")


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
        recommended = next((e for e in experiments if e.get("id") == recommended_id), None)

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
            badge = "<span class=\"evp-badge\">Recommended</span>" if is_recommended else "<span class=\"evp-outline\">Candidate</span>"
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
            colors = ["#27f4c0" if exp.get("id") == recommended_id else "#3ca1ff" for exp in experiments]

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
                xaxis=dict(title="Resource Cost (Low to High)", tickmode="linear", dtick=1, range=[0.5, 3.5]),
                yaxis=dict(title="Impact Score", range=[0, 11]),
                font=dict(color="#e6eefc", family="Space Grotesk"),
            )

            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    else:
        st.info("Enter a topic and run EVP to see ranked experiments, cost, and impact.")
