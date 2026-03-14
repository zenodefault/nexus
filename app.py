import json
import os
from typing import List

import streamlit as st

from evp.lab.paper_audit import (
    BridgeResult,
    build_knowledge_bridge,
    deconstruct_paper,
    extract_pdf_text,
    inspect_consistency,
)
from evp.utils.llm import LocalLLMClient, MockLLMClient


st.set_page_config(page_title="EVP Laboratory", page_icon="EVP", layout="wide")


st.title("EVP Laboratory")
st.caption("PDF Ingestion -> Structured Deconstruction -> Ghost Audit -> Knowledge Bridge")


with st.sidebar:
    st.subheader("Settings")
    llm_mode = st.radio("LLM Backend", options=["mock", "local"], index=0)
    st.markdown("Use `local` to call ACPX CLI (Gemini/Qwen) via env-configured command.")


def _get_llm(mode: str):
    return LocalLLMClient() if mode == "local" else MockLLMClient()


def _render_bridge_graph(bridge: BridgeResult):
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

    nodes = [
        Node(id=name, label=name, size=18, color="#3ca1ff")
        for name in bridge.nodes
    ]
    edges = [
        Edge(source=src, target=dst, label=f"{weight:.2f}")
        for src, dst, weight in bridge.edges
    ]

    config = Config(
        width="100%",
        height=500,
        directed=False,
        physics=True,
        nodeHighlightBehavior=True,
        collapsible=True,
    )
    agraph(nodes=nodes, edges=edges, config=config)


uploaded_pdf = st.file_uploader("Upload a research paper (.pdf)", type=["pdf"])


if "deconstruction" not in st.session_state:
    st.session_state.deconstruction = None
if "audit" not in st.session_state:
    st.session_state.audit = None
if "bridge" not in st.session_state:
    st.session_state.bridge = None
if "raw_text" not in st.session_state:
    st.session_state.raw_text = ""


if uploaded_pdf is not None:
    with st.spinner("Extracting text from PDF with PyMuPDF..."):
        try:
            pdf_bytes = uploaded_pdf.read()
            st.session_state.raw_text = extract_pdf_text(pdf_bytes)
        except Exception as exc:
            st.error(f"PDF extraction failed: {exc}")
            st.stop()

    st.success("PDF text extracted successfully.")

    llm_client = _get_llm(llm_mode)
    with st.spinner("Running deconstruction + ghost inspection + bridge builder..."):
        deconstruction = deconstruct_paper(st.session_state.raw_text, llm_client=llm_client)
        audit = inspect_consistency(
            deconstruction.abstract_summary,
            deconstruction.results_metrics,
            llm_client=llm_client,
        )
        bridge = build_knowledge_bridge(deconstruction)

    st.session_state.deconstruction = deconstruction
    st.session_state.audit = audit
    st.session_state.bridge = bridge


paper_tab, bridge_tab, data_tab = st.tabs([
    "📊 Paper Audit",
    "🕸️ Knowledge Bridge",
    "🧩 Extracted Data",
])


with paper_tab:
    st.subheader("Ghost Inspector Report")
    if st.session_state.audit is None:
        st.info("Upload a PDF to generate the audit report.")
    else:
        report = st.session_state.audit.model_dump()
        st.metric("Consistency", "Consistent" if report["is_consistent"] else "Flagged")
        st.write(report["verdict"])
        if report["discrepancies"]:
            st.markdown("**Discrepancies**")
            for item in report["discrepancies"]:
                st.write(f"- {item}")
        else:
            st.write("No discrepancies detected.")


with bridge_tab:
    st.subheader("Domain Bridge Graph")
    if st.session_state.bridge is None:
        st.info("Upload a PDF to build the knowledge bridge graph.")
    else:
        _render_bridge_graph(st.session_state.bridge)


with data_tab:
    st.subheader("Structured Deconstruction JSON")
    if st.session_state.deconstruction is None:
        st.info("Upload a PDF to view extracted structured data.")
    else:
        payload = {
            "deconstruction": st.session_state.deconstruction.model_dump(),
            "audit": st.session_state.audit.model_dump() if st.session_state.audit else {},
            "text_preview": st.session_state.raw_text[:2000],
        }
        st.code(json.dumps(payload, indent=2), language="json")
        st.download_button(
            label="Download JSON",
            data=json.dumps(payload, indent=2),
            file_name="paper_audit_report.json",
            mime="application/json",
        )
