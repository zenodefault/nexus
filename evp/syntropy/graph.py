from langgraph.graph import END, StateGraph

from evp.syntropy.agents import (
    archivist_agent,
    connector_agent,
    deconstructor_agent,
    grant_writer_agent,
)
from evp.syntropy.state import GraphState


def build_syntropy_app():
    workflow = StateGraph(GraphState)

    workflow.add_node("archivist", archivist_agent)
    workflow.add_node("deconstructor", deconstructor_agent)
    workflow.add_node("connector", connector_agent)
    workflow.add_node("grant_writer", grant_writer_agent)

    workflow.set_entry_point("archivist")
    workflow.add_edge("archivist", "deconstructor")
    workflow.add_edge("deconstructor", "connector")
    workflow.add_edge("connector", "grant_writer")
    workflow.add_edge("grant_writer", END)

    return workflow.compile()
