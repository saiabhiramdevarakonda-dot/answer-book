from langgraph.graph import END, START, StateGraph

from app.graph.solve.nodes import (
    generate_node,
    retry_node,
    route_node,
    verify_node,
)
from app.graph.state import SolveState


MAX_RETRIES = 2


def route_after_verify(state: SolveState) -> str:
    verification = state.get("verification", {})
    retry_count = state.get("retry_count", 0)

    if verification.get("passed") is True:
        return "end"

    if retry_count < MAX_RETRIES:
        return "retry"

    return "end"


def build_solve_graph():
    builder = StateGraph(SolveState)

    builder.add_node("route", route_node)
    builder.add_node("generate", generate_node)
    builder.add_node("verify", verify_node)
    builder.add_node("retry", retry_node)

    builder.add_edge(START, "route")
    builder.add_edge("route", "generate")
    builder.add_edge("generate", "verify")

    builder.add_conditional_edges(
        "verify",
        route_after_verify,
        {
            "end": END,
            "retry": "retry",
        },
    )

    builder.add_edge("retry", "generate")

    return builder.compile()