"""LangGraph Question Solve Engine: solver, verification, and bounded retry loop."""

from typing import Any, Dict
import time

from langgraph.graph import END, START, StateGraph

from app.generate.state import QuestionState, SolutionContract
from app.generate.nodes import (
    planner_node,
    math_solver_node,
    mcq_solver_node,
    theory_solver_node,
    normalizer_node,
)
from app.generate.nodes.verification import verification_node


MAX_VERIFICATION_RETRIES = 2


def route_by_type(state: QuestionState) -> str:
    """Route to the solver selected by the planner."""
    route = state.get("selected_route", "theory")
    if route in ("math", "mcq", "theory", "cached"):
        return route
    return "theory"


def route_after_verification(state: QuestionState) -> str:
    """Retry failed candidates a bounded number of times, then normalize."""
    verification = state.get("verification") or {}
    retry_count = int(state.get("retry_count", 0) or 0)

    if verification.get("passed") is True:
        return "normalizer"
    if verification.get("retryable") is False:
        return "normalizer"
    if retry_count < MAX_VERIFICATION_RETRIES:
        return "retry"
    return "normalizer"


def prepare_retry_node(state: QuestionState) -> Dict[str, Any]:
    """Pass safe verification feedback to the next candidate-generation attempt.

    Verifier feedback must never contain the official answer-key letter/text.
    """
    verification = state.get("verification") or {}
    reason = str(
        verification.get("retry_feedback")
        or verification.get("reason")
        or "The previous candidate did not pass verification. Re-evaluate it carefully."
    )
    return {
        "retry_count": int(state.get("retry_count", 0) or 0) + 1,
        "retry_reason": reason,
    }


def build_question_solver_graph():
    """Build and compile the graph for solving one question."""
    workflow = StateGraph(QuestionState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("math_solver", math_solver_node)
    workflow.add_node("mcq_solver", mcq_solver_node)
    workflow.add_node("theory_solver", theory_solver_node)
    workflow.add_node("verification", verification_node)
    workflow.add_node("prepare_retry", prepare_retry_node)
    workflow.add_node("normalizer", normalizer_node)

    workflow.add_edge(START, "planner")
    workflow.add_conditional_edges(
        "planner",
        route_by_type,
        {
            "math": "math_solver",
            "mcq": "mcq_solver",
            "theory": "theory_solver",
            "cached": "normalizer",
        },
    )

    workflow.add_edge("math_solver", "verification")
    workflow.add_edge("mcq_solver", "verification")
    workflow.add_edge("theory_solver", "verification")

    workflow.add_conditional_edges(
        "verification",
        route_after_verification,
        {
            "normalizer": "normalizer",
            "retry": "prepare_retry",
        },
    )
    workflow.add_conditional_edges(
        "prepare_retry",
        route_by_type,
        {
            "math": "math_solver",
            "mcq": "mcq_solver",
            "theory": "theory_solver",
            # A cached solution does not enter verification, so this is defensive.
            "cached": "normalizer",
        },
    )
    workflow.add_edge("normalizer", END)

    return workflow.compile()


question_solver_graph = build_question_solver_graph()


def solve_question(
    question_data: Dict[str, Any],
    paper_metadata: Dict[str, Any] | None = None,
) -> SolutionContract:
    """Run a single question through the solve-and-verify pipeline."""
    initial_state: QuestionState = {
        "number": str(question_data.get("number", "1")),
        "section": str(question_data.get("section", "A")),
        "text": str(question_data.get("text", "")),
        "marks": int(question_data.get("marks", 1)),
        "type": str(question_data.get("type", "short")),
        "options": question_data.get("options"),
        "has_figure": bool(question_data.get("has_figure", False)),
        "page": int(question_data.get("page", 1)),
        "choice_group": question_data.get("choice_group"),
        "chapter": question_data.get("chapter"),
        "metadata": paper_metadata or {},
        "retry_count": 0,
    }

    # Internal callers may pass an answer key for verification. It is never
    # copied into the solver prompt or the normalized solution.
    if question_data.get("expected_answer") is not None:
        initial_state["expected_answer"] = str(question_data["expected_answer"])

    started = time.perf_counter()
    final_state = question_solver_graph.invoke(initial_state)
    elapsed = time.perf_counter() - started



    return final_state["solution"]
