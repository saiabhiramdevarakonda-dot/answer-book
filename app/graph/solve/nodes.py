from app.answer_generation.generator import generate_answer
from app.verification.verifier import verify_solution
from app.graph.state import SolveState


def route_node(state: SolveState) -> SolveState:
    """
    Identify the question type and store the selected route.
    """

    question = state["question"]
    question_type = question.get("type", "").lower()

    supported_types = {
        "mcq",
        "numerical",
        "short",
        "long",
        "diagram",
    }

    if question_type not in supported_types:
        route = "unsupported"
    else:
        route = question_type

    return {
        **state,
        "route": route,
    }


def generate_node(state: SolveState) -> SolveState:
    question = state["question"]

    retry_reason = state.get("retry_reason")

    solution = generate_answer(
        question,
        retry_reason=retry_reason,
    )

    return {
        **state,
        "solution": solution,
    }


def verify_node(state: SolveState) -> SolveState:
    question = state["question"]
    solution = state["solution"]

    verification = verify_solution(
    question,
    solution,
    expected_answer=state.get("expected_answer"),
)

    return {
        **state,
        "verification": verification,
    }


def retry_node(state: SolveState) -> SolveState:
    """
    Prepare the state for another generation attempt
    after verification fails.
    """

    retry_count = state.get("retry_count", 0)
    verification = state.get("verification", {})

    return {
        **state,
        "retry_count": retry_count + 1,
        "retry_reason": verification.get(
            "reason",
            "Previous solution failed verification.",
        ),
    }