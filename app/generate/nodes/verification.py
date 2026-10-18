
from typing import Any

from app.generate.state import QuestionState
from app.verification.verifier import verify_solution


def verification_node(state: QuestionState) -> dict[str, Any]:
    """Verify a generated candidate and record the result in internal graph state."""

    raw_solution = state.get("raw_solution") or {}

    question = {
        "number": state.get("number", ""),
        "text": state.get("text", ""),
        "marks": state.get("marks", 1),
        "type": state.get("type", ""),
        "options": state.get("options") or [],
    }

    # A failed generation must never be marked as verified.
    if raw_solution.get("generation_error") or not str(
        raw_solution.get("answer", "")
    ).strip():
        verification = {
            "passed": False,
            "reason": raw_solution.get(
                "generation_error",
                "No answer was generated.",
            ),
            "verified_by": "none",
            "needs_teacher_check": True,
        }
    else:
        verification = verify_solution(
            question=question,
            solution=raw_solution,
            expected_answer=state.get("expected_answer"),
        )

    return {
        "verification": verification,
        "raw_solution": {
            **raw_solution,
            "verified_by": (
                verification.get("verified_by", "none")
                if verification.get("passed")
                else "none"
            ),
            "needs_teacher_check": (
                not verification.get("passed", False)
                or verification.get("needs_teacher_check", True)
            ),
            "confidence": (
                verification.get("confidence", raw_solution.get("confidence", 0.0))
            ),
        },
    }
