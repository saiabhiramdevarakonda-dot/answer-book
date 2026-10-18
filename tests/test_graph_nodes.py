
import pytest

from app.graph.solve.nodes import (
    generate_node,
    verify_node,
    retry_node,
    route_node,
)
from app.graph.solve.graph import build_solve_graph


def make_solution(answer="4 ohm"):
    return {
        "question_number": "3(b)",
        "answer": answer,
        "steps": ["Use the parallel resistance formula."],
        "mark_split": [{"marks": 3, "for": "Correct calculation"}],
        "common_mistakes": [],
        "confidence": 0.95,
        "verified_by": "none",
        "needs_teacher_check": True,
    }


def make_question():
    return {
        "number": "3(b)",
        "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
        "marks": 3,
        "type": "numerical",
    }


def test_generate_node(monkeypatch):
    def fake_generate_answer(question, retry_reason=None):
        return make_solution()

    monkeypatch.setattr(
        "app.graph.solve.nodes.generate_answer",
        fake_generate_answer,
    )

    result = generate_node({"question": make_question(), "retry_count": 0})

    assert result["solution"]["answer"] == "4 ohm"


def test_verify_node_passes_correct_numerical_answer(monkeypatch):
    def fake_verify(question, solution, expected_answer=None):
        return {
            "passed": True,
            "verified_by": "symbolic",
            "reason": "Correct numerical answer.",
        }

    monkeypatch.setattr(
        "app.graph.solve.nodes.verify_solution",
        fake_verify,
    )

    state = {
        "question": make_question(),
        "solution": make_solution(),
    }
    result = verify_node(state)

    assert result["verification"]["passed"] is True
    assert result["solution"]["verified_by"] == "symbolic"
    assert result["solution"]["needs_teacher_check"] is False


def test_verify_node_flags_failed_verification(monkeypatch):
    monkeypatch.setattr(
        "app.graph.solve.nodes.verify_solution",
        lambda *args, **kwargs: {
            "passed": False,
            "verified_by": "none",
            "reason": "Incorrect answer.",
        },
    )

    result = verify_node({
        "question": make_question(),
        "solution": make_solution("18 ohm"),
    })

    assert result["verification"]["passed"] is False
    assert result["solution"]["needs_teacher_check"] is True


def test_generate_node_passes_retry_reason(monkeypatch):
    captured = {}

    def fake_generate_answer(question, retry_reason=None):
        captured["retry_reason"] = retry_reason
        return make_solution()

    monkeypatch.setattr(
        "app.graph.solve.nodes.generate_answer",
        fake_generate_answer,
    )

    generate_node({
        "question": make_question(),
        "retry_reason": "Previous answer was incorrect.",
    })

    assert captured["retry_reason"] == "Previous answer was incorrect."


def test_retry_node_increments_retry_count():
    result = retry_node({
        "question": make_question(),
        "retry_count": 0,
        "verification": {"reason": "Incorrect answer."},
    })

    assert result["retry_count"] == 1
    assert result["retry_reason"] == "Incorrect answer."


@pytest.mark.parametrize(
    ("question_type", "expected_route"),
    [
        ("numerical", "numerical"),
        ("mcq", "mcq"),
        ("short", "short"),
        ("long", "long"),
        ("diagram", "diagram"),
        ("unknown", "unsupported"),
    ],
)
def test_route_node(question_type, expected_route):
    result = route_node({
        "question": {
            **make_question(),
            "type": question_type,
        }
    })

    assert result["route"] == expected_route
