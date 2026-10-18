
from unittest.mock import patch

from app.verification.verifier import verify_solution


QUESTION = {
    "number": "21",
    "type": "short",
    "text": "Why is π classified as an irrational number?",
    "marks": 2,
}


def make_rubric(score, is_correct, reason):
    return {
        "score": score,
        "max_marks": 2,
        "criteria": [
            {
                "concept": "Cannot be expressed as a ratio of integers",
                "marks_available": 1,
                "marks_awarded": min(score, 1),
                "reason": "Evaluates the fractional definition.",
            },
            {
                "concept": "Decimal expansion does not terminate or repeat",
                "marks_available": 1,
                "marks_awarded": max(0, score - 1),
                "reason": "Evaluates the decimal definition.",
            },
        ],
        "is_correct": is_correct,
        "confidence": 0.95,
        "reason": reason,
        "needs_teacher_check": False,
    }


def run_verifier_with_mock(rubric, answer):
    solution = {
        "answer": answer,
        "steps": [],
        "mark_split": [],
        "common_mistakes": [],
    }

    with patch(
        "app.verification.verifier.call_groq_json",
        return_value=rubric,
    ):
        return verify_solution(QUESTION, solution)


def test_descriptive_correct_answer_passes():
    rubric = make_rubric(
        2,
        True,
        "The answer correctly explains irrationality.",
    )

    result = run_verifier_with_mock(
        rubric,
        "Pi cannot be written as a ratio of two integers, "
        "and its decimal digits continue without repeating.",
    )

    assert result["passed"] is True
    assert result["verified_by"] == "llm_rubric"
    assert result["score"] == 2
    assert result["needs_teacher_check"] is False


def test_descriptive_incomplete_answer_gets_partial_credit():
    rubric = make_rubric(
        1,
        False,
        "The fractional definition is correct, but the decimal "
        "property is missing.",
    )

    result = run_verifier_with_mock(
        rubric,
        "Pi cannot be expressed as a ratio of two integers.",
    )

    assert result["passed"] is False
    assert result["verified_by"] == "llm_rubric"
    assert result["score"] == 1
    assert result["max_marks"] == 2


def test_descriptive_incorrect_answer_fails():
    rubric = make_rubric(
        0,
        False,
        "22/7 is an approximation of pi, not its exact value.",
    )

    result = run_verifier_with_mock(
        rubric,
        "Pi is rational because it equals exactly 22/7.",
    )

    assert result["passed"] is False
    assert result["verified_by"] == "llm_rubric"
    assert result["score"] == 0


def test_descriptive_invalid_boolean_requests_teacher_review():
    rubric = make_rubric(
        2,
        True,
        "The answer correctly explains irrationality.",
    )
    rubric["needs_teacher_check"] = "false"

    result = run_verifier_with_mock(
        rubric,
        "Pi cannot be written as a ratio of two integers, "
        "and its decimal digits continue without repeating.",
    )

    assert result["passed"] is False
    assert result["verified_by"] == "none"
    assert result["needs_teacher_check"] is True


def test_descriptive_invalid_available_marks_request_teacher_review():
    rubric = make_rubric(
        2,
        True,
        "The answer correctly explains irrationality.",
    )
    rubric["criteria"][0]["marks_available"] = 0.5

    result = run_verifier_with_mock(
        rubric,
        "Pi cannot be written as a ratio of two integers, "
        "and its decimal digits continue without repeating.",
    )

    assert result["passed"] is False
    assert result["verified_by"] == "none"
    assert result["needs_teacher_check"] is True
