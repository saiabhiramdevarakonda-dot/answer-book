
import json
import math
import re
from typing import Any

from app.llm.client import call_groq_json


def verify_solution(
    question: dict[str, Any],
    solution: dict[str, Any],
    expected_answer: str | None = None,
) -> dict[str, Any]:
    """Verify a candidate answer without changing the solution contract."""

    answer = str(solution.get("answer", "")).strip()

    if not answer:
        return {
            "passed": False,
            "reason": "No answer was generated.",
            "verified_by": "none",
            "needs_teacher_check": True,
        }

    question_type = str(question.get("type", "")).strip().lower()

    if question_type in {"numerical", "math"}:
        return _verify_numerical(
            str(question.get("text", "")),
            answer,
        )

    if question_type == "mcq":
        return _verify_mcq(question, answer, expected_answer)

    if question_type in {"short", "long", "theory", "diagram"}:
        return _verify_descriptive(question, solution)

    return {
        "passed": False,
        "reason": f"Unsupported question type: {question_type or 'missing'}.",
        "verified_by": "none",
        "needs_teacher_check": True,
        "retryable": False,
    }


def _verify_mcq(
    question: dict[str, Any],
    answer: str,
    expected_answer: str | None,
) -> dict[str, Any]:
    """Compare the candidate answer with an official answer-key entry."""

    if expected_answer is None:
        return {
            "passed": False,
            "reason": "Cannot verify MCQ without an answer key.",
            "verified_by": "none",
            "needs_teacher_check": True,
            "retryable": False,
        }

    expected = str(expected_answer).strip().upper()

    if not re.fullmatch(r"[A-D]", expected):
        return {
            "passed": False,
            "reason": "The answer-key entry is not a valid option letter.",
            "verified_by": "none",
            "needs_teacher_check": True,
            "retryable": False,
        }

    options = question.get("options") or {}

    # Support either ["lioness", "lion", ...] or {"A": "lioness", ...}.
    if isinstance(options, list):
        options = {
            chr(ord("A") + index): str(value)
            for index, value in enumerate(options)
            if index < 26
        }

    if not isinstance(options, dict):
        return {
            "passed": False,
            "reason": "Question options are not in a supported format.",
            "verified_by": "none",
            "needs_teacher_check": True,
        }

    normalized_options = {
        str(key).strip().upper(): str(value).strip()
        for key, value in options.items()
    }

    if normalized_options and expected not in normalized_options:
        return {
            "passed": False,
            "reason": "The expected answer-key option is unavailable.",
            "verified_by": "none",
            "needs_teacher_check": True,
            "retryable": False,
        }

    generated = answer.strip()
    expected_text = normalized_options.get(expected, "")

    # Accept "B", "B. lion", "(B)", "b) lion", and similar formats.
    match = re.match(
        r"^\s*\(?([A-D])\)?(?:[.)\s:,-]+|$)",
        generated,
        flags=re.IGNORECASE,
    )
    generated_letter = match.group(1).upper() if match else None

    passed = (
        generated_letter == expected
        or (
            bool(expected_text)
            and _normalize_text(generated) == _normalize_text(expected_text)
        )
    )

    return {
        "passed": passed,
        "reason": (
            "Generated answer matches the official answer key."
            if passed
            else "Generated answer does not match the official answer key."
        ),
        # Keep retry feedback free of the correct letter or option text so
        # the answer key cannot leak into a subsequent generation attempt.
        "retry_feedback": (
            None
            if passed
            else (
                "The candidate option does not match the official answer key. "
                "Re-evaluate the assertion, reason, their truth values, and "
                "whether the reason logically explains the assertion. "
                "Choose independently from the presented options."
            )
        ),
        "verified_by": "answer_key",
        "needs_teacher_check": not passed,
        "retryable": not passed,
    }


def _normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation for conservative text matching."""
    return re.sub(r"[\W_]+", " ", text.casefold()).strip()


def _verify_descriptive(
    question: dict[str, Any],
    solution: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a descriptive answer against an explicit marking rubric."""

    try:
        max_marks = _finite_number(
            question.get("marks", 0),
            "maximum marks",
        )

        if max_marks <= 0:
            raise ValueError("Valid maximum marks are required.")

        prompt = f"""
You are a strict but fair examination answer evaluator.

Evaluate the candidate answer against the question and its maximum marks.
Accept different wording when it expresses the same correct meaning.
Award partial credit where appropriate. Do not reward incorrect claims.

Question:
{question.get("text", "")}

Question type: {question.get("type", "")}
Maximum marks: {max_marks}

Candidate answer:
{solution.get("answer", "")}

Candidate steps:
{json.dumps(solution.get("steps", []), ensure_ascii=False)}

Candidate mark split:
{json.dumps(solution.get("mark_split", []), ensure_ascii=False)}

Return a JSON object with these fields:
{{
  "score": 0,
  "max_marks": {max_marks},
  "criteria": [
    {{
      "concept": "required concept",
      "marks_available": 1,
      "marks_awarded": 1,
      "reason": "brief explanation"
    }}
  ],
  "is_correct": true,
  "confidence": 0.9,
  "reason": "brief overall explanation",
  "needs_teacher_check": false
}}

Rules:
- Available criterion marks must add up to {max_marks}.
- Awarded criterion marks must add up to score.
- Score must be between zero and {max_marks}.
- Confidence must be between zero and one.
- is_correct and needs_teacher_check must be booleans.
- Use teacher review when correctness is ambiguous or uncertain.
- Agreement from an LLM is not independent mathematical proof.
"""

        result = call_groq_json(
            system_prompt=(
                "You evaluate school examination answers using marking rubrics. "
                "Return only valid JSON."
            ),
            user_prompt=prompt,
        )

        return _validate_rubric_result(result, max_marks)

    except Exception as exc:
        return {
            "passed": False,
            "reason": f"Rubric evaluation unavailable or invalid: {exc}",
            "verified_by": "none",
            "needs_teacher_check": True,
        }


def _validate_rubric_result(
    result: Any,
    max_marks: float,
) -> dict[str, Any]:
    """Validate rubric fields and totals before accepting the evaluation."""

    if not isinstance(result, dict):
        raise ValueError("Evaluator response must be a JSON object.")

    required = {
        "score",
        "max_marks",
        "criteria",
        "is_correct",
        "confidence",
        "reason",
        "needs_teacher_check",
    }

    if not required.issubset(result):
        raise ValueError("Evaluator response is missing required fields.")

    score = _finite_number(result["score"], "score")
    reported_max = _finite_number(result["max_marks"], "max_marks")
    confidence = _finite_number(result["confidence"], "confidence")
    criteria = result["criteria"]

    if not 0 <= score <= max_marks:
        raise ValueError("Score is outside the permitted range.")

    if abs(reported_max - max_marks) > 0.01:
        raise ValueError("Maximum marks do not match the question.")

    if not 0 <= confidence <= 1:
        raise ValueError("Confidence must be between zero and one.")

    if not isinstance(result["is_correct"], bool):
        raise ValueError("is_correct must be a boolean.")

    if not isinstance(result["needs_teacher_check"], bool):
        raise ValueError("needs_teacher_check must be a boolean.")

    if not isinstance(result["reason"], str) or not result["reason"].strip():
        raise ValueError("Evaluator reason must be a non-empty string.")

    if not isinstance(criteria, list) or not criteria:
        raise ValueError("Criteria must be a non-empty list.")

    available_total = 0.0
    awarded_total = 0.0

    for criterion in criteria:
        if not isinstance(criterion, dict):
            raise ValueError("Each criterion must be an object.")

        required_criterion_fields = {
            "concept",
            "marks_available",
            "marks_awarded",
            "reason",
        }

        if not required_criterion_fields.issubset(criterion):
            raise ValueError("A criterion is missing required fields.")

        if not isinstance(criterion["concept"], str):
            raise ValueError("Criterion concept must be a string.")

        if not isinstance(criterion["reason"], str):
            raise ValueError("Criterion reason must be a string.")

        available = _finite_number(
            criterion["marks_available"],
            "marks_available",
        )
        awarded = _finite_number(
            criterion["marks_awarded"],
            "marks_awarded",
        )

        if available <= 0 or awarded < 0 or awarded > available:
            raise ValueError("Invalid criterion marks.")

        available_total += available
        awarded_total += awarded

    if abs(available_total - max_marks) > 0.01:
        raise ValueError(
            "Criterion available marks do not equal maximum marks."
        )

    if abs(awarded_total - score) > 0.01:
        raise ValueError("Criterion marks do not add up to the score.")

    teacher_check = (
        result["needs_teacher_check"] or confidence < 0.80
    )

    passed = (
        result["is_correct"]
        and abs(score - max_marks) <= 0.01
        and not teacher_check
    )

    return {
        "passed": passed,
        "reason": result["reason"],
        "verified_by": "llm_rubric",
        "score": score,
        "max_marks": max_marks,
        "criteria": criteria,
        "confidence": confidence,
        "needs_teacher_check": teacher_check,
    }


def _finite_number(value: Any, field_name: str) -> float:
    """Convert a value to a finite number, rejecting booleans and invalid values."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not boolean.")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")

    return number


def _verify_numerical(
    question_text: str,
    answer: str,
) -> dict[str, Any]:
    """Verify supported two-resistor parallel-resistance questions."""

    if "parallel" not in question_text.casefold():
        return {
            "passed": False,
            "reason": (
                "No deterministic numerical verifier is available "
                "for this question."
            ),
            "verified_by": "none",
            "needs_teacher_check": True,
            "retryable": False,
        }

    # Correctly match both "6 ohm" and "6 Ω", including decimal values.
    resistor_pattern = r"(\d+(?:\.\d+)?)\s*(?:ohm\b|Ω)"

    resistor_values = re.findall(
        resistor_pattern,
        question_text,
        flags=re.IGNORECASE,
    )

    if len(resistor_values) != 2:
        return {
            "passed": False,
            "reason": "Could not identify exactly two resistor values.",
            "verified_by": "none",
            "needs_teacher_check": True,
            "retryable": False,
        }

    r1, r2 = map(float, resistor_values)

    if r1 <= 0 or r2 <= 0:
        return {
            "passed": False,
            "reason": "Resistor values must be positive.",
            "verified_by": "symbolic",
            "needs_teacher_check": True,
            "retryable": False,
        }

    expected = (r1 * r2) / (r1 + r2)

    answer_values = re.findall(
        resistor_pattern,
        answer,
        flags=re.IGNORECASE,
    )

    if not answer_values:
        return {
            "passed": False,
            "reason": (
                "Could not identify a resistance value in the generated answer."
            ),
            "verified_by": "symbolic",
            "needs_teacher_check": True,
        }

    generated = float(answer_values[-1])
    passed = math.isfinite(generated) and abs(generated - expected) < 0.01

    return {
        "passed": passed,
        "reason": (
            f"Generated answer matches the expected parallel resistance "
            f"of {expected:g} ohm."
            if passed
            else (
                f"Expected parallel resistance {expected:g} ohm, "
                f"but generated {generated:g} ohm."
            )
        ),
        "verified_by": "symbolic",
        "needs_teacher_check": not passed,
    }
