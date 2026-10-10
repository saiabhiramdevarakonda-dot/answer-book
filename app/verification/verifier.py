
import json
import math
import re
from typing import Any

from app.llm.gemini import GeminiProvider


def verify_solution(
    question: dict[str, Any],
    solution: dict[str, Any],
    expected_answer: str | None = None,
) -> dict[str, Any]:
    """Verify a generated solution using the appropriate method."""

    question_type = str(question.get("type", "")).lower()
    question_text = str(question.get("text", "")).lower()
    answer = str(solution.get("answer", "")).strip()

    if not answer:
        return {
            "passed": False,
            "reason": "No answer was generated.",
            "verified_by": "none",
        }

    if question_type == "numerical":
        return _verify_numerical(question_text, answer.lower())

    if question_type == "mcq":
        return _verify_mcq(question, answer, expected_answer)

    if question_type in {"short", "long", "diagram"}:
        return _verify_descriptive(question, solution)

    return {
        "passed": False,
        "reason": "Unsupported question type.",
        "verified_by": "none",
    }


def _verify_mcq(
    question: dict[str, Any],
    answer: str,
    expected_answer: str | None,
) -> dict[str, Any]:
    """Compare an MCQ answer with the supplied answer key."""

    if expected_answer is None:
        return {
            "passed": False,
            "reason": "Cannot verify MCQ without an answer key.",
            "verified_by": "none",
        }

    options = question.get("options", {})
    expected = expected_answer.strip().upper()

    if not isinstance(options, dict) or expected not in options:
        return {
            "passed": False,
            "reason": "The expected answer key option is unavailable.",
            "verified_by": "answer_key",
        }

    generated = answer.strip().lower()
    expected_text = str(options[expected]).strip().lower()

    matches = (
        generated == expected.lower()
        or generated.startswith(f"{expected.lower()}.")
        or generated.startswith(f"{expected.lower()})")
        or generated == expected_text
    )

    return {
        "passed": matches,
        "reason": (
            f"Generated answer matches answer key option {expected}."
            if matches
            else (
                f"Generated answer does not match answer key option "
                f"{expected}."
            )
        ),
        "verified_by": "answer_key",
    }


def _verify_descriptive(
    question: dict[str, Any],
    solution: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a descriptive answer against an explicit marking rubric."""

    marks = question.get("marks", 0)

    try:
        max_marks = float(marks)
    except (TypeError, ValueError):
        max_marks = 0.0

    if not math.isfinite(max_marks) or max_marks <= 0:
        return {
            "passed": False,
            "reason": "Valid maximum marks are required for rubric evaluation.",
            "verified_by": "none",
            "needs_teacher_check": True,
        }

    prompt = f"""
You are a strict but fair examination answer evaluator.

Evaluate the candidate answer against the question and its maximum marks.
Accept different wording when it expresses the same correct meaning.
Do not award marks for unsupported claims or incorrect statements.

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

Return ONLY valid JSON in this format:
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
- Score must be between 0 and {max_marks}.
- Criterion available marks must add up to {max_marks}.
- The sum of awarded criterion marks must equal score.
- Do not exceed the maximum marks.
- Confidence must be between 0 and 1.
- Use partial credit where appropriate.
- If the question is ambiguous, the answer is contradictory, or correctness
  cannot be assessed reliably, set needs_teacher_check to true.
- Do not claim independent mathematical proof merely because another LLM agrees.
"""

    try:
        raw = GeminiProvider().generate(prompt)
        result = _parse_json(raw)

        required_fields = {
            "score",
            "max_marks",
            "criteria",
            "is_correct",
            "confidence",
            "reason",
            "needs_teacher_check",
        }

        if not required_fields.issubset(result):
            raise ValueError("Evaluator response is missing required fields.")

        score = _finite_number(result["score"], "score")
        reported_max = _finite_number(result["max_marks"], "max_marks")
        confidence = _finite_number(result["confidence"], "confidence")
        criteria = result["criteria"]

        if not isinstance(criteria, list) or not criteria:
            raise ValueError("Criteria must be a non-empty list.")

        if not 0 <= score <= max_marks:
            raise ValueError("Score is outside the permitted range.")

        if abs(reported_max - max_marks) > 0.01:
            raise ValueError("Maximum marks do not match the question.")

        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1.")

        if not isinstance(result["is_correct"], bool):
            raise ValueError("is_correct must be a boolean.")

        if not isinstance(result["needs_teacher_check"], bool):
            raise ValueError("needs_teacher_check must be a boolean.")

        if not isinstance(result["reason"], str) or not result["reason"].strip():
            raise ValueError("Evaluator reason must be a non-empty string.")

        available_total = 0.0
        awarded_total = 0.0

        for criterion in criteria:
            if not isinstance(criterion, dict):
                raise ValueError("Each criterion must be an object.")

            if not {"concept", "marks_available", "marks_awarded", "reason"}.issubset(
                criterion
            ):
                raise ValueError("A criterion is missing required fields.")

            if not isinstance(criterion["concept"], str):
                raise ValueError("Criterion concept must be a string.")

            if not isinstance(criterion["reason"], str):
                raise ValueError("Criterion reason must be a string.")

            available = _finite_number(
                criterion["marks_available"], "marks_available"
            )
            awarded = _finite_number(
                criterion["marks_awarded"], "marks_awarded"
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

    except Exception as exc:
        return {
            "passed": False,
            "reason": f"Rubric evaluation unavailable or invalid: {exc}",
            "verified_by": "none",
            "needs_teacher_check": True,
        }


def _finite_number(value: Any, field_name: str) -> float:
    """Convert a value to a finite float or reject it."""

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be numeric, not boolean.")

    number = float(value)

    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite.")

    return number


def _parse_json(raw: str) -> dict[str, Any]:
    """Parse JSON, including responses wrapped in Markdown fences."""

    if not isinstance(raw, str):
        raise ValueError("Evaluator response must be a string.")

    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    result = json.loads(cleaned)

    if not isinstance(result, dict):
        raise ValueError("Evaluator response must be a JSON object.")

    return result


def _verify_numerical(
    question_text: str,
    answer: str,
) -> dict[str, Any]:
    """Verify supported two-resistor parallel-resistance questions."""

    if "parallel" not in question_text:
        return {
            "passed": False,
            "reason": (
                "No deterministic numerical verifier is available "
                "for this question yet."
            ),
            "verified_by": "none",
        }

    resistor_values = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:ohm|Ω)",
        question_text,
        flags=re.IGNORECASE,
    )

    if len(resistor_values) != 2:
        return {
            "passed": False,
            "reason": (
                "Could not identify exactly two resistor values "
                "for parallel-resistance verification."
            ),
            "verified_by": "none",
        }

    r1, r2 = map(float, resistor_values)

    if r1 <= 0 or r2 <= 0:
        return {
            "passed": False,
            "reason": "Resistor values must be positive.",
            "verified_by": "symbolic",
        }

    expected = (r1 * r2) / (r1 + r2)

    answer_values = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:ohm|Ω)",
        answer,
        flags=re.IGNORECASE,
    )

    if not answer_values:
        return {
            "passed": False,
            "reason": (
                "Could not identify a resistance value "
                "in the generated answer."
            ),
            "verified_by": "symbolic",
        }

    generated = float(answer_values[-1])

    if not math.isfinite(generated):
        return {
            "passed": False,
            "reason": "The generated resistance value is invalid.",
            "verified_by": "symbolic",
        }

    if abs(generated - expected) < 0.01:
        return {
            "passed": True,
            "reason": (
                f"The generated answer matches the expected "
                f"parallel resistance of {expected:g} ohm."
            ),
            "verified_by": "symbolic",
        }

    return {
        "passed": False,
        "reason": (
            f"The generated answer is {generated:g} ohm, but the "
            f"expected parallel resistance is {expected:g} ohm."
        ),
        "verified_by": "symbolic",
    }
