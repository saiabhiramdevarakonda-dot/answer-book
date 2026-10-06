import re
from typing import Any


def verify_solution(
    question: dict[str, Any],
    solution: dict[str, Any],
    expected_answer: str | None = None,
) -> dict[str, Any]:
    """
    Verify a generated solution.

    Numerical verification is handled deterministically for
    supported patterns. MCQs can be checked against an answer key.
    Other question types currently receive structural verification.
    """

    question_type = question.get("type", "").lower()
    question_text = question.get("text", "").lower()
    answer = str(solution.get("answer", "")).strip().lower()

    if not answer:
        return {
            "passed": False,
            "reason": "No answer was generated.",
            "verified_by": "none",
        }

    if question_type == "numerical":
        return _verify_numerical(question_text, answer)

    if question_type == "mcq":
        if expected_answer is None:
            return {
                "passed": True,
                "reason": "MCQ answer generated; no answer key provided.",
                "verified_by": "none",
            }

        options = question.get("options", {})

        if not isinstance(options, dict):
            return {
                "passed": False,
                "reason": "MCQ options are missing or have an invalid format.",
                "verified_by": "answer_key",
            }

        expected_option = expected_answer.strip().upper()

        if expected_option not in options:
            return {
                "passed": False,
                "reason": (
                    f"Answer key option {expected_option} was not found "
                    "in the extracted question options."
                ),
                "verified_by": "answer_key",
            }

        expected_text = str(options[expected_option]).strip().lower()
        generated_answer = answer.strip().lower()

        option_matches = (
            generated_answer == expected_option.lower()
            or generated_answer.startswith(f"{expected_option.lower()}.")
            or generated_answer.startswith(f"{expected_option.lower()})")
            or generated_answer == expected_text
        )

        if option_matches:
            return {
                "passed": True,
                "reason": (
                    f"Generated answer matches answer key option "
                    f"{expected_option}."
                ),
                "verified_by": "answer_key",
            }

        return {
            "passed": False,
            "reason": (
                f"Generated answer does not match answer key option "
                f"{expected_option}."
            ),
            "verified_by": "answer_key",
        }
    if question_type in {"short", "long", "diagram"}:
        return {
            "passed": True,
            "reason": "Initial structural verification passed.",
            "verified_by": "none",
        }

    return {
        "passed": False,
        "reason": "Unsupported question type.",
        "verified_by": "none",
    }


def _verify_numerical(
    question_text: str,
    answer: str,
) -> dict[str, Any]:
    """
    Verify supported numerical patterns.

    Currently supports two-resistor parallel-resistance questions.
    """

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

    r1 = float(resistor_values[0])
    r2 = float(resistor_values[1])
    expected = (r1 * r2) / (r1 + r2)

    answer_values = re.findall(
        r"(\d+(?:\.\d+)?)\s*(?:ohm|Ω)",
        answer,
    )

    if not answer_values:
        return {
            "passed": False,
            "reason": "Could not identify a resistance value in the generated answer.",
            "verified_by": "symbolic",
        }

    generated = float(answer_values[-1])

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