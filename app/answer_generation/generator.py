
from typing import Any

from app.answer_generation.schema import GeneratedSolution
from app.llm.client import call_groq_json


SUPPORTED_QUESTION_TYPES = {
    "mcq",
    "short",
    "numerical",
    "long",
    "diagram",
}


def generate_answer(
    question: dict[str, Any],
    retry_reason: str | None = None,
) -> dict[str, Any]:
    """
    Generate a candidate solution for one structured question.

    Verification is intentionally handled separately.

    If retry_reason is provided, it is included in the prompt so the
    LLM can correct a previously rejected solution.
    """

    question_number = str(question.get("number", ""))
    question_text = str(question.get("text", "")).strip()
    question_type = str(question.get("type", "")).lower()
    marks = question.get("marks", 0)
    options = question.get("options")

    # Return a structured failure instead of asking the model to solve
    # an unsupported or empty question.
    if question_type not in SUPPORTED_QUESTION_TYPES:
        return _build_result(
            question_number,
            GeneratedSolution(
                answer="Unsupported question type.",
                steps=[],
                mark_split=[],
                common_mistakes=[],
            ),
        )

    if not question_text:
        return _build_result(
            question_number,
            GeneratedSolution(
                answer="Question text is missing.",
                steps=[],
                mark_split=[],
                common_mistakes=[],
            ),
        )

    system_prompt = """
You are an expert school examination solver.

Generate a correct, clear candidate solution appropriate to the
question type and marks available.

Return ONLY a valid JSON object with exactly these fields:

{
  "answer": "final answer",
  "steps": ["step 1", "step 2"],
  "mark_split": [
    {"marks": 1, "for": "what earns the mark"}
  ],
  "common_mistakes": [
    {"wrong": "common wrong answer", "why": "why it is wrong"}
  ]
}

Rules:
- Return valid JSON only, without Markdown code fences.
- Do not include text outside the JSON object.
- Do not guess if the question lacks enough information.
- Make the mark split appropriate to the question's total marks.
- The answer and steps must be consistent with each other.
- Do not claim that the answer has been independently verified.
- Do not invent information absent from the question.
"""

    user_prompt = f"""
Question number: {question_number}
Question type: {question_type}
Total marks: {marks}

Question:
{question_text}

Options:
{options if options else "No options provided."}
"""

    if retry_reason:
        user_prompt += f"""

The previous candidate solution failed verification.

Verification feedback:
{retry_reason}

Generate a corrected candidate solution. Address the feedback carefully.
"""

    try:
        # call_groq_json returns a Python dictionary, not a JSON string.
        raw_response = call_groq_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if not isinstance(raw_response, dict):
            raise ValueError("The LLM response must be a JSON object.")

        solution = GeneratedSolution.model_validate(raw_response)

    except Exception as exc:
        # Never treat malformed model output as a valid answer.
        return {
            "question_number": question_number,
            "answer": "",
            "steps": [],
            "mark_split": [],
            "common_mistakes": [],
            "confidence": 0.0,
            "verified_by": "none",
            "needs_teacher_check": True,
            "generation_error": (
                f"Answer generation failed: {type(exc).__name__}: {exc}"
            ),
        }

    return _build_result(question_number, solution)


def _build_result(
    question_number: str,
    solution: GeneratedSolution,
) -> dict[str, Any]:
    """Convert a validated candidate into the existing solution format."""

    return {
        "question_number": question_number,
        "answer": solution.answer,
        "steps": solution.steps,
        "mark_split": [
            item.model_dump(by_alias=True)
            for item in solution.mark_split
        ],
        "common_mistakes": [
            item.model_dump()
            for item in solution.common_mistakes
        ],
        # A generated answer is not independently verified.
        "confidence": 0.0,
        "verified_by": "none",
        "needs_teacher_check": True,
    }
