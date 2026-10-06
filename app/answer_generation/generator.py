from typing import Any

from app.answer_generation.schema import GeneratedSolution
from app.llm.gemini import GeminiProvider


def generate_answer(
    question: dict[str, Any],
    retry_reason: str | None = None,
) -> dict[str, Any]:
    """
    Generate a candidate solution for one structured question.

    Verification is intentionally not performed here.

    If retry_reason is provided, it is included in the prompt so
    the LLM can correct the previous failed solution.
    """

    question_number = question.get("number", "")
    question_text = question.get("text", "")
    question_type = question.get("type", "")
    marks = question.get("marks", 0)
    options = question.get("options")

    provider = GeminiProvider()

    prompt = f"""
You are solving a school examination question.

Question number: {question_number}
Question type: {question_type}
Marks: {marks}

Question:
{question_text}
"""

    if options:
        prompt += f"""
Options:
{options}
"""

    if retry_reason:
        prompt += f"""
Previous solution failed verification.

Reason:
{retry_reason}

Generate a corrected solution.
"""

    prompt += """
Provide a clear candidate solution.

Return ONLY valid JSON with exactly these fields:

{
  "answer": "final answer",
  "steps": [
    "step 1",
    "step 2"
  ],
  "mark_split": [
    {
      "marks": 1,
      "for": "what earns the mark"
    }
  ],
  "common_mistakes": [
    {
      "wrong": "common wrong answer",
      "why": "why it is wrong"
    }
  ]
}

Rules:
- Return valid JSON only.
- Do not use markdown code fences.
- Do not include any text outside the JSON.
- Do not guess when the question does not contain enough information.
"""

    if question_type not in {"mcq", "short", "numerical", "long", "diagram"}:
        solution = GeneratedSolution(
            answer="Unsupported question type.",
            steps=[],
            mark_split=[],
            common_mistakes=[],
        )

    else:
        raw_response = provider.generate(prompt)

        try:
            solution = GeneratedSolution.model_validate_json(
                raw_response
            )

        except Exception:
            try:
                cleaned_response = raw_response.strip()

                if cleaned_response.startswith("```json"):
                    cleaned_response = cleaned_response[
                        len("```json"):
                    ].strip()

                elif cleaned_response.startswith("```"):
                    cleaned_response = cleaned_response[
                        len("```"):
                    ].strip()

                if cleaned_response.endswith("```"):
                    cleaned_response = cleaned_response[
                        :-len("```")
                    ].strip()

                solution = GeneratedSolution.model_validate_json(
                    cleaned_response
                )

            except Exception:
                solution = GeneratedSolution(
                    answer=raw_response,
                    steps=[],
                    mark_split=[],
                    common_mistakes=[],
                )

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
        "confidence": 0.0,
        "verified_by": "none",
        "needs_teacher_check": True,
    }