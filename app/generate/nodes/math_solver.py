"""Math Solver Node: Solves numerical, algebraic, and calculation questions using Groq."""

from typing import Dict, Any
from app.generate.state import QuestionState
from app.llm.client import call_groq_json


def math_solver_node(state: QuestionState) -> Dict[str, Any]:
    """Solves numerical and mathematical problems step-by-step with formulas and units."""
    q_num = state.get("number", "")
    text = state.get("text", "")
    marks = state.get("marks", 1)
    metadata = state.get("metadata", {})
    subject = metadata.get("subject", "Mathematics")
    grade = metadata.get("class", "Class 9")
    board = metadata.get("board", "CBSE")
    retry_reason = str(state.get("retry_reason") or "").strip()
    retry_section = (
        "\nPrevious candidate failed verification. Recheck the reasoning and correct this issue:\n" + retry_reason
        if retry_reason
        else ""
    )

    system_prompt = f"""You are a master {board} {subject} teacher for {grade} students.
Your goal is to provide a complete, mathematically rigorous, step-by-step solution for the numerical question.

Follow these strict rules:
1. Show every intermediate equation, formula, and calculation step clearly.
2. The 'answer' field must be concise (the final numerical value with units, e.g., '14161' or '49 N').
3. The 'steps' field must be an array of clear, logical derivation steps.
4. The 'mark_split' field must break down how the {marks} marks are earned (e.g. formula, substitution, final answer).
   CRITICAL: The sum of 'marks' across all items in 'mark_split' must equal exactly {marks}.
5. The 'common_mistakes' field must identify 2 realistic calculation slips, missing cross-terms, or unit errors students usually make.
6. Provide a 'confidence' score between 0.0 and 1.0.
7. Return ONLY valid JSON matching the schema below.
"""

    user_prompt = f"""Question Number: {q_num}
Total Marks: {marks}
Question:
{text}
{retry_section}

Output JSON Schema:
{{
  "question_number": "{q_num}",
  "answer": "<concise final answer with unit>",
  "steps": ["<step 1>", "<step 2>", ...],
  "mark_split": [
    {{"marks": <int>, "for": "<specific criterion>"}}
  ],
  "common_mistakes": [
    {{"wrong": "<wrong answer>", "why": "<conceptual/calculation reason>"}}
  ],
  "confidence": <float 0.0 to 1.0>,
  "verified_by": "none",
  "needs_teacher_check": false
}}"""

    result = call_groq_json(system_prompt, user_prompt)
    result["question_number"] = q_num
    return {"raw_solution": result}
