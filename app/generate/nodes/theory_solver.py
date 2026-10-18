"""Theory Solver Node: Solves short and long descriptive questions using Groq."""

from typing import Dict, Any
from app.generate.state import QuestionState
from app.llm.client import call_groq_json


def theory_solver_node(state: QuestionState) -> Dict[str, Any]:
    """Generates structured, keyword-rich answers aligned with board marking schemes."""
    q_num = state.get("number", "")
    text = state.get("text", "")
    marks = state.get("marks", 2)
    metadata = state.get("metadata", {})
    subject = metadata.get("subject", "Science / Mathematics")
    grade = metadata.get("class", "Class 9")
    board = metadata.get("board", "CBSE")
    retry_reason = str(state.get("retry_reason") or "").strip()
    retry_section = (
        "\nPrevious candidate failed verification. Recheck the answer and correct this issue:\n" + retry_reason
        if retry_reason
        else ""
    )

    system_prompt = f"""You are a senior {board} {subject} examiner for {grade} students.
Your goal is to provide a complete, clear, and structured answer for this descriptive question.

Follow these strict rules:
1. Never write walls of text. Use structured, numbered points in 'steps'.
2. Emphasize mandatory curriculum keywords and scientific definitions needed to earn full marks.
3. The 'answer' field must provide a concise 1-2 sentence core summary of the answer.
4. The 'mark_split' field must break down how all {marks} marks are awarded across the key points.
   CRITICAL: The sum of 'marks' in 'mark_split' must equal exactly {marks}.
5. The 'common_mistakes' field must identify 2 common vague or incomplete answers that cause students to lose marks.
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
  "answer": "<concise summary of the solution>",
  "steps": ["<key point 1 with technical keywords>", "<key point 2>", ...],
  "mark_split": [
    {{"marks": <int>, "for": "<specific marking point criteria>"}}
  ],
  "common_mistakes": [
    {{"wrong": "<common vague/incomplete answer>", "why": "<why it loses marks>"}}
  ],
  "confidence": <float 0.0 to 1.0>,
  "verified_by": "none",
  "needs_teacher_check": false
}}"""

    result = call_groq_json(system_prompt, user_prompt)
    result["question_number"] = q_num
    return {"raw_solution": result}
