"""MCQ Solver Node: solves multiple-choice and assertion-reason questions."""

from typing import Any, Dict

from app.generate.state import QuestionState
from app.llm.client import call_groq_json


def _format_options(options: Any) -> str:
    """Format options with explicit labels so the model can map text to letters."""
    if isinstance(options, dict):
        return "\n".join(
            f"({str(letter).strip().upper()}) {str(option).strip()}"
            for letter, option in options.items()
        )
    if isinstance(options, list):
        return "\n".join(
            f"({chr(ord('A') + index)}) {str(option).strip()}"
            for index, option in enumerate(options[:26])
        )
    return "No options list provided."


def mcq_solver_node(state: QuestionState) -> Dict[str, Any]:
    """Evaluate options and explain why distractors are incorrect."""
    q_num = state.get("number", "")
    text = state.get("text", "")
    marks = state.get("marks", 1)
    options_formatted = _format_options(state.get("options") or [])
    metadata = state.get("metadata", {})
    subject = metadata.get("subject", "Science / Mathematics")
    grade = metadata.get("class", "Class 9")
    board = metadata.get("board", "CBSE")
    retry_reason = str(state.get("retry_reason") or "").strip()
    retry_section = (
        "\nPrevious candidate failed verification. Re-evaluate the reasoning "
        "and correct the mistake without guessing:\n" + retry_reason
        if retry_reason
        else ""
    )

    system_prompt = f"""You are an expert {board} {subject} teacher for {grade} students.
Your goal is to solve the multiple-choice question with high accuracy and explain student misconceptions.

Follow these strict rules:

1. The 'answer' must include BOTH the letter and text of the correct choice (e.g. '(D) 25x² + 20xy + 4y²').
2. In 'steps', provide a systematic explanation:
   - For Assertion-Reason questions:
     (a) Evaluate the Assertion as true or false.
     (b) Evaluate the Reason as true or false.
     (c) Decide whether the Reason logically explains WHY the Assertion is true.
     (d) Match all three findings against the exact option text.
   - Do NOT infer causation from topic similarity.
   - A Reason can be true without correctly explaining the Assertion.
   - Explicitly distinguish a logical explanation from a related fact.
   - If the explanation relationship is ambiguous, lower confidence and set 'needs_teacher_check' to true.
   - For standard MCQs, state the governing concept and explain why the selected option is best supported.
3. In 'mark_split', break down the {marks} mark(s). The sum of 'marks' must equal exactly {marks}.
4. In 'common_mistakes', inspect the INCORRECT options (distractors) and explain the exact trap or error that leads students to choose them.
5. Provide a 'confidence' score between 0.0 and 1.0.
6. Return ONLY valid JSON matching the schema below.
"""

    user_prompt = f"""Question Number: {q_num}
Total Marks: {marks}
Question:
{text}

Options:
{options_formatted}
{retry_section}

Output JSON Schema:
{{
  "question_number": "{q_num}",
  "answer": "(<Letter>) <Option Text>",
  "steps": ["<step 1>", "<step 2>", "..."],
  "mark_split": [
    {{"marks": {marks}, "for": "selecting correct option with proper justification"}}
  ],
  "common_mistakes": [
    {{"wrong": "(<Letter>) <Wrong Option>", "why": "<why student falls for this trap>"}}
  ],
  "confidence": <float 0.0 to 1.0>,
  "verified_by": "none",
  "needs_teacher_check": false
}}"""

    result = call_groq_json(system_prompt, user_prompt)
    result["question_number"] = q_num
    return {"raw_solution": result}
