"""Normalizer Node: Validates and coerces solver output into the frozen API contract schema."""

from typing import Dict, Any, List
from app.generate.state import QuestionState, SolutionContract, MarkSplitItem, CommonMistakeItem


def normalizer_node(state: QuestionState) -> Dict[str, Any]:
    """Sanitizes and normalizes the raw solver output into the exact frozen API contract."""
    # Fast path: if the question was resolved via cache, preserve the cached solution
    if state.get("solution"):
        return {"solution": state["solution"]}

    raw = state.get("raw_solution") or {}
    q_num = str(state.get("number") or raw.get("question_number") or "")
    total_marks = int(state.get("marks") or 1)

    # 1. Answer (string)
    answer = str(raw.get("answer") or "").strip()

    # 2. Steps (List[str])
    raw_steps = raw.get("steps")
    if isinstance(raw_steps, list):
        steps = [str(s).strip() for s in raw_steps if str(s).strip()]
    elif isinstance(raw_steps, str):
        steps = [line.strip() for line in raw_steps.split("\n") if line.strip()]
    else:
        steps = ["Step-by-step solution derivation."]

    # 3. Mark Split (List[{"marks": int, "for": str}])
    raw_split = raw.get("mark_split")
    mark_split: List[MarkSplitItem] = []
    if isinstance(raw_split, list):
        for item in raw_split:
            if isinstance(item, dict):
                m = int(item.get("marks") or 1)
                f = str(item.get("for") or item.get("description") or "correct step")
                mark_split.append({"marks": m, "for": f})

    # Ensure mark_split is non-empty and well-formed
    if not mark_split:
        mark_split = [{"marks": total_marks, "for": "complete correct solution"}]

    # 4. Common Mistakes (List[{"wrong": str, "why": str}])
    raw_mistakes = raw.get("common_mistakes")
    common_mistakes: List[CommonMistakeItem] = []
    if isinstance(raw_mistakes, list):
        for item in raw_mistakes:
            if isinstance(item, dict):
                w = str(item.get("wrong") or "Incorrect answer")
                y = str(item.get("why") or "Common conceptual mistake")
                common_mistakes.append({"wrong": w, "why": y})

    # 5. Confidence Score (float between 0.0 and 1.0)
    try:
        confidence = float(raw.get("confidence", 0.90))
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.90

    # 6. Preserve the verifier's result only when verification passed.
    verification = state.get("verification") or {}
    if verification.get("passed"):
        verified_by = str(verification.get("verified_by") or "none")
    else:
        verified_by = "none"

    # 7. Failed verification always requires teacher review.
    needs_teacher_check = (
        not verification.get("passed", False)
        or bool(verification.get("needs_teacher_check", True))
        or bool(raw.get("needs_teacher_check", False))
        or confidence < 0.70
    )


    # Assemble final contract dictionary
    solution: SolutionContract = {
        "question_number": q_num,
        "answer": answer,
        "steps": steps,
        "mark_split": mark_split,
        "common_mistakes": common_mistakes,
        "confidence": confidence,
        "verified_by": verified_by,
        "needs_teacher_check": needs_teacher_check,
    }

    return {"solution": solution}
