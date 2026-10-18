"""Planner Node: Analyzes the question, checks the database cache, and routes appropriately."""

from typing import Any, Dict
from app.generate.state import QuestionState
from app.tools.hash_utils import generate_question_signature
from app.tools.question_lookup_tool import lookup_cached_question


def planner_node(state: QuestionState) -> Dict[str, Any]:
    """Inspect question type, options, and text to route to the appropriate solver.

    Routes:
    - 'cached': If an exact question signature exists in the cache (bypasses LLM)
    - 'mcq': If options exist or the question type is 'mcq'
    - 'math': If the type is 'numerical' or the question contains mathematical cues
    - 'theory': For conceptual definitions and descriptive questions
    """
    q_num = str(state.get("number") or "1")
    text = str(state.get("text") or "")
    marks = int(state.get("marks") or 1)

    metadata = state.get("metadata") or {}
    board = metadata.get("board", "CBSE")
    class_name = metadata.get("class", metadata.get("class_name", ""))

    # 1. Check the question cache (exact signature match).
    if text:
        sig = generate_question_signature(text, marks, board, class_name)

        try:
            cached_solution = lookup_cached_question(
                sig,
                target_question_number=q_num,
            )
        except Exception as exc:
            # A cache failure should not prevent the solver from handling the question.
            print(
                f"[Question cache] Lookup failed for Q{q_num}: "
                f"{type(exc).__name__}: {exc}"
            )
            cached_solution = None

        if cached_solution is not None:
            print(
                f"    [QUESTION CACHE HIT] Reusing cached solution "
                f"for Q{q_num} (sig: {sig[:10]}...)"
            )
            return {
                "selected_route": "cached",
                "solution": cached_solution,
            }

    # 2. Check for a multiple-choice question.
    q_type = str(state.get("type") or "").lower()
    options = state.get("options")
    lower_text = text.lower()

    if options or q_type == "mcq":
        return {"selected_route": "mcq"}

    # 3. Check for a numerical / mathematical problem.
    math_cues = [
        "calculate",
        "find the value",
        "evaluate",
        "expansion of",
        "solve",
        "probability",
    ]
    math_symbols = [
        "²",
        "³",
        "√",
        "+",
        "=",
        "×",
        "π",
        "ap whose",
        "0.9̄",
        "m s⁻²",
    ]

    is_numerical_type = q_type == "numerical"
    has_math_cues = any(cue in lower_text for cue in math_cues)
    has_math_symbols = any(symbol in lower_text for symbol in math_symbols)

    if is_numerical_type or (has_math_cues and has_math_symbols):
        return {"selected_route": "math"}

    # 4. Default to descriptive / theory question.
    return {"selected_route": "theory"}
