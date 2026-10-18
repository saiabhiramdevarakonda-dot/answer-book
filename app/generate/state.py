"""LangGraph State definitions for the Question Solve Engine.

Defines the exact structure of data passed between nodes in the graph.
"""

from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict


# --- Helper Types matching the Frozen API Contract ---

class CommonMistakeItem(TypedDict):
    wrong: str
    why: str


# Using functional syntax because 'for' is a reserved keyword in Python
MarkSplitItem = TypedDict(
    "MarkSplitItem",
    {
        "marks": int,
        "for": str,
    },
)


class SolutionContract(TypedDict):
    """The frozen output contract as specified in Section 3 of the brief."""
    question_number: str
    answer: str
    steps: List[str]
    mark_split: List[MarkSplitItem]
    common_mistakes: List[CommonMistakeItem]
    confidence: float
    verified_by: str
    needs_teacher_check: bool


# --- Main LangGraph States ---

class QuestionState(TypedDict, total=False):
    """State schema for processing an individual question in LangGraph."""

    # 1. Inputs (from Ingestion / static JSON)
    number: str
    section: str
    text: str
    marks: int
    type: str
    options: Optional[List[str]]
    has_figure: bool
    page: int
    choice_group: Optional[str]
    chapter: Optional[str]
    metadata: Dict[str, Any]  # e.g., {"subject": "Mathematics", "class": "Class 9", "board": "CBSE"}

    # 2. Intermediate Variables (set by internal graph nodes)
    selected_route: str       # "math" | "mcq" | "theory" | "cached" (decided by Planner)
    raw_solution: Optional[Dict[str, Any]]  # Raw output from the specialized solver
    verification: Optional[Dict[str, Any]]
    expected_answer: Optional[str]
    retry_count: int
    retry_reason: Optional[str]
    # 3. Final Normalized Output
    solution: Optional[SolutionContract]    # Sanitized final object matching API contract


class PaperState(TypedDict, total=False):
    """State schema for orchestrating an entire paper with multiple questions."""

    paper_id: str
    fingerprint: str
    status: str
    paper_metadata: Dict[str, Any]
    questions: List[Dict[str, Any]]
    total_questions: int
    current_index: int
    solutions: List[SolutionContract]
