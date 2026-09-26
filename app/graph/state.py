from typing import Any, TypedDict


class SolveState(TypedDict, total=False):
    question: dict[str, Any]
    route: str
    solution: dict[str, Any]
    verification: dict[str, Any]
    retry_count: int
    retry_reason: str
    expected_answer: str