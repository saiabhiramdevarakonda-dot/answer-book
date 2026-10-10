from app.graph.solve.nodes import generate_node, verify_node
from app.graph.solve.graph import build_solve_graph
from app.llm.gemini import GeminiProvider


FAKE_GEMINI_RESPONSE = """
{
    "answer": "4 ohm",
    "steps": [
        "Use the parallel resistance formula.",
        "Substitute 6 ohm and 12 ohm.",
        "Calculate the result."
    ],
    "mark_split": [
        {
            "marks": 1,
            "for": "Correct formula"
        },
        {
            "marks": 1,
            "for": "Correct substitution"
        },
        {
            "marks": 1,
            "for": "Correct final answer"
        }
    ],
    "common_mistakes": [
        {
            "wrong": "18 ohm",
            "why": "The resistors were incorrectly treated as series."
        }
    ]
}
"""


def test_generate_node(monkeypatch):
    monkeypatch.setattr(
        GeminiProvider,
        "generate",
        lambda self, prompt: FAKE_GEMINI_RESPONSE,
    )

    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "retry_count": 0,
    }

    result = generate_node(state)

    assert "solution" in result
    assert result["solution"]["answer"] == "4 ohm"


def test_verify_node():
    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "solution": {
            "question_number": "3(b)",
            "answer": "4 ohm",
            "steps": [],
            "mark_split": [],
            "common_mistakes": [],
        },
        "retry_count": 0,
    }

    result = verify_node(state)

    assert "verification" in result
    assert result["verification"]["passed"] is True


def test_solve_graph(monkeypatch):
    monkeypatch.setattr(
        GeminiProvider,
        "generate",
        lambda self, prompt: FAKE_GEMINI_RESPONSE,
    )

    graph = build_solve_graph()

    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "retry_count": 0,
    }

    result = graph.invoke(state)

    assert "solution" in result
    assert "verification" in result
    assert result["solution"]["answer"] == "4 ohm"
    assert result["verification"]["passed"] is True
def test_solve_graph_retries_when_verification_fails(monkeypatch):
    responses = [
        """
        {
            "answer": "18 ohm",
            "steps": ["Incorrect calculation."],
            "mark_split": [],
            "common_mistakes": []
        }
        """,
        """
        {
            "answer": "4 ohm",
            "steps": [
                "Use the parallel resistance formula.",
                "Substitute 6 ohm and 12 ohm.",
                "Calculate the result."
            ],
            "mark_split": [
                {"marks": 1, "for": "Correct formula"},
                {"marks": 1, "for": "Correct substitution"},
                {"marks": 1, "for": "Correct final answer"}
            ],
            "common_mistakes": [
                {
                    "wrong": "18 ohm",
                    "why": "The resistors were incorrectly treated as series."
                }
            ]
        }
        """
    ]

    def fake_generate(self, prompt):
        return responses.pop(0)

    monkeypatch.setattr(
        GeminiProvider,
        "generate",
        fake_generate,
    )

    graph = build_solve_graph()

    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "retry_count": 0,
    }

    result = graph.invoke(state)

    assert result["retry_count"] == 1
    assert result["solution"]["answer"] == "4 ohm"
    assert result["verification"]["passed"] is True


def test_generate_node_passes_retry_reason(monkeypatch):
    captured = {}

    def fake_generate_answer(question, retry_reason=None):
        captured["question"] = question
        captured["retry_reason"] = retry_reason

        return {
            "question_number": question["number"],
            "answer": "4 ohm",
            "steps": [],
            "mark_split": [],
            "common_mistakes": [],
            "confidence": 0.0,
            "verified_by": "none",
            "needs_teacher_check": True,
        }

    monkeypatch.setattr(
        "app.graph.solve.nodes.generate_answer",
        fake_generate_answer,
    )

    from app.graph.solve.nodes import generate_node

    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance.",
            "type": "numerical",
            "marks": 3,
        },
        "retry_reason": "The previous answer was incorrect.",
    }

    result = generate_node(state)

    assert result["solution"]["answer"] == "4 ohm"
    assert captured["retry_reason"] == "The previous answer was incorrect."
def test_route_node_for_numerical_question():
    from app.graph.solve.nodes import route_node

    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance.",
            "type": "numerical",
            "marks": 3,
        }
    }

    result = route_node(state)

    assert result["route"] == "numerical"


def test_route_node_for_mcq_question():
    from app.graph.solve.nodes import route_node

    state = {
        "question": {
            "number": "1",
            "text": "What is 2 + 2?",
            "type": "mcq",
            "marks": 1,
            "options": ["2", "3", "4", "5"],
        }
    }

    result = route_node(state)

    assert result["route"] == "mcq"


def test_route_node_for_unsupported_question():
    from app.graph.solve.nodes import route_node

    state = {
        "question": {
            "number": "5",
            "text": "Some question",
            "type": "unknown",
            "marks": 2,
        }
    }

    result = route_node(state)

    assert result["route"] == "unsupported"


def test_solve_graph_records_question_route(monkeypatch):
    fake_response = """
    {
        "answer": "5 ohm",
        "steps": [
            "For two 10 ohm resistors in parallel:",
            "R = (10 × 10) / (10 + 10)",
            "R = 5 ohm"
        ],
        "mark_split": [
            {
                "marks": 1,
                "for": "identifying the parallel combination"
            },
            {
                "marks": 1,
                "for": "using the correct formula"
            },
            {
                "marks": 1,
                "for": "correct final value"
            }
        ],
        "common_mistakes": []
    }
    """

    def fake_generate(self, prompt):
        return fake_response

    monkeypatch.setattr(
        "app.llm.gemini.GeminiProvider.generate",
        fake_generate,
    )

    graph = build_solve_graph()

    question = {
        "number": "4",
        "text": (
            "Two resistors of 10 ohm and 10 ohm "
            "are connected in parallel. Calculate the resistance."
        ),
        "type": "numerical",
        "marks": 3,
    }

    result = graph.invoke({
        "question": question,
    })

    assert result["route"] == "numerical"
    assert result["verification"]["passed"] is True


def test_verify_node_marks_verified_solution_without_teacher_check():
    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "solution": {
            "question_number": "3(b)",
            "answer": "4 ohm",
            "steps": [],
            "mark_split": [],
            "common_mistakes": [],
            "confidence": 0.0,
            "verified_by": "none",
            "needs_teacher_check": True,
        },
    }

    result = verify_node(state)

    assert result["verification"]["passed"] is True
    assert result["solution"]["verified_by"] == "symbolic"
    assert result["solution"]["needs_teacher_check"] is False


def test_verify_node_requires_teacher_check_without_independent_verification():
    state = {
        "question": {
            "number": "1",
            "text": "What is the capital of France?",
            "marks": 1,
            "type": "mcq",
        },
        "solution": {
            "question_number": "1",
            "answer": "Paris",
            "steps": [],
            "mark_split": [],
            "common_mistakes": [],
            "confidence": 0.0,
            "verified_by": "none",
            "needs_teacher_check": True,
        },
    }

    result = verify_node(state)

    assert result["verification"]["passed"] is False
    assert result["solution"]["verified_by"] == "none"
    assert result["solution"]["needs_teacher_check"] is True


def test_verify_node_requires_teacher_check_when_verification_fails():
    state = {
        "question": {
            "number": "3(b)",
            "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
            "marks": 3,
            "type": "numerical",
        },
        "solution": {
            "question_number": "3(b)",
            "answer": "18 ohm",
            "steps": [],
            "mark_split": [],
            "common_mistakes": [],
            "confidence": 0.0,
            "verified_by": "none",
            "needs_teacher_check": True,
        },
    }

    result = verify_node(state)

    assert result["verification"]["passed"] is False
    assert result["solution"]["verified_by"] == "symbolic"
    assert result["solution"]["needs_teacher_check"] is True

def test_solve_graph_verifies_mcq_with_answer_key(monkeypatch):
    fake_response = """
    {
        "answer": "B. lion",
        "steps": [
            "Identify the subject described in the sentence.",
            "The correct word is lion."
        ],
        "mark_split": [
            {
                "marks": 1,
                "for": "Correct option"
            }
        ],
        "common_mistakes": []
    }
    """

    monkeypatch.setattr(
        GeminiProvider,
        "generate",
        lambda self, prompt: fake_response,
    )

    graph = build_solve_graph()

    question = {
        "number": "5",
        "text": "The ___ guarded his territory fiercely.",
        "marks": 1,
        "type": "mcq",
        "options": {
            "A": "lioness",
            "B": "lion",
            "C": "tiger",
            "D": "tigress",
        },
    }

    result = graph.invoke(
        {
            "question": question,
            "expected_answer": "B",
        }
    )

    assert result["route"] == "mcq"
    assert result["solution"]["answer"] == "B. lion"
    assert result["verification"]["passed"] is True
    assert result["solution"]["verified_by"] == "answer_key"
    assert result["solution"]["needs_teacher_check"] is False
