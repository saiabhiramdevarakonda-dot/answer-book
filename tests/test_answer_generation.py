
from app.answer_generation.generator import generate_answer
import app.answer_generation.generator as generator


def test_numerical_answer_structure(monkeypatch):
    question = {
        "number": "3(b)",
        "section": "B",
        "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
        "marks": 3,
        "type": "numerical",
        "options": None,
        "has_figure": False,
        "page": 2,
        "choice_group": None,
    }

    fake_response = {
        "answer": "4 ohm",
        "steps": [
            "Use the parallel resistance formula.",
            "Substitute 6 ohm and 12 ohm.",
            "Calculate the result.",
        ],
        "mark_split": [
            {"marks": 1, "for": "Correct formula"},
            {"marks": 1, "for": "Correct substitution"},
            {"marks": 1, "for": "Correct final answer"},
        ],
        "common_mistakes": [
            {
                "wrong": "18 ohm",
                "why": "The resistors were incorrectly treated as series.",
            }
        ],
    }

    monkeypatch.setattr(
        generator,
        "call_groq_json",
        lambda system_prompt, user_prompt: fake_response,
    )

    result = generate_answer(question)

    assert result["question_number"] == "3(b)"
    assert result["answer"] == "4 ohm"
    assert len(result["steps"]) == 3
    assert len(result["mark_split"]) == 3
    assert len(result["common_mistakes"]) == 1
    assert result["confidence"] == 0.0
    assert result["verified_by"] == "none"
    assert result["needs_teacher_check"] is True


def test_retry_reason_is_included_in_prompt(monkeypatch):
    captured = {}

    def fake_call(system_prompt, user_prompt):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {
            "answer": "4 ohm",
            "steps": ["Apply the parallel resistance formula."],
            "mark_split": [{"marks": 3, "for": "Correct solution"}],
            "common_mistakes": [],
        }

    monkeypatch.setattr(generator, "call_groq_json", fake_call)

    generate_answer(
        {
            "number": "1",
            "text": "Calculate the parallel resistance of 6 ohm and 12 ohm.",
            "type": "numerical",
            "marks": 3,
        },
        retry_reason="Recheck the arithmetic.",
    )

    assert "Recheck the arithmetic." in captured["user_prompt"]


def test_invalid_question_returns_teacher_review(monkeypatch):
    def unexpected_call(*args, **kwargs):
        raise AssertionError("LLM should not be called for an empty question.")

    monkeypatch.setattr(generator, "call_groq_json", unexpected_call)

    result = generate_answer(
        {
            "number": "2",
            "text": "",
            "type": "numerical",
            "marks": 2,
        }
    )

    assert result["answer"] == "Question text is missing."
    assert result["needs_teacher_check"] is True
    assert result["verified_by"] == "none"
