from app.generate.nodes import mcq_solver


def test_mcq_options_are_labeled_and_retry_feedback_is_used(monkeypatch):
    captured = {}

    def fake_call(system_prompt, user_prompt):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return {
            "question_number": "1",
            "answer": "(B) Option B",
            "steps": ["Evaluate the options."],
            "mark_split": [{"marks": 1, "for": "Correct choice"}],
            "common_mistakes": [],
            "confidence": 0.8,
            "verified_by": "none",
            "needs_teacher_check": False,
        }

    monkeypatch.setattr(mcq_solver, "call_groq_json", fake_call)
    result = mcq_solver.mcq_solver_node({
        "number": "1",
        "text": "Assertion and reason question.",
        "marks": 1,
        "type": "mcq",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        # This is verifier-only state; it must not appear in the prompt.
        "expected_answer": "B",
        "retry_reason": (
            "Re-evaluate whether the reason logically explains the assertion."
        ),
    })

    prompt = captured["user_prompt"]
    assert "(A) Option A" in prompt
    assert "(B) Option B" in prompt
    assert "(C) Option C" in prompt
    assert "(D) Option D" in prompt
    assert "Re-evaluate whether the reason logically explains the assertion." in prompt
    assert "expected_answer" not in prompt
    assert result["raw_solution"]["answer"] == "(B) Option B"
