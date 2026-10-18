from app.generate.graph import build_question_solver_graph


def test_question_graph_retries_failed_candidate(monkeypatch):
    calls = {"solver": 0, "verification": 0}
    retry_feedback = (
        "Re-evaluate the assertion and reason independently; "
        "do not infer causation from related facts."
    )

    def planner(state):
        return {"selected_route": "mcq"}

    def solver(state):
        calls["solver"] += 1
        return {
            "raw_solution": {
                "question_number": "1",
                "answer": f"candidate-{calls['solver']}",
                "steps": ["Reason through the options."],
                "mark_split": [{"marks": 1, "for": "Correct option"}],
                "common_mistakes": [],
                "confidence": 0.8,
            }
        }

    def verification(state):
        calls["verification"] += 1
        passed = calls["verification"] == 2
        return {
            "verification": {
                "passed": passed,
                "verified_by": "answer_key" if passed else "answer_key",
                "reason": "Pass" if passed else "Mismatch",
                "retry_feedback": None if passed else retry_feedback,
                "needs_teacher_check": not passed,
            }
        }

    def normalizer(state):
        verification_result = state.get("verification", {})
        return {
            "solution": {
                "question_number": state.get("number", "1"),
                "answer": (state.get("raw_solution") or {}).get("answer", ""),
                "verified_by": verification_result.get("verified_by", "none")
                if verification_result.get("passed")
                else "none",
                "needs_teacher_check": not verification_result.get("passed", False),
            }
        }

    monkeypatch.setattr("app.generate.graph.planner_node", planner)
    monkeypatch.setattr("app.generate.graph.mcq_solver_node", solver)
    monkeypatch.setattr("app.generate.graph.verification_node", verification)
    monkeypatch.setattr("app.generate.graph.normalizer_node", normalizer)

    graph = build_question_solver_graph()
    result = graph.invoke({
        "number": "1",
        "text": "Assertion and reason question",
        "marks": 1,
        "type": "mcq",
        "options": ["Option A", "Option B", "Option C", "Option D"],
        "retry_count": 0,
    })

    assert calls["solver"] == 2
    assert calls["verification"] == 2
    assert result["retry_count"] == 1
    assert result["retry_reason"] == retry_feedback
    assert result["solution"]["answer"] == "candidate-2"
    assert result["solution"]["verified_by"] == "answer_key"
    assert result["solution"]["needs_teacher_check"] is False


def test_question_graph_stops_after_max_retries(monkeypatch):
    calls = {"solver": 0, "verification": 0}

    def planner(state):
        return {"selected_route": "mcq"}

    def solver(state):
        calls["solver"] += 1
        return {"raw_solution": {"answer": f"candidate-{calls['solver']}"}}

    def verification(state):
        calls["verification"] += 1
        return {
            "verification": {
                "passed": False,
                "verified_by": "answer_key",
                "reason": "Mismatch",
                "retry_feedback": "Re-evaluate independently.",
                "needs_teacher_check": True,
            }
        }

    def normalizer(state):
        return {"solution": {"answer": state["raw_solution"]["answer"]}}

    monkeypatch.setattr("app.generate.graph.planner_node", planner)
    monkeypatch.setattr("app.generate.graph.mcq_solver_node", solver)
    monkeypatch.setattr("app.generate.graph.verification_node", verification)
    monkeypatch.setattr("app.generate.graph.normalizer_node", normalizer)

    graph = build_question_solver_graph()
    result = graph.invoke({
        "number": "1",
        "text": "MCQ",
        "marks": 1,
        "type": "mcq",
        "options": ["A", "B", "C", "D"],
        "retry_count": 0,
    })

    assert calls["solver"] == 3  # initial attempt + two retries
    assert calls["verification"] == 3
    assert result["retry_count"] == 2
    assert result["solution"]["answer"] == "candidate-3"


def test_question_graph_does_not_retry_when_verification_is_impossible(monkeypatch):
    calls = {"solver": 0, "verification": 0}

    def planner(state):
        return {"selected_route": "mcq"}

    def solver(state):
        calls["solver"] += 1
        return {"raw_solution": {"answer": "B. Mars"}}

    def verification(state):
        calls["verification"] += 1
        return {
            "verification": {
                "passed": False,
                "verified_by": "none",
                "reason": "Cannot verify MCQ without an answer key.",
                "retryable": False,
                "needs_teacher_check": True,
            }
        }

    def normalizer(state):
        return {"solution": {"answer": state["raw_solution"]["answer"]}}

    monkeypatch.setattr("app.generate.graph.planner_node", planner)
    monkeypatch.setattr("app.generate.graph.mcq_solver_node", solver)
    monkeypatch.setattr("app.generate.graph.verification_node", verification)
    monkeypatch.setattr("app.generate.graph.normalizer_node", normalizer)

    graph = build_question_solver_graph()
    result = graph.invoke({
        "number": "1",
        "text": "Which planet is red?",
        "marks": 1,
        "type": "mcq",
        "options": ["Earth", "Mars", "Venus", "Jupiter"],
        "retry_count": 0,
    })

    assert calls["solver"] == 1
    assert calls["verification"] == 1
    assert result["retry_count"] == 0
