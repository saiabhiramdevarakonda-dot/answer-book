from app.verification.verifier import verify_solution


def test_verifier_accepts_correct_numerical_answer():
    question = {
        "number": "3(b)",
        "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
        "marks": 3,
        "type": "numerical",
    }

    solution = {
        "question_number": "3(b)",
        "answer": "4 ohm",
        "steps": [
            "Use the parallel resistance formula.",
            "Substitute the values.",
            "Calculate the result.",
        ],
        "mark_split": [],
        "common_mistakes": [],
    }

    result = verify_solution(question, solution)

    assert result["passed"] is True
    assert result["verified_by"] == "symbolic"


def test_verifier_rejects_wrong_numerical_answer():
    question = {
        "number": "3(b)",
        "text": "Calculate the resistance of 6 ohm and 12 ohm resistors in parallel.",
        "marks": 3,
        "type": "numerical",
    }

    solution = {
        "question_number": "3(b)",
        "answer": "18 ohm",
        "steps": [
            "Add the two resistances.",
        ],
        "mark_split": [],
        "common_mistakes": [],
    }

    result = verify_solution(question, solution)

    assert result["passed"] is False
    assert result["verified_by"] == "symbolic"
def test_parallel_resistance_with_different_values():
    question = {
        "type": "numerical",
        "text": "Two resistors of 10 ohm and 10 ohm are connected in parallel.",
    }

    solution = {
        "answer": "5 ohm",
    }

    result = verify_solution(question, solution)

    assert result["passed"] is True
    assert result["verified_by"] == "symbolic"


def test_parallel_resistance_rejects_wrong_value():
    question = {
        "type": "numerical",
        "text": "Two resistors of 4 ohm and 4 ohm are connected in parallel.",
    }

    solution = {
        "answer": "8 ohm",
    }

    result = verify_solution(question, solution)

    assert result["passed"] is False
    assert result["verified_by"] == "symbolic"
def test_verifier_accepts_mcq_answer_key_match():
    question = {
        "number": "5",
        "type": "mcq",
        "text": "The ___ guarded his territory fiercely.",
        "options": {
            "A": "lioness",
            "B": "lion",
            "C": "tiger",
            "D": "tigress",
        },
    }

    solution = {
        "answer": "B. lion",
    }

    result = verify_solution(
        question,
        solution,
        expected_answer="B",
    )

    assert result["passed"] is True
    assert result["verified_by"] == "answer_key"


def test_verifier_rejects_mcq_answer_key_mismatch():
    question = {
        "number": "5",
        "type": "mcq",
        "text": "The ___ guarded his territory fiercely.",
        "options": {
            "A": "lioness",
            "B": "lion",
            "C": "tiger",
            "D": "tigress",
        },
    }

    solution = {
        "answer": "C. tiger",
    }

    result = verify_solution(
        question,
        solution,
        expected_answer="B",
    )

    assert result["passed"] is False
    assert result["verified_by"] == "answer_key"

def test_verifier_accepts_mcq_answer_key_match():
    question = {
        "number": "5",
        "type": "mcq",
        "text": "The ___ guarded his territory fiercely.",
        "options": {
            "A": "lioness",
            "B": "lion",
            "C": "tiger",
            "D": "tigress",
        },
    }

    solution = {"answer": "B. lion"}

    result = verify_solution(
        question,
        solution,
        expected_answer="B",
    )

    assert result["passed"] is True
    assert result["verified_by"] == "answer_key"


def test_verifier_rejects_mcq_answer_key_mismatch():
    question = {
        "number": "5",
        "type": "mcq",
        "text": "The ___ guarded his territory fiercely.",
        "options": {
            "A": "lioness",
            "B": "lion",
            "C": "tiger",
            "D": "tigress",
        },
    }

    solution = {"answer": "C. tiger"}

    result = verify_solution(
        question,
        solution,
        expected_answer="B",
    )

    assert result["passed"] is False
    assert result["verified_by"] == "answer_key"

def test_verifier_does_not_pass_mcq_without_answer_key():
    question = {
        "number": "6",
        "type": "mcq",
        "text": "Which planet is known as the Red Planet?",
        "options": {
            "A": "Earth",
            "B": "Mars",
            "C": "Venus",
            "D": "Jupiter",
        },
    }

    solution = {
        "answer": "B. Mars",
    }

    result = verify_solution(question, solution)

    assert result["passed"] is False
    assert result["verified_by"] == "none"



def test_mcq_retry_feedback_does_not_reveal_answer_key():
    question = {
        "number": "1",
        "type": "mcq",
        "text": "Assertion and reason question.",
        "options": {
            "A": "Both true and reason explains assertion",
            "B": "Both true but reason does not explain assertion",
            "C": "Assertion true, reason false",
            "D": "Assertion false, reason true",
        },
    }

    result = verify_solution(
        question,
        {"answer": "(A) Both true and reason explains assertion"},
        expected_answer="B",
    )

    assert result["passed"] is False
    assert result["retryable"] is True
    feedback = result["retry_feedback"].casefold()
    assert "independently" in feedback
    assert "option b" not in feedback
    assert "expected answer" not in feedback
    assert "both true but reason does not explain assertion" not in feedback
