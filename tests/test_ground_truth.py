
from pathlib import Path

import pytest

from evaluation.ground_truth import (
    extract_mcq_answers,
    load_pdf_text,

)

def test_extract_mcq_answers_from_paper_455():
    """Extract all 20 official MCQ answers from Paper 455."""
    root = Path(__file__).resolve().parent.parent
    answer_key = root / "answer_key" / "answer_key_455.pdf"

    if not answer_key.is_file():
        pytest.skip("Paper 455 answer-key PDF is not available.")

    text = load_pdf_text(answer_key)
    answers = extract_mcq_answers(text)

    assert len(answers) == 20
    assert list(answers.keys()) == [str(n) for n in range(1, 21)]
    assert answers["1"] == "C"


def test_extract_mcq_answers_rejects_missing_sections():
    """Reject text that does not contain both answer-key sections."""
    with pytest.raises(ValueError, match="Section A and Section B"):
        extract_mcq_answers("This text has no answer-key sections.")


def test_extract_mcq_answers_rejects_missing_mcq_answers():
    """Reject a Section A that contains no extractable MCQ answers."""
    text = "Section A\nNo MCQ answers here.\nSection B\nDescriptive answers."

    with pytest.raises(ValueError, match="No MCQ answer keys"):
        extract_mcq_answers(text)

def test_extract_answer_text_stops_before_next_question():
    """Ensure the next question is excluded from the extracted answer."""
    from evaluation.ground_truth import extract_answer_text

    branch = (
        "Question 21: Explain photosynthesis.\n"
        "Answer: Photosynthesis converts light energy into chemical energy.\n"
        "What is chlorophyll?"
    )

    result = extract_answer_text(branch)

    assert result == (
        "Photosynthesis converts light energy into chemical energy."
    )


def test_split_or_branches_separates_alternative():
    """Ensure an OR alternative is separated correctly."""
    from evaluation.ground_truth import split_or_branches

    question = (
        "Question 21: Explain photosynthesis.\n"
        "OR\n"
        "Question 22: Explain respiration."
    )

    assert split_or_branches(question) == {
        "main": "Question 21: Explain photosynthesis.",
        "or": "Question 22: Explain respiration.",
    }
