
"""Automated tests for database persistence and caching."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("sqlalchemy")

# Ensure project root is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytestmark = [pytest.mark.live, pytest.mark.db]

try:
    from sqlalchemy import inspect

    from app.store.connection import engine, get_db_session
    from app.store.models import Base, PaperTable
    from app.tools.hash_utils import generate_question_signature
    from app.tools.paper_cache_tool import (
        check_paper_cache,
        get_paper_resume_info,
        mark_paper_status,
        upsert_paper_record,
    )
    from app.tools.question_lookup_tool import lookup_cached_question
    from app.tools.question_save_tool import save_question_checkpoint

except Exception as exc:
    pytest.skip(
        f"Database environment not configured: {exc}",
        allow_module_level=True,
    )


TEST_PAPER_ID = "test_pytest_paper_001"
TEST_FINGERPRINT = "test_pytest_fp_abcdef123456"


@pytest.fixture(scope="module", autouse=True)
def create_test_tables():
    """Ensure required tables exist for database tests."""
    Base.metadata.create_all(bind=engine)
    yield
    # Intentionally do not drop tables: the configured database
    # may be shared with other application components.


@pytest.fixture(autouse=True)
def clean_test_records():
    """Remove only this test's paper and its associated questions."""
    def cleanup():
        with get_db_session() as session:
            paper = (
                session.query(PaperTable)
                .filter_by(paper_id=TEST_PAPER_ID)
                .first()
            )
            if paper is not None:
                session.delete(paper)

    cleanup()
    yield
    cleanup()


def test_db_tables_exist():
    """Verify the required tables exist."""
    tables = set(inspect(engine).get_table_names())

    assert "papers" in tables
    assert "solved_questions" in tables


def test_save_and_pull_question_checkpoint():
    """Save a question and retrieve it for resume."""
    upsert_paper_record(
        paper_id=TEST_PAPER_ID,
        fingerprint=TEST_FINGERPRINT,
        status="solving",
        subject="Mathematics Part 1",
        class_name="Class 9",
        board="CBSE",
        total_questions=1,
        total_marks=2,
    )

    question = {
        "number": "1",
        "text": "Find the value of x if 2x + 4 = 10.",
        "marks": 2,
        "section": "A",
        "type": "numerical",
        "options": None,
        "has_figure": False,
    }

    signature = generate_question_signature(
        question["text"], question["marks"], "CBSE", "Class 9"
    )

    solution = {
        "question_number": "1",
        "answer": "x = 3",
        "steps": ["2x = 10 - 4", "2x = 6", "x = 3"],
        "mark_split": [
            {"marks": 1, "for": "isolating x"},
            {"marks": 1, "for": "final value"},
        ],
        "common_mistakes": [
            {
                "wrong": "x = 7",
                "why": "Added 4 instead of subtracting.",
            }
        ],
        "confidence": 0.99,
        "verified_by": "test",
        "needs_teacher_check": False,
    }

    assert save_question_checkpoint(
        TEST_PAPER_ID, signature, question, solution
    ) is True

    saved_solutions, solved_numbers = get_paper_resume_info(TEST_PAPER_ID)

    assert "1" in solved_numbers
    assert len(saved_solutions) == 1

    saved = saved_solutions[0]
    assert saved["question_number"] == "1"
    assert saved["answer"] == "x = 3"
    assert len(saved["steps"]) == 3
    assert len(saved["mark_split"]) == 2
    assert saved["confidence"] == 0.99
    assert saved["needs_teacher_check"] is False


def test_question_signature_cache_lookup():
    """Find a saved question by signature and remap its question number."""
    upsert_paper_record(
        TEST_PAPER_ID, TEST_FINGERPRINT, status="solving"
    )

    question = {
        "number": "5",
        "text": "What is the capital of France?",
        "marks": 1,
        "section": "A",
        "type": "short",
    }

    signature = generate_question_signature(
        question["text"], question["marks"], "CBSE", "Class 9"
    )

    solution = {
        "question_number": "5",
        "answer": "Paris",
        "steps": ["Recall the capital city of France."],
        "mark_split": [{"marks": 1, "for": "correct city"}],
        "common_mistakes": [],
        "confidence": 1.0,
        "verified_by": "test",
        "needs_teacher_check": False,
    }

    save_question_checkpoint(
        TEST_PAPER_ID, signature, question, solution
    )

    cached = lookup_cached_question(
        signature, target_question_number="14"
    )

    assert cached is not None
    assert cached["question_number"] == "14"
    assert cached["answer"] == "Paris"

    assert lookup_cached_question(
        "non_existent_signature_hash"
    ) is None


def test_paper_level_cache_preflight():
    """Test paper cache status transitions and cache retrieval."""
    upsert_paper_record(
        TEST_PAPER_ID,
        TEST_FINGERPRINT,
        status="solving",
        subject="Science",
    )

    assert check_paper_cache(TEST_FINGERPRINT) is None

    question = {
        "number": "1",
        "text": "Define inertia.",
        "marks": 2,
    }

    signature = generate_question_signature(
        question["text"], 2, "CBSE", "Class 9"
    )

    solution = {
        "question_number": "1",
        "answer": (
            "Inertia is the resistance of an object to changes "
            "in its state of motion."
        ),
        "steps": ["State the definition."],
        "mark_split": [{"marks": 2, "for": "definition"}],
        "common_mistakes": [],
        "confidence": 0.95,
        "verified_by": "test",
        "needs_teacher_check": False,
    }

    save_question_checkpoint(
        TEST_PAPER_ID, signature, question, solution
    )

    mark_paper_status(TEST_PAPER_ID, "ready")

    cached = check_paper_cache(TEST_FINGERPRINT)

    assert cached is not None
    assert cached["paper_id"] == TEST_PAPER_ID
    assert cached["status"] == "ready"
    assert len(cached["solutions"]) == 1
    assert cached["solutions"][0]["answer"] == solution["answer"]
