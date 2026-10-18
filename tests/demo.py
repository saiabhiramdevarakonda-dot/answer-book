"""Command-line demo for the answer-book solve-and-verify workflow.

Run from the repository root:
    python -m tests.demo PAPER_NUMBER
    python -m tests.demo PAPER_NUMBER QUESTION_NUMBER

Official answer keys are used only to verify MCQ answers, never as generation
context. Descriptive questions are evaluated by the configured verifier.
"""

import re
import sys
from pathlib import Path
from unittest.mock import patch

from evaluation.ground_truth import extract_mcq_answers, load_pdf_text

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "test_papers"
KEYS_DIR = ROOT / "answer_papers"
MCQ_MAX_NUMBER = 20


def get_section(pdf_text, section_name, next_section=None):
    """Return text between section headings."""
    start_match = re.search(
        rf"\bSection\s+{re.escape(section_name)}\b",
        pdf_text,
        re.IGNORECASE,
    )
    if not start_match:
        raise ValueError(f"Could not find Section {section_name}.")

    remaining = pdf_text[start_match.end():]
    if next_section:
        end_match = re.search(
            rf"\bSection\s+{re.escape(next_section)}\b",
            remaining,
            re.IGNORECASE,
        )
        if end_match:
            remaining = remaining[:end_match.start()]
    return remaining


def find_question_headers(section_text, minimum, maximum=None):
    """Find numbered question headers."""
    pattern = re.compile(r"(?m)^\s*(\d{1,3})[.)]?\s+(\S.*)$")
    headers = []
    for match in pattern.finditer(section_text):
        number = int(match.group(1))
        if number < minimum:
            continue
        if maximum is not None and number > maximum:
            continue
        headers.append(match)
    return headers


def get_question_block(section_text, headers, target_number):
    """Return a target question's text block."""
    for index, header in enumerate(headers):
        if int(header.group(1)) != int(target_number):
            continue
        end = headers[index + 1].start() if index + 1 < len(headers) else len(section_text)
        return header, section_text[header.start():end]
    raise ValueError(f"Question {target_number} was not found in its section.")


def parse_mcq(header, block):
    """Parse a numbered MCQ and its A-D options."""
    lines = block.splitlines()
    question_lines = [header.group(2).strip()]
    options = {}
    current_option = None

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\[Ch\s", stripped, re.IGNORECASE):
            break
        if re.match(r"^Page\s+\d+", stripped, re.IGNORECASE):
            break
        if re.fullmatch(r"\d+", stripped):
            continue

        option_match = re.match(r"^\(([A-D])\)\s*(.*)$", stripped)
        if option_match:
            current_option = option_match.group(1)
            options[current_option] = option_match.group(2).strip()
        elif current_option and stripped:
            options[current_option] += " " + stripped
        else:
            question_lines.append(stripped)

    if len(options) != 4:
        raise ValueError(f"Q{header.group(1)} did not yield exactly four MCQ options.")

    question_text = " ".join(question_lines).strip()
    if not question_text:
        raise ValueError(f"Question {header.group(1)} has no text.")

    return {
        "number": str(header.group(1)),
        "section": "A",
        "text": question_text,
        "marks": 1,
        "type": "mcq",
        "options": options,
        "has_figure": False,
        "page": None,
        "choice_group": None,
    }


def extract_mcqs(pdf_text, target_question=None):
    """Extract all MCQs or only one selected question from Section A."""
    section = get_section(pdf_text, "A", "B")
    headers = find_question_headers(section, minimum=1, maximum=MCQ_MAX_NUMBER)
    if not headers:
        raise ValueError("No numbered questions found in Section A.")

    if target_question is not None:
        header, block = get_question_block(section, headers, target_question)
        return [parse_mcq(header, block)]

    questions = []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(section)
        try:
            questions.append(parse_mcq(header, section[header.start():end]))
        except ValueError:
            # Do not silently count malformed questions as extracted.
            continue
    if not questions:
        raise ValueError("No valid MCQs could be extracted.")
    return questions


def parse_descriptive(header, block):
    """Parse a Section B descriptive question and its mark value."""
    lines = block.splitlines()
    question_lines = [header.group(2).strip()]

    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\[Ch\s", stripped, re.IGNORECASE):
            break
        if re.match(r"^Page\s+\d+", stripped, re.IGNORECASE):
            break
        if re.fullmatch(r"\d+", stripped):
            continue
        question_lines.append(stripped)

    text = " ".join(question_lines).strip()
    if not text:
        raise ValueError(f"Question {header.group(1)} has no readable text.")

    mark_match = re.search(
        r"(?:\[(\d+)\s*marks?\]|\((\d+)\s*marks?\)|(\d+)\s*marks?\b)",
        text,
        re.IGNORECASE,
    )
    if mark_match:
        marks = int(next(group for group in mark_match.groups() if group))
        text = (text[:mark_match.start()] + text[mark_match.end():]).strip()
    else:
        trailing_number = re.search(r"\s+(\d+)\s*$", text)
        if trailing_number:
            marks = int(trailing_number.group(1))
            text = text[:trailing_number.start()].strip()
        else:
            marks = 2

    return {
        "number": str(header.group(1)),
        "section": "B",
        "text": text,
        "marks": marks,
        "type": "short" if marks <= 2 else "long",
        "options": {},
        "has_figure": bool(re.search(r"\b(figure|diagram|draw|sketch)\b", text, re.I)),
        "page": None,
        "choice_group": "OR" if re.search(r"\bOR\b", text, re.I) else None,
    }


def extract_descriptive_question(pdf_text, target_question):
    """Extract a selected descriptive question from Section B."""
    section = get_section(pdf_text, "B", "C")
    headers = find_question_headers(section, minimum=MCQ_MAX_NUMBER + 1)
    if not headers:
        raise ValueError("No numbered descriptive questions found in Section B.")
    header, block = get_question_block(section, headers, target_question)
    return [parse_descriptive(header, block)]


def display_confidence(solution, verification):
    """Display model-reported confidence values without implying calibration."""
    print("\nConfidence scores (model-reported; not calibrated probabilities):")

    generation_confidence = solution.get("confidence")
    if generation_confidence is None:
        print("  Generation confidence: N/A")
    else:
        try:
            value = float(generation_confidence)
            print(f"  Generation confidence: {value:.1%}")
        except (TypeError, ValueError):
            print(f"  Generation confidence: invalid value ({generation_confidence!r})")

    verification_confidence = verification.get("confidence")
    if verification_confidence is None:
        print("  Verification confidence: N/A")
    else:
        try:
            value = float(verification_confidence)
            print(f"  Verification confidence: {value:.1%}")
        except (TypeError, ValueError):
            print(f"  Verification confidence: invalid value ({verification_confidence!r})")

    score = verification.get("score")
    max_marks = verification.get("max_marks")
    if score is not None:
        print(f"  Rubric score: {score}/{max_marks if max_marks is not None else '?'}")


def main():
    if len(sys.argv) not in (2, 3) or not sys.argv[1].isdigit():
        print("Usage:")
        print("  python -m tests.demo PAPER_NUMBER")
        print("  python -m tests.demo PAPER_NUMBER QUESTION_NUMBER")
        raise SystemExit(2)

    paper_id = sys.argv[1]
    question_id = sys.argv[2] if len(sys.argv) == 3 else None
    if question_id is not None and not question_id.isdigit():
        print("QUESTION_NUMBER must be a number.")
        raise SystemExit(2)

    paper_path = PAPERS_DIR / f"question_paper_{paper_id}.pdf"
    key_path = KEYS_DIR / f"answer_key_{paper_id}.pdf"
    if not paper_path.is_file():
        print(f"Question paper not found: {paper_path}")
        raise SystemExit(1)

    is_descriptive_test = question_id is not None and int(question_id) > MCQ_MAX_NUMBER
    has_answer_key = key_path.is_file()
    if not is_descriptive_test and not has_answer_key:
        print(f"Official answer-key PDF not found: {key_path}")
        print("MCQ answers cannot be independently checked against ground truth.")

    print("\n" + "=" * 65)
    print(f"ANSWER BOOK DEMO -- PAPER {paper_id}")
    print("=" * 65)

    try:
        paper_text = load_pdf_text(paper_path)
        if is_descriptive_test:
            questions = extract_descriptive_question(paper_text, question_id)
            answer_key = {}
        else:
            questions = extract_mcqs(paper_text, target_question=question_id)
            answer_key = extract_mcq_answers(load_pdf_text(key_path)) if has_answer_key else {}
    except Exception as exc:
        print(f"Input processing failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    print(f"Paper loaded: {paper_path.name}")
    print(f"Questions selected: {len(questions)}")
    if not is_descriptive_test:
        print(f"Answer-key entries loaded: {len(answer_key)}")
    else:
        print("Mode: DESCRIPTIVE QUESTION")
        print("Answer-key comparison: Not applicable")
    print(f"Target question: Q{question_id}" if question_id is not None else "Mode: All extractable Section A MCQs")
    print("Answer-key values are hidden during generation.")

    try:
        from app.generate.graph import question_solver_graph
    except Exception as exc:
        print(f"Graph initialization failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)

    passed_count = failed_count = unverified_count = 0
    generation_error_count = missing_key_count = teacher_review_count = 0

    # Bypass cached answers so the demo actually tests the current generation pipeline.
    cache_bypass = patch("app.generate.nodes.planner.lookup_cached_question", return_value=None)
    cache_bypass.start()
    try:
        for index, question in enumerate(questions, start=1):
            number = str(question["number"])
            is_mcq = question["type"] == "mcq"
            expected = answer_key.get(number) if is_mcq else None

            print("\n" + "-" * 65)
            print(f"QUESTION {index}/{len(questions)} -- Q{number}")
            print(f"Type: {question['type']} | Marks: {question['marks']}")
            print("-" * 65)
            print(question["text"])
            if is_mcq:
                for letter, option_text in question["options"].items():
                    print(f"  ({letter}) {option_text}")
                if expected is None:
                    print("Status: NOT VERIFIED -- official answer-key entry unavailable")
                    print("Teacher review required: YES")
                    missing_key_count += 1
                    teacher_review_count += 1
                    continue

            graph_input = {
                "number": number,
                "section": str(question.get("section", "A")),
                "text": question["text"],
                "marks": int(question.get("marks", 1)),
                "type": question["type"],
                "options": list(question["options"].values()) if isinstance(question.get("options"), dict) else question.get("options", []),
                "has_figure": question.get("has_figure", False),
                "page": question.get("page") or 1,
                "choice_group": question.get("choice_group"),
            }
            if is_mcq:
                graph_input["expected_answer"] = expected

            try:
                result = question_solver_graph.invoke(graph_input)
                solution = result.get("solution", {}) or {}
                verification = result.get("verification", {}) or {}
                method = verification.get("verified_by", "none")
                verifier_passed = verification.get("passed") is True

                print("\nGenerated answer:")
                print(solution.get("answer", "(no answer generated)"))
                display_confidence(solution, verification)

                steps = solution.get("steps", [])
                if steps:
                    print("\nSolution steps:")
                    for step_number, step in enumerate(steps, start=1):
                        print(f"  {step_number}. {step}")

                mark_split = solution.get("mark_split", [])
                if mark_split:
                    print("\nMark split:")
                    for item in mark_split:
                        if isinstance(item, dict):
                            print(f"  {item.get('marks', '?')} marks: {item.get('for_', item.get('for', ''))}")
                        else:
                            print(f"  {item}")

                print("\nVerification result:")
                print("Verification method:", method)
                print("Reason:", verification.get("reason", "No verification details"))
                print("Retry count:", result.get("retry_count", 0))

                needs_review = verification.get("needs_teacher_check", False) is True
                if method == "none":
                    print("Status: NOT INDEPENDENTLY VERIFIED")
                    unverified_count += 1
                    needs_review = True
                elif verifier_passed:
                    print("Status: PASS -- LLM RUBRIC EVALUATION" if method == "llm_rubric" else "Status: PASS")
                    passed_count += 1
                else:
                    print("Status: FAIL")
                    failed_count += 1
                    needs_review = True

                print("Teacher review required:", "YES" if needs_review else "NO")
                if needs_review:
                    teacher_review_count += 1

            except Exception as exc:
                print("\nStatus: GENERATION ERROR")
                print(f"Error type: {type(exc).__name__}")
                print(f"Error details: {exc}")
                print("Teacher review required: YES")
                generation_error_count += 1
                teacher_review_count += 1
    finally:
        cache_bypass.stop()

    processed_count = passed_count + failed_count + unverified_count + generation_error_count + missing_key_count
    verified_count = passed_count + failed_count

    print("\n" + "=" * 65)
    print("FINAL DEMO SUMMARY")
    print("=" * 65)
    print(f"Paper number:               {paper_id}")
    print(f"Questions in this run:      {len(questions)}")
    print(f"Questions processed:        {processed_count}")
    print(f"Verification passed:        {passed_count}")
    print(f"Verification failed:        {failed_count}")
    print(f"Not independently verified: {unverified_count}")
    print(f"Generation errors:          {generation_error_count}")
    print(f"Missing answer-key entry:   {missing_key_count}")
    print(f"Teacher review flagged:     {teacher_review_count}")
    if verified_count:
        print(f"Pass rate among independently verified answers: {passed_count / verified_count * 100:.1f}%")
    else:
        print("Pass rate among independently verified answers: N/A")


if __name__ == "__main__":
    main()
