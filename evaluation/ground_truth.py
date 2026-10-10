import re
from pathlib import Path
from pypdf import PdfReader


MCQ_ANSWER_RE = re.compile(
    r"Answer:\s*Correct Option:\s*\(([A-D])\)",
    re.IGNORECASE,
)


def load_pdf_text(path: str | Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def extract_mcq_answers(text: str) -> dict[str, str]:
    if "Section A" not in text or "Section B" not in text:
        raise ValueError("Could not find Section A and Section B in the answer key.")

    section_a = text.split("Section A", 1)[1].split("Section B", 1)[0]

    answers = [
        option.upper()
        for option in MCQ_ANSWER_RE.findall(section_a)
    ]

    if not answers:
        raise ValueError("No MCQ answer keys were found in Section A.")

    return {
        str(question_number): option
        for question_number, option in enumerate(answers, start=1)
    }
def split_or_branches(question_block: str) -> dict[str, str]:
    parts = re.split(r"(?m)^\s*OR\s*$", question_block, maxsplit=1)

    result = {
        "main": parts[0].strip(),
    }

    if len(parts) == 2:
        result["or"] = parts[1].strip()

    return result
def extract_answer_text(branch: str) -> str:
    match = re.search(
        r"(?ms)\bAnswer:\s*(.*?)(?=\n\s*?\s*Why\?|\n\s*\[Ch |\Z)",
        branch,
    )

    if not match:
        raise ValueError("Could not find an Answer: block.")

    return " ".join(match.group(1).split())
