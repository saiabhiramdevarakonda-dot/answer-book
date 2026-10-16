from tests.demo import parse_descriptive


def test_parse_descriptive_removes_trailing_marks_number():
    """A trailing standalone number should be treated as marks."""
    header = type("Header", (), {"group": lambda self, n: "21" if n == 1 else "Why is π classified as an irrational number?"})()

    result = parse_descriptive(header, "Q21. Why is π classified as an irrational number?\n2")

    assert result["text"] == "Why is π classified as an irrational number?"
    assert result["marks"] == 2


def test_parse_descriptive_preserves_numbers_inside_question():
    """Numbers that are part of the question text must be preserved."""
    header = type("Header", (), {"group": lambda self, n: "22" if n == 1 else "Find the value of 119² by expressing 119 as 100 + 10 + 9 and applying a suitable identity."})()

    result = parse_descriptive(
        header,
        "Q22. Find the value of 119² by expressing 119 as 100 + 10 + 9 and applying a suitable identity.\n2",
    )

    assert "119²" in result["text"]
    assert "100 + 10 + 9" in result["text"]
    assert result["marks"] == 2
