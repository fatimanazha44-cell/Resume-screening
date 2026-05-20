"""Tests for reporter.py — pure transform, no API calls."""

import json
from pathlib import Path

from src.models import (
    CandidateProfile,
    DimensionScore,
    Gap,
    ProcessingError,
    Role,
    ScoredCandidate,
    UsageAccumulator,
)
from src.reporter import format_cost_report, write_json, write_markdown


def _make_scored(name: str, overall: int, source: str) -> ScoredCandidate:
    return ScoredCandidate(
        profile=CandidateProfile(
            name=name,
            years_experience=5.0,
            skills=["Python", "SQL"],
            past_roles=[
                Role(
                    title="Backend Engineer",
                    company="Acme",
                    duration_months=24,
                    description="Worked on APIs.",
                )
            ],
            education=["BS Computer Science"],
            raw_summary="I build backend systems.",
        ),
        skills_match=DimensionScore(score=overall - 5, reasoning="Skills reason"),
        experience_match=DimensionScore(score=overall, reasoning="Experience reason"),
        role_relevance=DimensionScore(score=overall + 2, reasoning="Role reason"),
        overall_fit=DimensionScore(score=overall, reasoning="Overall reason"),
        reasoning="Two-sentence reasoning. Looks strong.",
        gaps=[Gap(category="skill", detail="No Kafka experience")],
        source_file=source,
    )


def test_json_output_sorts_by_overall_fit(tmp_path: Path) -> None:
    results = [
        _make_scored("Bob", 70, "bob.pdf"),
        _make_scored("Alice", 90, "alice.pdf"),
        _make_scored("Carol", 80, "carol.pdf"),
    ]

    out = tmp_path / "results.json"
    write_json(results, out)

    data = json.loads(out.read_text())
    names = [c["profile"]["name"] for c in data["candidates"]]
    assert names == ["Alice", "Carol", "Bob"]
    assert data["errors"] == []


def test_json_separates_errors_from_candidates(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(
            source_file="broken.pdf", stage="parse", message="PDF corrupt"
        ),
    ]

    out = tmp_path / "results.json"
    write_json(results, out)

    data = json.loads(out.read_text())
    assert len(data["candidates"]) == 1
    assert data["candidates"][0]["profile"]["name"] == "Alice"
    assert len(data["errors"]) == 1
    assert data["errors"][0]["stage"] == "parse"


def test_markdown_contains_ranked_table_and_sections(tmp_path: Path) -> None:
    results = [
        _make_scored("Bob", 70, "bob.pdf"),
        _make_scored("Alice", 90, "alice.pdf"),
    ]

    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    content = out.read_text()

    assert "# Resume Match Report" in content
    assert "## Ranked candidates" in content
    assert "Alice" in content
    assert "Bob" in content
    # Alice should appear before Bob in the ranked table
    assert content.index("Alice") < content.index("Bob")


def test_markdown_includes_error_section_when_errors_present(tmp_path: Path) -> None:
    results = [
        _make_scored("Alice", 90, "alice.pdf"),
        ProcessingError(
            source_file="broken.pdf", stage="extract", message="API timeout"
        ),
    ]

    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    content = out.read_text()

    assert "## Could not process" in content
    assert "broken.pdf" in content
    assert "extract" in content
    assert "API timeout" in content


def test_markdown_omits_error_section_when_no_errors(tmp_path: Path) -> None:
    results = [_make_scored("Alice", 90, "alice.pdf")]

    out = tmp_path / "report.md"
    write_markdown(results, out, Path("jd.txt"))
    content = out.read_text()

    assert "## Could not process" not in content


# ── UsageAccumulator helper ──────────────────────────────────────────────────


class _FakeUsage:
    def __init__(self, inp: int, out: int, cw: int, cr: int) -> None:
        self.input_tokens = inp
        self.output_tokens = out
        self.cache_creation_input_tokens = cw
        self.cache_read_input_tokens = cr


def _make_acc(
    input_tokens: int = 1_000,
    output_tokens: int = 500,
    cache_creation: int = 200,
    cache_read: int = 300,
) -> UsageAccumulator:
    acc = UsageAccumulator()
    acc.add(_FakeUsage(input_tokens, output_tokens, cache_creation, cache_read))
    return acc


# ── format_cost_report() ─────────────────────────────────────────────────────


def test_format_cost_report_known_model_no_approximation_note() -> None:
    acc = _make_acc()
    report = format_cost_report(acc, "claude-sonnet-4-6")
    assert "claude-sonnet-4-6" in report
    assert "approximated" not in report


def test_format_cost_report_unknown_model_shows_fallback_note() -> None:
    acc = _make_acc()
    report = format_cost_report(acc, "claude-unknown-9")
    assert "claude-unknown-9" in report
    assert "approximated" in report


def test_format_cost_report_zero_tokens_shows_zero_cost() -> None:
    acc = UsageAccumulator()
    report = format_cost_report(acc, "claude-sonnet-4-6")
    assert "$0.0000" in report


def test_format_cost_report_correct_input_cost() -> None:
    # 1 000 000 input tokens at $3.00/M = exactly $3.0000
    acc = _make_acc(input_tokens=1_000_000, output_tokens=0, cache_creation=0, cache_read=0)
    report = format_cost_report(acc, "claude-sonnet-4-6")
    assert "$3.0000" in report


def test_format_cost_report_contains_all_token_type_counts() -> None:
    acc = _make_acc(input_tokens=100, output_tokens=200, cache_creation=300, cache_read=400)
    report = format_cost_report(acc, "claude-sonnet-4-6")
    assert "100" in report
    assert "200" in report
    assert "300" in report
    assert "400" in report


# ── write_markdown() cost section ────────────────────────────────────────────


def test_write_markdown_includes_cost_section_when_acc_and_model_given(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.md"
    write_markdown([_make_scored("Alice", 90, "alice.pdf")], out, Path("jd.txt"),
                   acc=_make_acc(), model="claude-sonnet-4-6")
    content = out.read_text()
    assert "## Cost breakdown" in content
    assert "claude-sonnet-4-6" in content
    assert "Input" in content
    assert "Output" in content


def test_write_markdown_omits_cost_section_when_acc_not_given(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    write_markdown([_make_scored("Alice", 90, "alice.pdf")], out, Path("jd.txt"))
    assert "## Cost breakdown" not in out.read_text()


def test_write_markdown_cost_section_unknown_model_shows_approximation(
    tmp_path: Path,
) -> None:
    out = tmp_path / "report.md"
    write_markdown([_make_scored("Alice", 90, "alice.pdf")], out, Path("jd.txt"),
                   acc=_make_acc(), model="claude-unknown-9")
    assert "approximated" in out.read_text()
