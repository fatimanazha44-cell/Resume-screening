"""
Final stage of the pipeline: take the in-memory results and write them to
disk as machine-readable JSON and a human-readable markdown report.

No LLM calls here — this is a pure transform from Pydantic models to text.
That's why test_reporter.py can run in milliseconds with no API key.

Two outputs:
  - results.json: {"candidates": [...sorted by overall_fit desc...],
                   "errors":     [...resumes that failed at any stage...]}
  - report.md:    a ranked table plus per-candidate breakdowns and gaps.
"""

import json
from pathlib import Path

from src.models import ProcessingError, ScoredCandidate, UsageAccumulator

# Prices per million tokens. Approximations; verify at https://anthropic.com/pricing
_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00, "cache_write": 3.75,  "cache_read": 0.30},
    "claude-opus-4-7":           {"input": 15.00, "output": 75.00, "cache_write": 18.75, "cache_read": 1.50},
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00,  "cache_write": 1.00,  "cache_read": 0.08},
}
_DEFAULT_PRICING = _PRICING["claude-sonnet-4-6"]

Result = ScoredCandidate | ProcessingError


def format_cost_report(acc: UsageAccumulator, model: str) -> str:
    """Return a plain-text cost breakdown suitable for printing to stdout."""
    p = _PRICING.get(model, _DEFAULT_PRICING)
    M = 1_000_000
    input_cost       = acc.input_tokens          * p["input"]       / M
    output_cost      = acc.output_tokens         * p["output"]      / M
    cache_write_cost = acc.cache_creation_tokens * p["cache_write"] / M
    cache_read_cost  = acc.cache_read_tokens      * p["cache_read"]  / M
    total            = input_cost + output_cost + cache_write_cost + cache_read_cost

    note = "" if model in _PRICING else f" (pricing approximated from {next(iter(_PRICING))})"
    lines = [
        f"Cost breakdown  model: {model}{note}",
        f"  Input tokens      : {acc.input_tokens:>10,}   ${input_cost:.4f}",
        f"  Output tokens     : {acc.output_tokens:>10,}   ${output_cost:.4f}",
        f"  Cache writes      : {acc.cache_creation_tokens:>10,}   ${cache_write_cost:.4f}",
        f"  Cache reads (hits): {acc.cache_read_tokens:>10,}   ${cache_read_cost:.4f}",
        f"  {'─' * 44}",
        f"  Total             : {acc.input_tokens + acc.output_tokens:>10,}   ${total:.4f}",
    ]
    return "\n".join(lines)


def _split(results: list[Result]) -> tuple[list[ScoredCandidate], list[ProcessingError]]:
    candidates = [r for r in results if isinstance(r, ScoredCandidate)]
    errors = [r for r in results if isinstance(r, ProcessingError)]
    candidates.sort(key=lambda c: c.overall_fit.score, reverse=True)
    return candidates, errors


def write_json(results: list[Result], path: str | Path) -> None:
    """Write {candidates, errors} to a JSON file. Candidates sorted by overall_fit desc."""

    candidates, errors = _split(results)
    payload = {
        "candidates": [c.model_dump() for c in candidates],
        "errors": [e.model_dump() for e in errors],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def write_markdown(
    results: list[Result],
    path: str | Path,
    jd_path: str | Path,
    acc: UsageAccumulator | None = None,
    model: str | None = None,
) -> None:
    """Write a markdown report: ranked table + per-candidate breakdowns + errors.

    If acc and model are supplied, a cost breakdown section is appended.
    """

    candidates, errors = _split(results)

    lines: list[str] = []
    lines.append("# Resume Match Report")
    lines.append("")
    lines.append(f"Job description: `{jd_path}`")
    lines.append(f"Candidates scored: {len(candidates)}  ·  Failed: {len(errors)}")
    lines.append("")

    lines.append("## Ranked candidates")
    lines.append("")
    if candidates:
        lines.append("| Rank | Name | Overall | Skills | Experience | Role | Source |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for i, c in enumerate(candidates, start=1):
            lines.append(
                f"| {i} "
                f"| {c.profile.name} "
                f"| {c.overall_fit.score} "
                f"| {c.skills_match.score} "
                f"| {c.experience_match.score} "
                f"| {c.role_relevance.score} "
                f"| {c.source_file} |"
            )
    else:
        lines.append("_No candidates scored._")
    lines.append("")

    for c in candidates:
        lines.append(f"### {c.profile.name} — {c.overall_fit.score}")
        lines.append("")
        lines.append(f"_Source: `{c.source_file}` · {c.profile.years_experience} yrs experience_")
        lines.append("")
        lines.append(c.reasoning)
        lines.append("")
        lines.append("**Score breakdown**")
        lines.append("")
        lines.append(f"- Skills ({c.skills_match.score}): {c.skills_match.reasoning}")
        lines.append(f"- Experience ({c.experience_match.score}): {c.experience_match.reasoning}")
        lines.append(f"- Role relevance ({c.role_relevance.score}): {c.role_relevance.reasoning}")
        lines.append(f"- Overall fit ({c.overall_fit.score}): {c.overall_fit.reasoning}")
        lines.append("")
        if c.gaps:
            lines.append("**Gaps**")
            lines.append("")
            for gap in c.gaps:
                lines.append(f"- _{gap.category}_ — {gap.detail}")
            lines.append("")

    if errors:
        lines.append("## Could not process")
        lines.append("")
        lines.append("| Source | Stage | Message |")
        lines.append("| --- | --- | --- |")
        for e in errors:
            # Replace pipes in the message so they don't break the table layout.
            msg = e.message.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {e.source_file} | {e.stage} | {msg} |")
        lines.append("")

    if acc is not None and model is not None:
        p = _PRICING.get(model, _DEFAULT_PRICING)
        M = 1_000_000
        input_cost       = acc.input_tokens          * p["input"]       / M
        output_cost      = acc.output_tokens         * p["output"]      / M
        cache_write_cost = acc.cache_creation_tokens * p["cache_write"] / M
        cache_read_cost  = acc.cache_read_tokens      * p["cache_read"]  / M
        total            = input_cost + output_cost + cache_write_cost + cache_read_cost

        note = "" if model in _PRICING else f" *(pricing approximated from `{next(iter(_PRICING))}`)*"
        lines.append("## Cost breakdown")
        lines.append("")
        lines.append(f"Model: `{model}`{note}")
        lines.append("")
        lines.append("| Token type | Tokens | Cost (USD) |")
        lines.append("| --- | ---: | ---: |")
        lines.append(f"| Input             | {acc.input_tokens:,} | ${input_cost:.4f} |")
        lines.append(f"| Output            | {acc.output_tokens:,} | ${output_cost:.4f} |")
        lines.append(f"| Cache writes      | {acc.cache_creation_tokens:,} | ${cache_write_cost:.4f} |")
        lines.append(f"| Cache reads (hits)| {acc.cache_read_tokens:,} | ${cache_read_cost:.4f} |")
        lines.append(f"| **Total**         | **{acc.input_tokens + acc.output_tokens:,}** | **${total:.4f}** |")
        lines.append("")

    Path(path).write_text("\n".join(lines))
