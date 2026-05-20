"""
Optional third LLM call: a "critic" reviews the scores and fixes obvious
errors.

Same pattern as extractor.py and scorer.py — we give Claude a tool whose
input_schema is a Pydantic model (CritiqueReport). The critic prompt tells
Claude to default to agreement and only revise when it sees a clear error.

If the critic says `did_revise: false`, we return the original scores
unchanged. Otherwise we build a new ScoredCandidate from the revised ones.
"""

from anthropic import Anthropic

from src.models import CritiqueReport, ScoredCandidate, ScoreReport, UsageAccumulator
from src.prompts import CRITIQUE_SYSTEM_PROMPT

DEFAULT_MODEL = "claude-sonnet-4-6"
CRITIQUE_TOOL_NAME = "record_critique"


def critique_and_maybe_revise(
    client: Anthropic,
    scored: ScoredCandidate,
    jd: str,
    model: str = DEFAULT_MODEL,
    acc: UsageAccumulator | None = None,
) -> ScoredCandidate:
    """Review a scored candidate. Return the original or a revised copy."""

    tool = {
        "name": CRITIQUE_TOOL_NAME,
        "description": "Record whether the original scores are defensible, and any revisions.",
        "input_schema": CritiqueReport.model_json_schema(),
    }

    # We rebuild a ScoreReport so we can show the critic exactly what the
    # scorer output, serialized cleanly.
    original = ScoreReport(
        skills_match=scored.skills_match,
        experience_match=scored.experience_match,
        role_relevance=scored.role_relevance,
        overall_fit=scored.overall_fit,
        reasoning=scored.reasoning,
        gaps=scored.gaps,
    )

    user_content = [
        # Same JD caching trick as scorer.py — if the critic runs on many
        # candidates, the JD tokens are served from cache after call #1.
        {
            "type": "text",
            "text": f"<job_description>\n{jd}\n</job_description>",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": f"<candidate_profile>\n{scored.profile.model_dump_json(indent=2)}\n</candidate_profile>",
        },
        {
            "type": "text",
            "text": f"<original_scores>\n{original.model_dump_json(indent=2)}\n</original_scores>",
        },
    ]

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=CRITIQUE_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": CRITIQUE_TOOL_NAME},
        messages=[{"role": "user", "content": user_content}],
    )

    if acc is not None:
        acc.add(response.usage)

    tool_use = next(block for block in response.content if block.type == "tool_use")
    report = CritiqueReport.model_validate(tool_use.input)

    if not report.did_revise:
        return scored

    revised = report.revised_scores
    return ScoredCandidate(
        profile=scored.profile,
        source_file=scored.source_file,
        skills_match=revised.skills_match,
        experience_match=revised.experience_match,
        role_relevance=revised.role_relevance,
        overall_fit=revised.overall_fit,
        reasoning=revised.reasoning,
        gaps=revised.gaps,
    )
