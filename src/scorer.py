"""
Second LLM call: score a CandidateProfile against a job description.

Same tool-use + Pydantic pattern as extractor.py — see that file for the
three-beat explanation. The one twist here is prompt caching: the JD is
sent as a separate content block tagged with `cache_control: ephemeral`
so that scoring N resumes against the same JD only pays the JD's input
tokens once.
"""

from statistics import median

from anthropic import Anthropic

from src.models import (
    CandidateProfile,
    DimensionScore,
    ScoreReport,
    ScoredCandidate,
    UsageAccumulator,
)
from src.prompts import SCORING_SYSTEM_PROMPT

DEFAULT_MODEL = "claude-sonnet-4-6"
SCORING_TOOL_NAME = "record_score"


def _score_report(
    client: Anthropic,
    profile: CandidateProfile,
    jd: str,
    model: str,
    acc: UsageAccumulator | None = None,
) -> ScoreReport:
    """One scoring LLM call. Returns the raw ScoreReport."""

    tool = {
        "name": SCORING_TOOL_NAME,
        "description": "Record the score for a candidate against the job description.",
        "input_schema": ScoreReport.model_json_schema(),
    }

    profile_text = profile.model_dump_json(indent=2)

    # Two blocks: JD (cached so re-scoring against the same JD is cheap)
    # then the per-candidate profile JSON.
    content = [
        {
            "type": "text",
            "text": f"<job_description>\n{jd}\n</job_description>",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": f"<candidate_profile>\n{profile_text}\n</candidate_profile>",
        },
    ]

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SCORING_SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": SCORING_TOOL_NAME},
        messages=[{"role": "user", "content": content}],
    )

    if acc is not None:
        acc.add(response.usage)

    tool_use = next(block for block in response.content if block.type == "tool_use")
    return ScoreReport.model_validate(tool_use.input)


def score_candidate(
    client: Anthropic,
    profile: CandidateProfile,
    jd: str,
    source_file: str,
    model: str = DEFAULT_MODEL,
    acc: UsageAccumulator | None = None,
) -> ScoredCandidate:
    """Score a candidate profile against a JD. Single LLM call."""

    report = _score_report(client, profile, jd, model, acc=acc)
    return ScoredCandidate(
        profile=profile,
        skills_match=report.skills_match,
        experience_match=report.experience_match,
        role_relevance=report.role_relevance,
        overall_fit=report.overall_fit,
        reasoning=report.reasoning,
        gaps=report.gaps,
        source_file=source_file,
    )


def score_candidate_ensemble(
    client: Anthropic,
    profile: CandidateProfile,
    jd: str,
    source_file: str,
    n: int = 3,
    model: str = DEFAULT_MODEL,
    acc: UsageAccumulator | None = None,
) -> ScoredCandidate:
    """Run scoring N times and return per-dimension medians.

    Reasoning text is taken from the first run only — merging prose across
    runs produces worse output than just picking one.
    """

    if n < 1:
        raise ValueError("n must be >= 1")

    reports = [_score_report(client, profile, jd, model, acc=acc) for _ in range(n)]
    first = reports[0]

    def _median_dim(attr: str) -> DimensionScore:
        scores = [int(median([getattr(r, attr).score for r in reports]))]
        return DimensionScore(
            score=scores[0],
            reasoning=getattr(first, attr).reasoning,
        )

    return ScoredCandidate(
        profile=profile,
        skills_match=_median_dim("skills_match"),
        experience_match=_median_dim("experience_match"),
        role_relevance=_median_dim("role_relevance"),
        overall_fit=_median_dim("overall_fit"),
        reasoning=first.reasoning,
        gaps=first.gaps,
        source_file=source_file,
    )
