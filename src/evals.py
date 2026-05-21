"""
Eval harness for the extract + score pipeline.

Runs every case in `tests/evals/cases.py` through the real extractor and
scorer, compares each dimension to its expected score range, and prints
a pass/fail table. Exits nonzero on any failure.

Run: `uv run python -m src.evals`

Why ranges instead of exact scores: the LLM gives slightly different
numbers each time it's called. A strong match should *always* score 80+,
but 82 vs 88 is indistinguishable noise. Ranges test a property of the
output, not a frozen value — the right way to test an LLM pipeline.
"""

import os
import sys

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

from src.extractor import DEFAULT_MODEL, extract_candidate
from src.scorer import score_candidate
from tests.evals.cases import ALL_CASES, EvalCase


def run_case(client: Anthropic, case: EvalCase, model: str) -> bool:
    """Run one eval case. Return True if every dimension passed."""
    try:
        profile = extract_candidate(client, case.resume_text, model=model)
        scored = score_candidate(
            client, profile, case.jd, source_file=f"{case.name}.eval", model=model
        )
    except Exception as e:
        print(f"\n[FAIL] {case.name}")
        print(f"       {case.description}")
        print(f"       ERROR: {e}")
        return False

    dimensions = [
        ("skills_match", scored.skills_match.score, case.expected_skills_match),
        ("experience_match", scored.experience_match.score, case.expected_experience_match),
        ("role_relevance", scored.role_relevance.score, case.expected_role_relevance),
        ("overall_fit", scored.overall_fit.score, case.expected_overall_fit),
    ]

    all_passed = all(lo <= score <= hi for _, score, (lo, hi) in dimensions)
    banner = "PASS" if all_passed else "FAIL"

    print(f"\n[{banner}] {case.name}")
    print(f"       {case.description}")
    for name, score, (lo, hi) in dimensions:
        mark = "✓" if lo <= score <= hi else "✗"
        print(f"  {mark} {name:<20} {score:>3}  expected {lo}–{hi}")

    return all_passed


def main() -> int:
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set.")
        return 1

    client = Anthropic()
    model = DEFAULT_MODEL
    print(f"Running {len(ALL_CASES)} eval cases against {model}...")

    passed = sum(run_case(client, c, model) for c in ALL_CASES)

    print(f"\n{passed}/{len(ALL_CASES)} cases passed")
    return 0 if passed == len(ALL_CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
