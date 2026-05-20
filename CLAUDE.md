# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching project that scores resume PDFs against a job description using the Claude API. Each resume goes through a pipeline: PDF parse -> structured extraction (LLM) -> scoring (LLM) -> optional self-critique (LLM). Output is a ranked JSON file and markdown report.

## Commands

```bash
# Install dependencies
uv sync

# Generate sample resume PDFs (not checked in)
uv run python scripts/generate_sample_resumes.py

# Run the pipeline
python main.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/

# Override output directory (default ./output) or model (default DEFAULT_MODEL)
python main.py --jd ... --resumes ... --output out/ --model claude-sonnet-4-6

# Run with self-critique (adds one extra LLM call per resume)
python main.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/ --self-critique

# Run the Streamlit web UI
uv run streamlit run app.py

# Unit tests (no LLM calls, fast)
uv run pytest

# Run a single test
uv run pytest tests/test_pdf_parser.py

# Evals (makes real LLM calls, checks score ranges)
uv run python -m src.evals
```

## Environment

Requires `ANTHROPIC_API_KEY` in a `.env` file (loaded via `python-dotenv`). See `.env.example`.

## Architecture

**Core pattern repeated in every LLM module:** define a tool from a Pydantic model's JSON schema -> force Claude to call it via `tool_choice` -> validate the response back into a Pydantic object. This appears in `extractor.py`, `scorer.py`, and `critique.py`.

**Two entry points:**
- `main.py` — CLI that loops over resume PDFs sequentially
- `app.py` — Streamlit web UI with file upload

Both entry points define their own `process_one()` that wires the same parse → extract → score (→ critique) pipeline. They are **not** sharing a helper — if you change the pipeline shape, update both files.

**Pipeline stages (each in its own module under `src/`):**
1. `pdf_parser.py` — pdfplumber text extraction, no LLM
2. `extractor.py` — resume text -> `CandidateProfile` (LLM call #1, tool: `record_candidate`)
3. `scorer.py` — profile + JD -> `ScoredCandidate` (LLM call #2, tool: `record_score`). Uses prompt caching on the JD text block (`cache_control: ephemeral`)
4. `critique.py` — optional review pass (LLM call #3, tool: `record_critique`)
5. `reporter.py` — writes `results.json` and `report.md`, no LLM. Also exposes `format_cost_report()` for the stdout cost summary.

**Key files:**
- `src/models.py` — all Pydantic models (`CandidateProfile`, `ScoreReport`, `ScoredCandidate`, `CritiqueReport`, `ProcessingError`) plus `UsageAccumulator` (not a Pydantic model — a plain class that accumulates `response.usage` token counts across all LLM calls in a run). These are the contracts between modules.
- `src/prompts.py` — all LLM prompts in one place. Edit prompts here, not in the module files.
- `src/evals.py` — eval harness runner
- `tests/evals/cases.py` — golden eval cases with expected score ranges (not exact values)

## Design conventions

- Default model is `claude-sonnet-4-6`, set in `src/extractor.py` as `DEFAULT_MODEL`. `scorer.py` and `critique.py` redefine the same constant locally — keep them in sync.
- Errors in individual resumes produce a `ProcessingError` (`stage` is one of `parse`/`extract`/`score`) instead of crashing the batch. A failure inside the optional critique pass does **not** produce a `ProcessingError`; it logs and falls back to the pre-critique scores.
- `scorer.py` also exposes `score_candidate_ensemble()` which runs N scoring calls and returns median scores per dimension. Reasoning text is taken from the first run only — don't try to merge text across runs.
- **Cost tracking:** every LLM function (`extract_candidate`, `score_candidate`, `score_candidate_ensemble`, `critique_and_maybe_revise`) accepts an optional `acc: UsageAccumulator | None = None`. The entry points (`main.py`, `app.py`) create one accumulator per run and pass it through. After the run, `format_cost_report(acc, model)` prints a breakdown (input tokens, output tokens, cache writes, cache reads, total USD) to stdout, and `write_markdown` appends the same data as a `## Cost breakdown` table at the end of `report.md`. Pricing lives in `_PRICING` in `reporter.py`; update it there if rates change.
- Evals test score ranges (e.g., 80-100 for strong match) because LLM output varies between calls. Add new eval cases by appending to `ALL_CASES` in `tests/evals/cases.py`.
- Per `CONTRIBUTING.MD`, prompts are treated as the teaching artifact: any prompt change should be visible in the PR description (or saved under `prompts/`), and new conventions should be reflected back into this file.