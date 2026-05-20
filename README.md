# Resume-to-JD Matching Agent

A teaching project that scores resume PDFs against a job description using
the Claude API.

Each resume goes through three steps, each a separate Python module:

1. **Parse** the PDF into plain text (`pdf_parser.py`, no LLM)
2. **Extract** structured info — name, skills, past roles, years of experience
   (`extractor.py`, one LLM call)
3. **Score** the candidate against the JD across four dimensions
   (`scorer.py`, one LLM call)

The output is a ranked JSON file plus a readable markdown report.

## How the code runs (end to end)

### Step 1: Entry point (`main.py`)

```
python main.py --jd sample_data/sample_jd.txt --resumes sample_data/resumes/
```

Python hits `if __name__ == "__main__"` and calls `main()`.

### Step 2: Parse arguments and validate

`main()` reads the command-line args (`--jd`, `--resumes`, `--output`, `--model`), then checks:

- Is `ANTHROPIC_API_KEY` set?
- Does the JD file exist?
- Does the resumes directory exist?

If all good, it calls `run()`.

### Step 3: Setup

`run()` does two things:

1. **Reads the job description** from the text file into a string using `open()`
2. **Lists the resumes directory** using `os.listdir()` and collects all `.pdf` files

Then it creates an Anthropic API client and starts looping through each PDF.

### Step 4: Process each resume

For each PDF, `process_one()` runs a 3-stage pipeline. If any stage fails,
it records the error and moves on to the next resume.

**Stage A — Parse the PDF** (`src/pdf_parser.py`)

- Uses `pdfplumber` to extract raw text from the PDF
- No LLM involved here, just text extraction
- Input: `alice_chen.pdf` -> Output: a big string of resume text

**Stage B — Extract candidate profile** (`src/extractor.py`) — LLM call #1

- Sends the resume text to Claude with a system prompt saying "you are a resume parser"
- Uses **tool use**: defines a tool called `record_candidate` whose schema matches
  `CandidateProfile` (name, skills, years_experience, past_roles, education)
- Forces Claude to call that tool via `tool_choice` — so Claude **must** return
  structured JSON, not free text
- Validates the JSON with Pydantic to get a typed `CandidateProfile` object
- Input: raw text -> Output: structured profile like `{name: "Alice Chen", skills: ["Python", "ML"], ...}`

**Stage C — Score against the JD** (`src/scorer.py`) — LLM call #2

- Sends both the job description and the candidate profile to Claude
- System prompt says "you are a hiring evaluator"
- Again uses tool use: a tool called `record_score` matching `ScoreReport` schema
- Claude scores 4 dimensions (0-100 each): skills_match, experience_match, role_relevance, overall_fit
- Also returns reasoning and gaps
- Input: JD + profile -> Output: `ScoredCandidate` with scores like `{overall_fit: 85, skills_match: 90, ...}`

**Stage D — Self-critique (optional)** (`src/critique.py`) — LLM call #3

- Only runs if you pass `--self-critique`
- Sends the JD + profile + the scores from Stage C to Claude as a "senior hiring manager"
- Claude reviews the scores and decides: are they reasonable, or off by more than 10 points?
- If `did_revise: false` -> keeps original scores
- If `did_revise: true` -> returns corrected scores

### Step 5: Write output (`src/reporter.py`)

After all resumes are processed, `run()`:

1. Creates the output directory
2. Writes `results.json` — all candidates ranked by overall_fit score, plus any errors
3. Writes `report.md` — a readable markdown table with rankings, per-candidate score breakdowns, gaps, and a **cost breakdown table** at the end
4. Prints a cost breakdown to stdout — input tokens, output tokens, cache writes, cache reads, and total USD

### The big picture

```
PDF file
   |  pdfplumber
Raw text
   |  Claude + tool use (LLM #1)  ─┐
Structured profile                  │  UsageAccumulator.add(response.usage)
   |  Claude + tool use (LLM #2)  ─┤  accumulates across every call
Scores + reasoning                  │
   |  Claude + tool use (LLM #3, optional) ─┘
Reviewed scores
   |
results.json + report.md + cost breakdown (stdout + report.md)
```

The key pattern that repeats in every LLM call: **define a tool from a Pydantic
schema -> force Claude to call it -> validate the response back into a Pydantic
object**. This guarantees structured, typed output every time.

## What's being taught

- Reading structured data out of LLM responses using Claude's **tool use**
  feature + **Pydantic** for validation.
- A **prompt** module where all LLM instructions live in one place, so you
  can iterate on wording without touching the surrounding code.
- **Prompt caching** — the JD is identical across every resume in a run, so
  Claude serves it from cache at ~10% of the normal cost.
- **Cost tracking** — every API response carries a `usage` object with input
  tokens, output tokens, cache write tokens, and cache read tokens. A
  `UsageAccumulator` collects these across the whole run so you can see
  exactly what a batch cost in USD.
- An optional **self-critique** pass behind a flag that reviews scores.
- An **eval harness** for testing LLM pipelines, because you can't assert
  exact-score equality on a probabilistic output.

No asyncio, no frameworks, no agent libraries. Everything is plain
synchronous Python you can read top-to-bottom in one sitting.

## Setup

```bash
uv sync
```

Create a `.env` file in the project root with your API key:

```
ANTHROPIC_API_KEY=your-api-key-here
```

Generate the sample resumes (they're not checked in):

```bash
uv run python scripts/generate_sample_resumes.py
```

This creates five fictional candidates in `sample_data/resumes/`. Edit
[scripts/generate_sample_resumes.py](scripts/generate_sample_resumes.py)
and re-run to experiment.

## Run

```bash
python main.py \
    --jd sample_data/sample_jd.txt \
    --resumes sample_data/resumes/ \
    --output output/
```

Flags:

- `--model MODEL_ID` — override the Claude model (default: `claude-sonnet-4-6`)
- `--self-critique` — add one more LLM call per resume that reviews and
  may revise the scores. +1 call per resume.

Outputs:

- `output/results.json` — machine-readable ranked list plus any errors
- `output/report.md` — human-readable markdown report with a cost breakdown table at the end
- stdout — cost breakdown printed after the run (input tokens, output tokens, cache writes, cache reads, total USD)

## Test

Fast local tests for the deterministic parts of the pipeline (parser and
reporter) — no LLM calls:

```bash
uv run pytest
```

## Evals

LLM calls can't be unit-tested with exact equality — scores move a few
points call to call. So we test *properties* of the output: a strong-match
resume should score 80+ overall, a weak-match resume should score under 45,
and so on.

```bash
uv run python -m src.evals
```

Runs the golden cases in [tests/evals/cases.py](tests/evals/cases.py)
through the real `extract -> score` pipeline and checks each dimension
against an expected range. Exits nonzero on failure.

Add a case by appending to `ALL_CASES` at the bottom of that file. When
you iterate on a prompt, run the evals to see whether the change moved
scores in the right direction.

## Project layout

```
resume_matcher/
├── main.py              # CLI entry point — loops over resumes one at a time
├── pyproject.toml
├── src/
│   ├── pdf_parser.py    # PDF -> plain text (no LLM)
│   ├── extractor.py     # text -> CandidateProfile (LLM call)
│   ├── scorer.py        # (profile, JD) -> ScoredCandidate (LLM call)
│   ├── critique.py      # optional 2nd-pass review (LLM call)
│   ├── reporter.py      # results -> JSON + markdown
│   ├── prompts.py       # every LLM prompt in one place
│   ├── models.py        # Pydantic contracts
│   └── evals.py         # eval harness runner
├── tests/
│   ├── test_pdf_parser.py
│   ├── test_reporter.py
│   └── evals/
│       └── cases.py     # golden eval cases
├── sample_data/
│   ├── sample_jd.txt
│   └── resumes/         # generated by scripts/generate_sample_resumes.py
├── scripts/
│   └── generate_sample_resumes.py
└── output/
```

## Design notes

- **One LLM call per stage, not per field.** Structured output via tool use
  means one call returns every field at once. Students see the full
  `messages.create()` call with `tools`, `tool_choice`, and `model_validate`
  in one place (`extractor.py`).
- **Prompts are data, code is logic.** All prompts live in
  [src/prompts.py](src/prompts.py). You can change wording without touching
  any function.
- **Bad PDFs don't crash the run.** Each resume is wrapped in try/except. A
  failure at any stage becomes a `ProcessingError` in the output instead of
  stopping the batch.
- **Opt-in quality flag.** `--self-critique` is off by default so a basic
  run costs two LLM calls per resume. Turn it on when you want to add a
  reflection pass.
- **Cost visibility.** Every LLM function accepts an optional `UsageAccumulator`.
  The entry points create one per run and pass it through the whole pipeline.
  At the end, `reporter.format_cost_report()` converts accumulated token counts
  to USD using a pricing table in `reporter.py` — update that table if rates change.
