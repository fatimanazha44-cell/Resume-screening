"""
First LLM call: turn raw resume text into a validated CandidateProfile.

This module establishes the tool-use + Pydantic pattern that scorer.py and
critique.py both follow. Read this file first and the other two will make
sense quickly — they are variations on the same three beats:

    1. Build a tool whose `input_schema` is a Pydantic model's JSON schema.
       Claude treats that schema as the contract for the tool's arguments.
    2. Force Claude to call exactly that tool via `tool_choice`. This turns
       a free-form chat call into a structured-output call: the response is
       guaranteed to contain a tool_use block with JSON that fits the schema.
    3. Pull the tool_use block out of `response.content` and hand its
       `input` dict to `Model.model_validate(...)`. Pydantic raises if the
       LLM returned anything malformed, so the rest of the pipeline can
       trust the object it gets back.

Prompts live in src/prompts.py, never inlined here.
"""

from anthropic import Anthropic

from src.models import CandidateProfile, UsageAccumulator
from src.prompts import EXTRACTION_SYSTEM_PROMPT

DEFAULT_MODEL = "claude-sonnet-4-6"
EXTRACTION_TOOL_NAME = "record_candidate"


def extract_candidate(
    client: Anthropic,
    resume_text: str,
    model: str = DEFAULT_MODEL,
    acc: UsageAccumulator | None = None,
) -> CandidateProfile:
    """Extract a structured CandidateProfile from raw resume text."""

    # The tool's input_schema IS the Pydantic schema. Claude will fill in
    # arguments that conform to it, which is how we get structured output
    # without parsing free text.
    tool = {
        "name": EXTRACTION_TOOL_NAME,
        "description": "Record structured information extracted from the resume.",
        "input_schema": CandidateProfile.model_json_schema(),
    }

    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=EXTRACTION_SYSTEM_PROMPT,
        tools=[tool],
        # Forcing this exact tool removes the "should I call a tool?" branch.
        # The response is guaranteed to contain a tool_use block for it.
        tool_choice={"type": "tool", "name": EXTRACTION_TOOL_NAME},
        messages=[{"role": "user", "content": resume_text}],
    )

    if acc is not None:
        acc.add(response.usage)

    tool_use = next(block for block in response.content if block.type == "tool_use")
    return CandidateProfile.model_validate(tool_use.input)
