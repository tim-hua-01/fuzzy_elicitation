#!/usr/bin/env python3
"""
Grade philosophical answers using any LLM model, with support for Anthropic extended thinking.

Anthropic models automatically use extended thinking with a default budget of 1500 tokens.
Use --reasoning-budget to customize or set to 0 to disable.

Usage:
    # Grade with Anthropic (uses default 1500 token reasoning budget)
    python grade_with_model.py --run v2_tryhard_norubric --grader anthropic/claude-sonnet-4-5
    
    # Grade with Anthropic and custom reasoning budget
    python grade_with_model.py --run v2_tryhard_norubric --grader anthropic/claude-sonnet-4-5 --reasoning-budget 10000
    
    # Grade with OpenAI
    python grade_with_model.py --run v2_tryhard_norubric --grader openai/gpt-4o
    
    # Grade with OpenRouter
    python grade_with_model.py --run v2_tryhard_norubric --grader openrouter/anthropic/claude-3.5-sonnet
    
    # Grade human baselines
    python grade_with_model.py --human --grader anthropic/claude-sonnet-4-5
"""

import argparse
import asyncio
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

MAX_CONCURRENT = 25


def get_rubric_version(rubric_name: str) -> str:
    """Extract version from rubric name (e.g., 'response_rubric_v2' -> 'v2')."""
    if "_v" in rubric_name:
        return "v" + rubric_name.split("_v")[-1]
    return "v1"


def load_rubric(rubric_name: str = "response_rubric_v2") -> str:
    """Load the full rubric including output format."""
    rubric_path = Path(__file__).parent / f"{rubric_name}.md"
    if not rubric_path.exists():
        rubric_path = Path(__file__).parent / "response_rubric.md"

    with open(rubric_path) as f:
        return f.read()


def build_grading_prompt(question: str, answer: str, rubric: str) -> str:
    """Build the prompt for grading."""
    return f"""You are grading a philosophical answer for quality of argumentation and exposition.

## Question

{question}

## Answer to Grade

{answer}

## Grading Rubric

{rubric}"""


def parse_grade_response(response_text: str) -> dict:
    """Parse the JSON grade response from the LLM."""
    text = response_text.strip()
    # First: try raw
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    text = text.strip()
    return json.loads(text)


async def grade_anthropic(
    semaphore: asyncio.Semaphore,
    client: AsyncAnthropic,
    prompt: str,
    model: str,
    reasoning_budget: int | None = None,
) -> dict[str, str | None]:
    """Grade using Anthropic API with optional extended thinking."""
    model_name = model.replace("anthropic/", "")

    create_params = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16000,
        "temperature": 1,
    }
    
    if reasoning_budget:
        create_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": reasoning_budget
        }
        create_params["max_tokens"] = max(16000, reasoning_budget + 5000)

    async with semaphore:
        response = await client.messages.create(**create_params)

    # Extract text content from response
    content_text = ""
    thinking_text = None
    
    for block in response.content:
        if block.type == "text":
            content_text += block.text
        elif block.type == "thinking":
            thinking_text = block.thinking

    return {
        "content": content_text,
        "reasoning": thinking_text,
    }


async def grade_openai(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    prompt: str,
    model: str,
) -> dict[str, str | None]:
    """Grade using OpenAI API."""
    model_name = model.replace("openai/", "")

    async with semaphore:
        response = await client.responses.create(
            model=model_name,
            input=[{"role": "user", "content": prompt}],
            temperature=1, 
        )

    return {
        "content": response.output_text,
        "reasoning": None,
    }


async def grade_openrouter(
    semaphore: asyncio.Semaphore,
    http_client: httpx.AsyncClient,
    api_key: str,
    prompt: str,
    model: str,
    provider: str | None = None,
) -> dict[str, str | None]:
    """Grade using OpenRouter API."""
    model_name = model.replace("openrouter/", "")

    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 1,
        }

        if provider:
            payload["provider"] = {"order": [provider]}

        response = await http_client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        message = data["choices"][0]["message"]
        return {
            "content": message["content"],
            "reasoning": message.get("reasoning"),
        }


MAX_RETRIES = 3


async def grade_single(
    entry: dict,
    grader_model: str,
    rubric: str,
    anthropic_client: AsyncAnthropic | None,
    openai_client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient | None,
    openrouter_key: str | None,
    semaphore: asyncio.Semaphore,
    provider: str | None = None,
    reasoning_budget: int | None = None,
) -> dict:
    """Grade a single answer entry with retries for parsing failures."""
    prompt = build_grading_prompt(entry["question"], entry["answer"], rubric)
    is_anthropic = grader_model.startswith("anthropic/")
    is_openai = grader_model.startswith("openai/")

    last_error = None
    last_response = None

    for attempt in range(MAX_RETRIES):
        try:
            if is_anthropic:
                response_data = await grade_anthropic(semaphore, anthropic_client, prompt, grader_model, reasoning_budget)
            elif is_openai:
                response_data = await grade_openai(semaphore, openai_client, prompt, grader_model)
            else:
                response_data = await grade_openrouter(semaphore, http_client, openrouter_key, prompt, grader_model, provider)

            last_response = response_data.get("content", "")
            scores = parse_grade_response(response_data["content"])

            # Ensure total is calculated
            if "total" not in scores:
                scores["total"] = sum(v for k, v in scores.items() if isinstance(v, (int, float)) and k != "total")

            result = {
                "question_id": entry["question_id"],
                "model": entry.get("model", "human"),
                "prompt_variant": entry.get("prompt_variant", ""),
                "sample_idx": entry.get("sample_idx", 0),
                "is_human": entry.get("is_human", False),
                "grader_model": grader_model,
                "scores": scores,
                "error": None,
            }

            # Include reasoning if present
            if response_data.get("reasoning"):
                result["grader_reasoning"] = response_data["reasoning"]

            # Note if retries were needed
            if attempt > 0:
                result["retries"] = attempt

            return result

        except json.JSONDecodeError as e:
            last_error = f"JSON parse error (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(0.5)
            continue

        except Exception as e:
            last_error = str(e)
            break

    # All retries failed
    return {
        "question_id": entry["question_id"],
        "model": entry.get("model", "human"),
        "prompt_variant": entry.get("prompt_variant", ""),
        "sample_idx": entry.get("sample_idx", 0),
        "is_human": entry.get("is_human", False),
        "grader_model": grader_model,
        "scores": None,
        "error": last_error,
        "last_response": last_response[:500] if last_response else None,
    }


def sanitize_model_name(model: str) -> str:
    """Convert model name to safe filename (e.g., 'anthropic/claude-sonnet-4-5' -> 'anthropic_claude-sonnet-4-5')."""
    return model.replace("/", "_").replace(":", "_")


def append_to_csv(results: list[dict], csv_path: Path):
    """Append grading results to a CSV file with character counts."""
    # Get score keys from first valid result
    score_keys = []
    for r in results:
        if r.get("scores"):
            score_keys = [k for k in r["scores"].keys() if k != "total"]
            break

    # Build fieldnames dynamically with character counts
    base_fields = ["question_id", "answer", "model", "prompt_variant", "sample_idx", "is_human", "grader_model"]
    fieldnames = base_fields + score_keys + ["total", "timestamp", "answer_char_count", "reasoning_char_count"]

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    timestamp = datetime.now().isoformat()

    # If file exists, read existing fieldnames to ensure consistency
    if file_exists:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_fields = reader.fieldnames or []
            # Merge any new score fields
            for field in fieldnames:
                if field not in existing_fields:
                    # Insert new fields before timestamp
                    if field in ["answer_char_count", "reasoning_char_count"]:
                        existing_fields.append(field)
                    else:
                        idx = existing_fields.index("timestamp") if "timestamp" in existing_fields else len(existing_fields)
                        existing_fields.insert(idx, field)
            fieldnames = existing_fields

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for r in results:
            if r["scores"] is None:
                continue

            # Calculate character counts
            answer_text = r.get("answer", "")
            reasoning_text = r.get("reasoning", "")
            answer_char_count = len(answer_text) if answer_text else 0
            reasoning_char_count = len(reasoning_text) if reasoning_text else 0

            row = {
                "question_id": r["question_id"],
                "answer": answer_text,  # Keep full text in CSV
                "model": r["model"],
                "prompt_variant": r["prompt_variant"],
                "sample_idx": r["sample_idx"],
                "is_human": r["is_human"],
                "grader_model": r["grader_model"],
                "timestamp": timestamp,
                "answer_char_count": answer_char_count,
                "reasoning_char_count": reasoning_char_count,
            }

            # Add all score fields dynamically with field name mapping
            for key, value in r["scores"].items():
                # Map known field name variations
                if key == "precision_in_distinctions":
                    key = "precision_distinctions"
                row[key] = value

            # Filter out any keys not in fieldnames to prevent crash
            filtered_row = {k: v for k, v in row.items() if k in fieldnames}
            
            # Warn if we had to filter fields
            filtered_keys = set(row.keys()) - set(filtered_row.keys())
            if filtered_keys:
                print(f"Warning: Skipping unexpected fields for question {r['question_id']}: {filtered_keys}")
            
            writer.writerow(filtered_row)


async def grade_run(
    run_name: str,
    grader_model: str,
    provider: str | None = None,
    rubric_name: str = "response_rubric_v2",
    reasoning_budget: int | None = None,
):
    """Grade all answers from a run."""
    base_dir = Path(__file__).parent
    rubric_version = get_rubric_version(rubric_name)
    run_dir = base_dir / "data" / rubric_version / "runs" / run_name
    answers_file = run_dir / "answers.jsonl"

    if not answers_file.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_file}")

    # Load answers
    entries = []
    with open(answers_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    # Nice config printout
    print("\n" + "=" * 50)
    print("GRADE ANSWERS WITH MODEL")
    print("=" * 50)
    print(f"  Run:              {run_name}")
    print(f"  Grader:           {grader_model}")
    print(f"  Provider:         {provider or '(default)'}")
    print(f"  Rubric:           {rubric_name}")
    print(f"  Reasoning budget: {reasoning_budget or '(disabled)'}")
    print(f"  Answers:          {len(entries)}")
    print("=" * 50 + "\n")

    # Setup grader
    rubric = load_rubric(rubric_name)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    is_anthropic = grader_model.startswith("anthropic/")
    is_openai = grader_model.startswith("openai/")

    anthropic_client = None
    openai_client = None
    http_client = None
    openrouter_key = None

    if is_anthropic:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        anthropic_client = AsyncAnthropic(api_key=api_key)
    elif is_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        openai_client = AsyncOpenAI(api_key=api_key)
    else:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        http_client = httpx.AsyncClient(timeout=240.0)

    try:
        # Grade all entries concurrently
        tasks = [
            grade_single(
                entry,
                grader_model,
                rubric,
                anthropic_client,
                openai_client,
                http_client,
                openrouter_key,
                semaphore,
                provider,
                reasoning_budget,
            )
            for entry in entries
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Grading", unit="ans")
    finally:
        # Properly close all async clients
        if anthropic_client:
            await anthropic_client.close()
        if openai_client:
            await openai_client.close()
        if http_client:
            await http_client.aclose()

    # Add answers and reasoning to results for CSV
    for r, entry in zip(results, entries):
        r["answer"] = entry["answer"]
        r["reasoning"] = entry.get("reasoning", "")

    # Save grades to run folder with model name suffix
    safe_model_name = sanitize_model_name(grader_model)
    grades_file = run_dir / f"grades_{safe_model_name}.jsonl"
    with open(grades_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Save CSV to version folder (e.g., data/v2/)
    csv_path = base_dir / "data" / rubric_version / f"all_results_{safe_model_name}.csv"
    append_to_csv(results, csv_path)

    # Summary
    success = sum(1 for r in results if r["scores"] is not None)
    errors = sum(1 for r in results if r["error"] is not None)
    avg_total = sum(r["scores"]["total"] for r in results if r["scores"]) / max(success, 1)

    print(f"\nDone! Grades saved to {grades_file}")
    print(f"CSV saved to {csv_path}")
    print(f"Success: {success}, Errors: {errors}")
    print(f"Average total score: {avg_total:.1f}")


async def grade_human_baselines(
    grader_model: str,
    provider: str | None = None,
    rubric_name: str = "response_rubric_v2",
    reasoning_budget: int | None = None,
):
    """Grade the human baseline answers."""
    base_dir = Path(__file__).parent
    rubric_version = get_rubric_version(rubric_name)
    questions_file = base_dir / "main_questions.jsonl"

    # Load questions with human answers
    entries = []
    with open(questions_file) as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                entries.append({
                    "question_id": Path(q["source_paper"]).stem,
                    "question": q["question"],
                    "answer": q["human_answer"],
                    "model": "human",
                    "prompt_variant": "",
                    "sample_idx": 0,
                    "is_human": True,
                })

    # Nice config printout
    print("\n" + "=" * 50)
    print("GRADE HUMAN BASELINES WITH MODEL")
    print("=" * 50)
    print(f"  Grader:           {grader_model}")
    print(f"  Provider:         {provider or '(default)'}")
    print(f"  Rubric:           {rubric_name}")
    print(f"  Reasoning budget: {reasoning_budget or '(disabled)'}")
    print(f"  Answers:          {len(entries)}")
    print("=" * 50 + "\n")

    # Setup grader
    rubric = load_rubric(rubric_name)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    is_anthropic = grader_model.startswith("anthropic/")
    is_openai = grader_model.startswith("openai/")

    anthropic_client = None
    openai_client = None
    http_client = None
    openrouter_key = None

    if is_anthropic:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        anthropic_client = AsyncAnthropic(api_key=api_key)
    elif is_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        openai_client = AsyncOpenAI(api_key=api_key)
    else:
        openrouter_key = os.environ.get("OPENROUTER_API_KEY")
        if not openrouter_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        http_client = httpx.AsyncClient(timeout=240.0)

    try:
        # Grade all entries concurrently
        tasks = [
            grade_single(
                entry,
                grader_model,
                rubric,
                anthropic_client,
                openai_client,
                http_client,
                openrouter_key,
                semaphore,
                provider,
                reasoning_budget,
            )
            for entry in entries
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Grading humans", unit="ans")
    finally:
        # Properly close all async clients
        if anthropic_client:
            await anthropic_client.close()
        if openai_client:
            await openai_client.close()
        if http_client:
            await http_client.aclose()

    # Add answers and reasoning to results for CSV
    for r, entry in zip(results, entries):
        r["answer"] = entry["answer"]
        r["reasoning"] = entry.get("reasoning", "")

    # Save to human_grades folder with model name suffix
    safe_model_name = sanitize_model_name(grader_model)
    human_grades_dir = base_dir / "data" / rubric_version / "human_grades"
    human_grades_dir.mkdir(parents=True, exist_ok=True)

    grades_file = human_grades_dir / f"grades_{safe_model_name}.jsonl"
    with open(grades_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Save CSV to version folder (e.g., data/v2/)
    csv_path = base_dir / "data" / rubric_version / f"all_results_{safe_model_name}.csv"
    append_to_csv(results, csv_path)

    # Summary
    success = sum(1 for r in results if r["scores"] is not None)
    errors = sum(1 for r in results if r["error"] is not None)
    avg_total = sum(r["scores"]["total"] for r in results if r["scores"]) / max(success, 1)

    print(f"\nDone! Grades saved to {grades_file}")
    print(f"CSV saved to {csv_path}")
    print(f"Success: {success}, Errors: {errors}")
    print(f"Average total score: {avg_total:.1f}")


async def main():
    parser = argparse.ArgumentParser(description="Grade philosophical answers using any LLM model with optional extended thinking")
    parser.add_argument("--run", help="Run name to grade (from data/vX/runs/)")
    parser.add_argument("--human", action="store_true", help="Grade human baselines instead")
    parser.add_argument("--grader", required=True, help="Grader model (e.g., anthropic/claude-sonnet-4-5, openai/gpt-4o, openrouter/...)")
    parser.add_argument("--provider", help="OpenRouter provider (e.g., together, openai)")
    parser.add_argument("--rubric", default="response_rubric_v2", help="Rubric file name without .md (default: response_rubric_v2)")
    parser.add_argument("--reasoning-budget", type=int, help="Anthropic extended thinking budget in tokens (default: 1500 for Anthropic models, e.g., 10000)")

    args = parser.parse_args()

    if args.reasoning_budget and not args.grader.startswith("anthropic/"):
        parser.error("--reasoning-budget can only be used with Anthropic models (grader must start with 'anthropic/')")

    # Set default reasoning budget for Anthropic models if not specified
    if args.grader.startswith("anthropic/") and args.reasoning_budget is None:
        args.reasoning_budget = 1500

    if args.human:
        await grade_human_baselines(args.grader, args.provider, args.rubric, args.reasoning_budget)
    elif args.run:
        await grade_run(args.run, args.grader, args.provider, args.rubric, args.reasoning_budget)
    else:
        parser.error("Must specify either --run or --human")


if __name__ == "__main__":
    asyncio.run(main())
