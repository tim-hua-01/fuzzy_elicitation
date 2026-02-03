#!/usr/bin/env python3
"""
Grade philosophical answers using LLMs.

Usage:
    # Grade model answers from a run
    python grade_responses.py --run llama-3-70b_answer_with_rubric_20260129_120000 --grader openai/gpt-4o

    # Grade human baselines
    python grade_responses.py --human --grader openai/gpt-4o
"""

import argparse
import asyncio
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

MAX_CONCURRENT = 25


def get_rubric_version(rubric_name: str) -> str:
    """Extract version from rubric name (e.g., 'response_rubric_v2' -> 'v2')."""
    if "_v" in rubric_name:
        return "v" + rubric_name.split("_v")[-1]
    return "v1"  # default


def load_rubric(rubric_name: str = "response_rubric_v2") -> str:
    """Load the full rubric including output format."""
    rubric_path = Path(__file__).parent / f"{rubric_name}.md"
    if not rubric_path.exists():
        # Fallback to original
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
    # Try to extract JSON from the response
    text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    text = text.strip()
    return json.loads(text)


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
        "reasoning": None,  # OpenAI API doesn't expose reasoning tokens
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
    openai_client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient | None,
    openrouter_key: str | None,
    semaphore: asyncio.Semaphore,
    provider: str | None = None,
) -> dict:
    """Grade a single answer entry with retries for parsing failures."""
    prompt = build_grading_prompt(entry["question"], entry["answer"], rubric)
    is_openai = grader_model.startswith("openai/")

    last_error = None
    last_response = None

    for attempt in range(MAX_RETRIES):
        try:
            if is_openai:
                response_data = await grade_openai(semaphore, openai_client, prompt, grader_model)
            else:
                response_data = await grade_openrouter(semaphore, http_client, openrouter_key, prompt, grader_model, provider)

            last_response = response_data.get("content", "")
            scores = parse_grade_response(response_data["content"])

            # Ensure total is calculated (dynamic - sum all numeric values except 'total')
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
                await asyncio.sleep(0.5)  # Brief delay before retry
            continue

        except Exception as e:
            # Non-retryable error
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
        "last_response": last_response[:500] if last_response else None,  # For debugging
    }


def append_to_csv(results: list[dict], csv_path: Path):
    """Append grading results to the all_results.csv file. Dynamically handles score columns."""
    # Get score keys from first valid result
    score_keys = []
    for r in results:
        if r.get("scores"):
            score_keys = [k for k in r["scores"].keys() if k != "total"]
            break

    # Build fieldnames dynamically
    base_fields = ["question_id", "answer", "model", "prompt_variant", "sample_idx", "is_human", "grader_model"]
    fieldnames = base_fields + score_keys + ["total", "timestamp"]

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
                    existing_fields.insert(-1, field)  # Insert before timestamp
            fieldnames = existing_fields

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for r in results:
            if r["scores"] is None:
                continue

            row = {
                "question_id": r["question_id"],
                "answer": r.get("answer", "")[:500],  # Truncate for CSV
                "model": r["model"],
                "prompt_variant": r["prompt_variant"],
                "sample_idx": r["sample_idx"],
                "is_human": r["is_human"],
                "grader_model": r["grader_model"],
                "timestamp": timestamp,
            }

            # Add all score fields dynamically
            for key, value in r["scores"].items():
                row[key] = value

            writer.writerow(row)


async def grade_run(run_name: str, grader_model: str, provider: str | None = None, rubric_name: str = "response_rubric_v2"):
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
    print("GRADE ANSWERS")
    print("=" * 50)
    print(f"  Run:         {run_name}")
    print(f"  Grader:      {grader_model}")
    print(f"  Provider:    {provider or '(default)'}")
    print(f"  Rubric:      {rubric_name}")
    print(f"  Answers:     {len(entries)}")
    print("=" * 50 + "\n")

    # Setup grader
    rubric = load_rubric(rubric_name)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    is_openai = grader_model.startswith("openai/")

    openai_client = None
    http_client = None
    openrouter_key = None

    if is_openai:
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
            grade_single(entry, grader_model, rubric, openai_client, http_client, openrouter_key, semaphore, provider)
            for entry in entries
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Grading", unit="ans")
    finally:
        # Close http client if created
        if http_client:
            await http_client.aclose()

    # Add answers to results for CSV
    for r, entry in zip(results, entries):
        r["answer"] = entry["answer"]

    # Save grades to run folder
    grades_file = run_dir / "grades.jsonl"
    with open(grades_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Append to versioned CSV
    csv_path = base_dir / "data" / rubric_version / "all_results.csv"
    append_to_csv(results, csv_path)

    # Summary
    success = sum(1 for r in results if r["scores"] is not None)
    errors = sum(1 for r in results if r["error"] is not None)
    avg_total = sum(r["scores"]["total"] for r in results if r["scores"]) / max(success, 1)

    print(f"\nDone! Grades saved to {grades_file}")
    print(f"Results appended to {csv_path}")
    print(f"Success: {success}, Errors: {errors}")
    print(f"Average total score: {avg_total:.1f}")


async def grade_human_baselines(grader_model: str, provider: str | None = None, rubric_name: str = "response_rubric_v2"):
    """Grade the human baseline answers."""
    base_dir = Path(__file__).parent
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
    print("GRADE HUMAN BASELINES")
    print("=" * 50)
    print(f"  Grader:      {grader_model}")
    print(f"  Provider:    {provider or '(default)'}")
    print(f"  Rubric:      {rubric_name}")
    print(f"  Answers:     {len(entries)}")
    print("=" * 50 + "\n")

    # Setup grader
    rubric = load_rubric(rubric_name)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    is_openai = grader_model.startswith("openai/")

    openai_client = None
    http_client = None
    openrouter_key = None

    if is_openai:
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
            grade_single(entry, grader_model, rubric, openai_client, http_client, openrouter_key, semaphore, provider)
            for entry in entries
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Grading humans", unit="ans")
    finally:
        # Close http client if created
        if http_client:
            await http_client.aclose()

    # Add answers to results for CSV
    for r, entry in zip(results, entries):
        r["answer"] = entry["answer"]

    # Save to human_grades folder
    grader_name = grader_model.replace("/", "_")
    rubric_version = get_rubric_version(rubric_name)
    human_grades_dir = base_dir / "data" / rubric_version / "human_grades"
    human_grades_dir.mkdir(parents=True, exist_ok=True)

    grades_file = human_grades_dir / f"{grader_name}.jsonl"
    with open(grades_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Append to versioned CSV
    csv_path = base_dir / "data" / rubric_version / "all_results.csv"
    append_to_csv(results, csv_path)

    # Summary
    success = sum(1 for r in results if r["scores"] is not None)
    errors = sum(1 for r in results if r["error"] is not None)
    avg_total = sum(r["scores"]["total"] for r in results if r["scores"]) / max(success, 1)

    print(f"\nDone! Grades saved to {grades_file}")
    print(f"Success: {success}, Errors: {errors}")
    print(f"Average total score: {avg_total:.1f}")


async def main():
    parser = argparse.ArgumentParser(description="Grade philosophical answers using LLMs")
    parser.add_argument("--run", help="Run name to grade (from data/runs/)")
    parser.add_argument("--human", action="store_true", help="Grade human baselines instead")
    parser.add_argument("--grader", required=True, help="Grader model (e.g., openai/gpt-4o)")
    parser.add_argument("--provider", help="OpenRouter provider (e.g., together, openai)")
    parser.add_argument("--rubric", default="response_rubric_v2", help="Rubric file name without .md (default: response_rubric_v2)")

    args = parser.parse_args()

    if args.human:
        await grade_human_baselines(args.grader, args.provider, args.rubric)
    elif args.run:
        await grade_run(args.run, args.grader, args.provider, args.rubric)
    else:
        parser.error("Must specify either --run or --human")


if __name__ == "__main__":
    asyncio.run(main())
