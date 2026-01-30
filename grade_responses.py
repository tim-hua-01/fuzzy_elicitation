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

load_dotenv()

MAX_CONCURRENT = 10


def load_rubric() -> str:
    """Load the full rubric including output format."""
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
    client: AsyncOpenAI,
    prompt: str,
    model: str,
) -> str:
    """Grade using OpenAI API."""
    model_name = model.replace("openai/", "")

    response = await client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": prompt}],
        temperature=1, 
    )

    return response.output_text


async def grade_openrouter(
    semaphore: asyncio.Semaphore,
    api_key: str,
    prompt: str,
    model: str,
) -> str:
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
            "temperature": 1,  # Deterministic grading
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]


async def grade_single(
    entry: dict,
    grader_model: str,
    rubric: str,
    openai_client: AsyncOpenAI | None,
    openrouter_key: str | None,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Grade a single answer entry."""
    prompt = build_grading_prompt(entry["question"], entry["answer"], rubric)

    is_openai = grader_model.startswith("openai/")

    try:
        if is_openai:
            response_text = await grade_openai(openai_client, prompt, grader_model)
        else:
            response_text = await grade_openrouter(semaphore, openrouter_key, prompt, grader_model)

        scores = parse_grade_response(response_text)

        # Ensure total is calculated
        if "total" not in scores:
            score_keys = [
                "thesis_clarity", "charitable_engagement", "objection_handling",
                "example_quality", "precision_distinctions", "constructive_contribution",
                "argumentative_risk", "problem_reframing", "explanatory_unification",
                "scope_honesty"
            ]
            scores["total"] = sum(scores.get(k, 0) for k in score_keys)

        return {
            "question_id": entry["question_id"],
            "model": entry.get("model", "human"),
            "prompt_variant": entry.get("prompt_variant", ""),
            "sample_idx": entry.get("sample_idx", 0),
            "is_human": entry.get("is_human", False),
            "grader_model": grader_model,
            "scores": scores,
            "error": None,
        }

    except Exception as e:
        return {
            "question_id": entry["question_id"],
            "model": entry.get("model", "human"),
            "prompt_variant": entry.get("prompt_variant", ""),
            "sample_idx": entry.get("sample_idx", 0),
            "is_human": entry.get("is_human", False),
            "grader_model": grader_model,
            "scores": None,
            "error": str(e),
        }


def append_to_csv(results: list[dict], csv_path: Path):
    """Append grading results to the all_results.csv file."""
    fieldnames = [
        "question_id", "answer", "model", "prompt_variant", "sample_idx", "is_human",
        "grader_model", "thesis_clarity", "charitable_engagement", "objection_handling",
        "example_quality", "precision_distinctions", "constructive_contribution",
        "argumentative_risk", "problem_reframing", "explanatory_unification",
        "scope_honesty", "total", "timestamp"
    ]

    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    timestamp = datetime.now().isoformat()

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

            # Add score fields
            for key in [
                "thesis_clarity", "charitable_engagement", "objection_handling",
                "example_quality", "precision_distinctions", "constructive_contribution",
                "argumentative_risk", "problem_reframing", "explanatory_unification",
                "scope_honesty", "total"
            ]:
                row[key] = r["scores"].get(key, "")

            writer.writerow(row)


async def grade_run(run_name: str, grader_model: str):
    """Grade all answers from a run."""
    base_dir = Path(__file__).parent
    run_dir = base_dir / "data" / "runs" / run_name
    answers_file = run_dir / "answers.jsonl"

    if not answers_file.exists():
        raise FileNotFoundError(f"Answers file not found: {answers_file}")

    # Load answers
    entries = []
    with open(answers_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    print(f"Grading {len(entries)} answers from run: {run_name}")
    print(f"Grader: {grader_model}")

    # Setup grader
    rubric = load_rubric()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    is_openai = grader_model.startswith("openai/")

    openai_client = None
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

    # Grade all entries concurrently
    tasks = [
        grade_single(entry, grader_model, rubric, openai_client, openrouter_key, semaphore)
        for entry in entries
    ]

    results = await asyncio.gather(*tasks)

    # Add answers to results for CSV
    for r, entry in zip(results, entries):
        r["answer"] = entry["answer"]

    # Save grades to run folder
    grades_file = run_dir / "grades.jsonl"
    with open(grades_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Append to master CSV
    csv_path = base_dir / "data" / "all_results.csv"
    append_to_csv(results, csv_path)

    # Summary
    success = sum(1 for r in results if r["scores"] is not None)
    errors = sum(1 for r in results if r["error"] is not None)
    avg_total = sum(r["scores"]["total"] for r in results if r["scores"]) / max(success, 1)

    print(f"\nDone! Grades saved to {grades_file}")
    print(f"Success: {success}, Errors: {errors}")
    print(f"Average total score: {avg_total:.1f}")


async def grade_human_baselines(grader_model: str):
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

    print(f"Grading {len(entries)} human baseline answers")
    print(f"Grader: {grader_model}")

    # Setup grader
    rubric = load_rubric()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    is_openai = grader_model.startswith("openai/")

    openai_client = None
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

    # Grade all entries concurrently
    tasks = [
        grade_single(entry, grader_model, rubric, openai_client, openrouter_key, semaphore)
        for entry in entries
    ]

    results = await asyncio.gather(*tasks)

    # Add answers to results for CSV
    for r, entry in zip(results, entries):
        r["answer"] = entry["answer"]

    # Save to human_grades folder
    grader_name = grader_model.replace("/", "_")
    human_grades_dir = base_dir / "data" / "human_grades"
    human_grades_dir.mkdir(parents=True, exist_ok=True)

    grades_file = human_grades_dir / f"{grader_name}.jsonl"
    with open(grades_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Append to master CSV
    csv_path = base_dir / "data" / "all_results.csv"
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

    args = parser.parse_args()

    if args.human:
        await grade_human_baselines(args.grader)
    elif args.run:
        await grade_run(args.run, args.grader)
    else:
        parser.error("Must specify either --run or --human")


if __name__ == "__main__":
    asyncio.run(main())
