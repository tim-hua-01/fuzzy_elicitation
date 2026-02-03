#!/usr/bin/env python3
"""
Generate philosophical answers using LLMs.

Usage:
    python generate_answers.py --model openai/gpt-4o --prompt answer_with_rubric --samples 5
    python generate_answers.py --model openrouter/meta-llama/llama-3-70b-instruct --prompt answer_without_rubric --samples 3

    # Generate and grade in one command
    python generate_answers.py --model openrouter/z-ai/glm-4.7 --prompt answer_with_rubric --samples 5 --grade
"""

import argparse
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

MAX_CONCURRENT = 25
MAX_RETRIES = 3
RETRY_DELAY_BASE = 2  # seconds


def get_rubric_version(rubric_name: str) -> str:
    """Extract version from rubric name (e.g., 'response_rubric_v2' -> 'v2')."""
    if "_v" in rubric_name:
        return "v" + rubric_name.split("_v")[-1]
    return "v1"  # default


def load_rubric_for_answering(rubric_name: str = "response_rubric_v2") -> str:
    """Load rubric and strip the Output format section."""
    rubric_path = Path(__file__).parent / f"{rubric_name}.md"
    if not rubric_path.exists():
        # Fallback to original
        rubric_path = Path(__file__).parent / "response_rubric.md"

    with open(rubric_path) as f:
        content = f.read()

    # Remove everything from "## Output format" onwards
    content = re.split(r"## Output [Ff]ormat", content)[0].strip()
    return content


def load_prompt_template(prompt_name: str) -> str:
    """Load a prompt template from the prompts folder."""
    prompt_path = Path(__file__).parent / "prompts" / f"{prompt_name}.txt"
    with open(prompt_path) as f:
        return f.read()


def format_prompt(template: str, question: str, rubric: str) -> str:
    """Format the prompt template with question and optionally rubric."""
    result = template.replace("{question}", question)
    if "{rubric}" in result:
        result = result.replace("{rubric}", rubric)
    return result


def load_questions() -> list[dict]:
    """Load questions from main_questions.jsonl."""
    questions_path = Path(__file__).parent / "main_questions.jsonl"
    questions = []
    with open(questions_path) as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


async def generate_openai_single(
    semaphore: asyncio.Semaphore,
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    temperature: float,
) -> dict[str, str | None]:
    """Generate a single response using OpenAI API with retries."""
    # Strip "openai/" prefix
    model_name = model.replace("openai/", "")

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                response = await client.responses.create(
                    model=model_name,
                    input=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                )
                return {
                    "content": response.output_text,
                    "reasoning": None,  # OpenAI API doesn't expose reasoning tokens
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE ** (attempt + 1)
                    print(f"OpenAI request failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise


async def generate_openrouter_single(
    semaphore: asyncio.Semaphore,
    http_client: httpx.AsyncClient,
    api_key: str,
    prompt: str,
    model: str,
    temperature: float,
    provider: str | None = None,
) -> dict[str, str | None]:
    """Generate a single response using OpenRouter API with retries."""
    # Strip "openrouter/" prefix
    model_name = model.replace("openrouter/", "")

    async with semaphore:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        if provider:
            payload["provider"] = {"order": [provider]}

        for attempt in range(MAX_RETRIES):
            try:
                response = await http_client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                response.raise_for_status()
                data = response.json()

                message = data["choices"][0]["message"]
                return {
                    "content": message["content"],
                    "reasoning": message.get("reasoning"),
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAY_BASE ** (attempt + 1)
                    print(f"OpenRouter request failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying in {delay}s: {e}")
                    await asyncio.sleep(delay)
                else:
                    raise


async def generate_single_sample(
    semaphore: asyncio.Semaphore,
    question: dict,
    sample_idx: int,
    template: str,
    rubric: str,
    model: str,
    prompt_name: str,
    temperature: float,
    is_openai: bool,
    client: AsyncOpenAI | None,
    http_client: httpx.AsyncClient | None,
    api_key: str | None,
    provider: str | None,
) -> dict:
    """Generate a single answer sample for a question."""
    question_id = Path(question["source_paper"]).stem
    question_text = question["question"]
    prompt = format_prompt(template, question_text, rubric)

    if is_openai:
        response_data = await generate_openai_single(
            semaphore, client, prompt, model, temperature
        )
    else:
        response_data = await generate_openrouter_single(
            semaphore, http_client, api_key, prompt, model, temperature, provider
        )

    entry = {
        "question_id": question_id,
        "question": question_text,
        "answer": response_data["content"],
        "model": model,
        "prompt_variant": prompt_name,
        "sample_idx": sample_idx,
        "is_human": False,
    }

    # Include reasoning if present
    if response_data.get("reasoning"):
        entry["reasoning"] = response_data["reasoning"]

    return entry


def parse_question_indices(indices_str: str, total: int) -> list[int]:
    """Parse question indices from string like '0,2,5' or '0-3' or '0,2-4,7'."""
    indices = set()
    for part in indices_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            indices.update(range(int(start), int(end) + 1))
        else:
            indices.add(int(part))
    # Filter to valid range and sort
    return sorted(i for i in indices if 0 <= i < total)


async def generate_answers(
    model: str,
    prompt_name: str,
    n_samples: int,
    temperature: float,
    run_name: str | None = None,
    provider: str | None = None,
    question_indices: str | None = None,
    rubric_name: str = "response_rubric_v2",
) -> Path:
    """Generate answers for all questions and save to a run folder."""
    # Setup
    base_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rubric_version = get_rubric_version(rubric_name)

    if run_name is None:
        model_short = model.split("/")[-1]
        run_name = f"{model_short}_{prompt_name}_{timestamp}"

    run_dir = base_dir / "data" / rubric_version / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    all_questions = load_questions()
    template = load_prompt_template(prompt_name)
    rubric = load_rubric_for_answering(rubric_name)

    # Filter questions if indices specified
    if question_indices:
        indices = parse_question_indices(question_indices, len(all_questions))
        questions = [all_questions[i] for i in indices]
        print(f"Selected {len(questions)}/{len(all_questions)} questions: {indices}")
    else:
        questions = all_questions

    # Save config
    config = {
        "model": model,
        "prompt_variant": prompt_name,
        "n_samples": n_samples,
        "temperature": temperature,
        "provider": provider,
        "question_indices": question_indices,
        "rubric": rubric_name,
        "timestamp": timestamp,
        "n_questions": len(questions),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Nice config printout
    total_samples = len(questions) * n_samples
    print("\n" + "=" * 50)
    print("GENERATE ANSWERS")
    print("=" * 50)
    print(f"  Model:       {model}")
    print(f"  Provider:    {provider or '(default)'}")
    print(f"  Prompt:      {prompt_name}")
    print(f"  Rubric:      {rubric_name}")
    print(f"  Temperature: {temperature}")
    print(f"  Questions:   {len(questions)}")
    print(f"  Samples:     {n_samples} per question")
    print(f"  Total calls: {total_samples}")
    print(f"  Output:      {run_dir}")
    print("=" * 50 + "\n")

    # Determine provider
    is_openai = model.startswith("openai/")

    client = None
    http_client = None
    api_key = None

    if is_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = AsyncOpenAI(api_key=api_key)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")
        http_client = httpx.AsyncClient(timeout=240.0)

    # Generate answers - one task per (question, sample) pair
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    try:
        tasks = [
            generate_single_sample(
                semaphore, q, sample_idx, template, rubric, model, prompt_name,
                temperature, is_openai, client, http_client, api_key, provider
            )
            for q in questions
            for sample_idx in range(n_samples)
        ]

        results = await tqdm_asyncio.gather(*tasks, desc="Generating", unit="sample", return_exceptions=True)
    finally:
        # Close http client if created
        if http_client:
            await http_client.aclose()

    # Separate successful results from failures
    successful = []
    failures = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            q_idx = i // n_samples
            sample_idx = i % n_samples
            question_id = Path(questions[q_idx]["source_paper"]).stem
            failures.append((question_id, sample_idx, result))
        else:
            successful.append(result)

    # Report failures
    if failures:
        print(f"\n⚠️  WARNING: {len(failures)} sample(s) failed to generate:")
        for question_id, sample_idx, error in failures:
            print(f"  - {question_id} sample {sample_idx}: {error}")
        print()

    if not successful:
        raise RuntimeError(f"All {len(results)} samples failed to generate. Check errors above.")

    # Sort results by question_id and sample_idx for consistent ordering
    successful.sort(key=lambda x: (x["question_id"], x["sample_idx"]))

    # Write to file
    answers_file = run_dir / "answers.jsonl"
    with open(answers_file, "w") as f:
        for entry in successful:
            f.write(json.dumps(entry) + "\n")

    print(f"\nDone! {len(successful)}/{len(results)} answers saved to {answers_file}")
    return run_dir, run_name


async def main():
    parser = argparse.ArgumentParser(description="Generate philosophical answers using LLMs")
    parser.add_argument("--model", required=True, help="Model (e.g., openai/gpt-4o, openrouter/meta-llama/llama-3-70b-instruct)")
    parser.add_argument("--prompt", required=True, help="Prompt template name (e.g., answer_with_rubric)")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples per question")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--run-name", help="Custom run name (default: auto-generated)")
    parser.add_argument("--provider", help="OpenRouter provider (e.g., together, openai)")
    parser.add_argument("--questions", help="Question indices to generate (e.g., '0,2,5' or '0-3' or '0,2-4,7')")
    parser.add_argument("--rubric", default="response_rubric_v2", help="Rubric file name without .md (default: response_rubric_v2)")
    parser.add_argument("--grade", action="store_true", help="Automatically grade answers after generation")
    parser.add_argument("--grader", help="Grader model (default: same as --model)")

    args = parser.parse_args()

    run_dir, run_name = await generate_answers(
        model=args.model,
        prompt_name=args.prompt,
        n_samples=args.samples,
        temperature=args.temperature,
        run_name=args.run_name,
        provider=args.provider,
        question_indices=args.questions,
        rubric_name=args.rubric,
    )

    # Auto-grade if requested
    if args.grade:
        from grade_responses import grade_run
        grader_model = args.grader or args.model
        print()  # blank line
        await grade_run(
            run_name=run_name,
            grader_model=grader_model,
            provider=args.provider,
            rubric_name=args.rubric,
        )


if __name__ == "__main__":
    asyncio.run(main())
