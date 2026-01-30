#!/usr/bin/env python3
"""
Generate philosophical answers using LLMs.

Usage:
    python generate_answers.py --model openai/gpt-4o --prompt answer_with_rubric --samples 5
    python generate_answers.py --model openrouter/meta-llama/llama-3-70b-instruct --prompt answer_without_rubric --samples 3
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

load_dotenv()

MAX_CONCURRENT = 10


def load_rubric_for_answering() -> str:
    """Load rubric and strip the Output format section."""
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


async def generate_openai(
    client: AsyncOpenAI,
    prompt: str,
    model: str,
    n_samples: int,
    temperature: float,
) -> list[str]:
    """Generate responses using OpenAI API with native n parameter."""
    # Strip "openai/" prefix
    model_name = model.replace("openai/", "")

    response = await client.responses.create(
        model=model_name,
        input=[{"role": "user", "content": prompt}],
        n=n_samples,
        temperature=temperature,
    )

    # Extract all outputs
    outputs = []
    if hasattr(response, 'output') and response.output:
        for item in response.output:
            if hasattr(item, 'content'):
                for content_block in item.content:
                    if hasattr(content_block, 'text'):
                        outputs.append(content_block.text)

    # Fallback to output_text for single response
    if not outputs and hasattr(response, 'output_text'):
        outputs = [response.output_text]

    return outputs


async def generate_openrouter_single(
    semaphore: asyncio.Semaphore,
    api_key: str,
    prompt: str,
    model: str,
    temperature: float,
) -> str:
    """Generate a single response using OpenRouter API."""
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

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]


async def generate_openrouter(
    api_key: str,
    prompt: str,
    model: str,
    n_samples: int,
    temperature: float,
) -> list[str]:
    """Generate multiple responses using OpenRouter API via async gather."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    tasks = [
        generate_openrouter_single(semaphore, api_key, prompt, model, temperature)
        for _ in range(n_samples)
    ]

    return await asyncio.gather(*tasks)


async def generate_answers(
    model: str,
    prompt_name: str,
    n_samples: int,
    temperature: float,
    run_name: str | None = None,
) -> Path:
    """Generate answers for all questions and save to a run folder."""
    # Setup
    base_dir = Path(__file__).parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_name is None:
        model_short = model.split("/")[-1]
        run_name = f"{model_short}_{prompt_name}_{timestamp}"

    run_dir = base_dir / "data" / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    questions = load_questions()
    template = load_prompt_template(prompt_name)
    rubric = load_rubric_for_answering()

    # Save config
    config = {
        "model": model,
        "prompt_variant": prompt_name,
        "n_samples": n_samples,
        "temperature": temperature,
        "timestamp": timestamp,
        "n_questions": len(questions),
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Generating answers for {len(questions)} questions x {n_samples} samples")
    print(f"Model: {model}")
    print(f"Prompt: {prompt_name}")
    print(f"Output: {run_dir}")

    # Determine provider
    is_openai = model.startswith("openai/")

    if is_openai:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        client = AsyncOpenAI(api_key=api_key)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set")

    # Generate answers
    answers_file = run_dir / "answers.jsonl"

    with open(answers_file, "w") as f:
        for q_idx, q in enumerate(questions):
            question_id = Path(q["source_paper"]).stem
            question_text = q["question"]

            prompt = format_prompt(template, question_text, rubric)

            print(f"[{q_idx + 1}/{len(questions)}] {question_id}...")

            if is_openai:
                responses = await generate_openai(
                    client, prompt, model, n_samples, temperature
                )
            else:
                responses = await generate_openrouter(
                    api_key, prompt, model, n_samples, temperature
                )

            for sample_idx, answer in enumerate(responses):
                entry = {
                    "question_id": question_id,
                    "question": question_text,
                    "answer": answer,
                    "model": model,
                    "prompt_variant": prompt_name,
                    "sample_idx": sample_idx,
                    "is_human": False,
                }
                f.write(json.dumps(entry) + "\n")

            print(f"  Generated {len(responses)} samples")

    print(f"\nDone! Answers saved to {answers_file}")
    return run_dir


async def main():
    parser = argparse.ArgumentParser(description="Generate philosophical answers using LLMs")
    parser.add_argument("--model", required=True, help="Model (e.g., openai/gpt-4o, openrouter/meta-llama/llama-3-70b-instruct)")
    parser.add_argument("--prompt", required=True, help="Prompt template name (e.g., answer_with_rubric)")
    parser.add_argument("--samples", type=int, default=1, help="Number of samples per question")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--run-name", help="Custom run name (default: auto-generated)")

    args = parser.parse_args()

    await generate_answers(
        model=args.model,
        prompt_name=args.prompt,
        n_samples=args.samples,
        temperature=args.temperature,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    asyncio.run(main())
