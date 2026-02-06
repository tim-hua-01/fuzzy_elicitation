#!/usr/bin/env python3
"""
Pairwise comparison of philosophical answers using Claude 4.5 Sonnet.

Loads graded answers from v2 runs, finds low-scoring (<=30) and high-scoring (>=33) pairs
for the same question, and asks Claude to judge which response is better argued — without
any rubric. Each pair is evaluated twice with flipped order to check consistency.

This serves as a rubric-less validation of the rubric-based grading scores.

Usage:
    python pairwise_compare.py
    python pairwise_compare.py --n-pairs 30 --low-threshold 28 --high-threshold 35
    python pairwise_compare.py --grades-suffix anthropic_claude-sonnet-4-5
"""

import argparse
import asyncio
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

MAX_CONCURRENT = 25

COMPARISON_PROMPT_TEMPLATE = """You are an expert philosophy evaluator. You will be shown a philosophical question and two responses to that question. Your task is to judge which response is the better holistically.

## Question

{question}

## Response A

{response_a}

## Response B

{response_b}

## Your Judgment

Which response is better argued overall? Respond with ONLY a JSON object (no other text):
{{"winner": "A" or "B", "confidence": "high" or "medium" or "low", "reason": "100-300 word explanation"}}"""


def load_all_entries(runs_dir: Path, grades_suffix: str | None = None) -> list[dict]:
    """Load all graded entries from all runs, joining answers with grades.

    Each entry has: question_id, question, answer, model, prompt_variant, sample_idx, total, scores.
    """
    grades_filename = f"grades_{grades_suffix}.jsonl" if grades_suffix else "grades.jsonl"
    entries: list[dict] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue

        answers_file = run_dir / "answers.jsonl"
        grades_file = run_dir / grades_filename

        if not answers_file.exists() or not grades_file.exists():
            continue

        # Load answers (has question text)
        answers: dict[tuple, dict] = {}
        with open(answers_file) as f:
            for line in f:
                if line.strip():
                    a = json.loads(line)
                    key = (a["question_id"], a["prompt_variant"], a["sample_idx"])
                    answers[key] = a

        # Load grades (has scores)
        with open(grades_file) as f:
            for line in f:
                if line.strip():
                    g = json.loads(line)
                    if g.get("scores") is None or g.get("error"):
                        continue
                    key = (g["question_id"], g["prompt_variant"], g["sample_idx"])
                    if key not in answers:
                        continue
                    entry = {
                        "question_id": g["question_id"],
                        "question": answers[key]["question"],
                        "answer": answers[key]["answer"],
                        "model": g["model"],
                        "prompt_variant": g["prompt_variant"],
                        "sample_idx": g["sample_idx"],
                        "total": g["scores"]["total"],
                        "scores": g["scores"],
                    }
                    entries.append(entry)

    return entries


def find_pairs(
    entries: list[dict],
    low_threshold: int,
    high_threshold: int,
    n_pairs: int,
    seed: int,
    stratified: bool = True,
    cross_variant_only: bool = False,
) -> tuple[list[tuple[dict, dict]], list[float]]:
    """Find pairs of (low, high) scoring entries for the same question.

    Returns (pairs, weights) where weights are inverse-probability-of-sampling weights
    for recovering population-level estimates from the stratified sample.
    Weight = N_stratum / n_stratum (how many population pairs each sampled pair represents).
    For simple random sampling, all weights are N_total / n_total.

    If stratified=True, samples evenly across (low_variant, high_variant) strata
    to ensure coverage of all cross-prompt combinations. Otherwise, samples uniformly.

    If cross_variant_only=True, excludes pairs where both entries share the same prompt_variant.
    """
    # Group by question_id
    by_question: dict[str, dict[str, list[dict]]] = {}
    for e in entries:
        qid = e["question_id"]
        if qid not in by_question:
            by_question[qid] = {"low": [], "high": []}
        if e["total"] <= low_threshold:
            by_question[qid]["low"].append(e)
        elif e["total"] >= high_threshold:
            by_question[qid]["high"].append(e)

    # Collect all possible pairs
    all_pairs: list[tuple[dict, dict]] = []
    for qid, groups in by_question.items():
        for low in groups["low"]:
            for high in groups["high"]:
                if cross_variant_only and low["prompt_variant"] == high["prompt_variant"]:
                    continue
                all_pairs.append((low, high))

    if not all_pairs:
        raise ValueError(
            f"No valid pairs found with low_threshold={low_threshold} and high_threshold={high_threshold}. "
            f"Available score range: {min(e['total'] for e in entries)}-{max(e['total'] for e in entries)}"
        )

    rng = random.Random(seed)

    if not stratified:
        n_to_sample = min(n_pairs, len(all_pairs))
        selected = rng.sample(all_pairs, n_to_sample)
        # Uniform weight: each sampled pair represents N/n population pairs
        w = len(all_pairs) / n_to_sample
        weights = [w] * n_to_sample
        return selected, weights

    # Stratified sampling: group by (low_variant, high_variant)
    strata: dict[tuple[str, str], list[tuple[dict, dict]]] = {}
    for pair in all_pairs:
        key = (pair[0]["prompt_variant"], pair[1]["prompt_variant"])
        strata.setdefault(key, []).append(pair)

    # Print strata info
    print(f"\n  Strata ({len(strata)} variant pairs):")
    for key in sorted(strata.keys()):
        print(f"    {key[0]} -> {key[1]}: {len(strata[key])} possible pairs")

    # Round-robin allocation: distribute n_pairs evenly, then redistribute leftovers
    stratum_keys = sorted(strata.keys())
    base_per_stratum = n_pairs // len(stratum_keys)
    remainder = n_pairs % len(stratum_keys)

    # Initial allocation (give +1 to the first `remainder` strata)
    allocation: dict[tuple[str, str], int] = {}
    for i, key in enumerate(stratum_keys):
        allocation[key] = base_per_stratum + (1 if i < remainder else 0)

    # Cap at available pairs, redistribute excess
    redistributed = True
    while redistributed:
        redistributed = False
        excess = 0
        for key in stratum_keys:
            available = len(strata[key])
            if allocation[key] > available:
                excess += allocation[key] - available
                allocation[key] = available
                redistributed = True

        if excess > 0:
            # Distribute excess to strata that still have capacity
            for key in stratum_keys:
                if excess <= 0:
                    break
                capacity = len(strata[key]) - allocation[key]
                give = min(excess, capacity)
                allocation[key] += give
                excess -= give

    # Sample from each stratum, compute weights
    selected: list[tuple[dict, dict]] = []
    weights: list[float] = []
    for key in stratum_keys:
        n_from_stratum = allocation[key]
        if n_from_stratum > 0:
            N_stratum = len(strata[key])
            w = N_stratum / n_from_stratum  # each sampled pair represents this many population pairs
            sampled = rng.sample(strata[key], n_from_stratum)
            selected.extend(sampled)
            weights.extend([w] * n_from_stratum)

    # Shuffle pairs and weights together
    combined = list(zip(selected, weights))
    rng.shuffle(combined)
    selected = [c[0] for c in combined]
    weights = [c[1] for c in combined]

    return selected, weights


MAX_RETRIES = 3


async def compare_pair(
    semaphore: asyncio.Semaphore,
    client: AsyncAnthropic,
    question: str,
    response_a: str,
    response_b: str,
    reasoning_budget: int,
) -> dict | None:
    """Ask Claude to compare two responses. Returns parsed JSON judgment, or None on failure."""
    prompt = COMPARISON_PROMPT_TEMPLATE.format(
        question=question,
        response_a=response_a,
        response_b=response_b,
    )

    create_params: dict = {
        "model": "claude-sonnet-4-5",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max(4096, reasoning_budget + 2000),
        "temperature": 1,
    }

    if reasoning_budget > 0:
        create_params["thinking"] = {
            "type": "enabled",
            "budget_tokens": reasoning_budget,
        }

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.messages.create(**create_params)

            content_text = ""
            thinking_text: str | None = None
            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "thinking":
                    thinking_text = block.thinking

            # Parse JSON response
            text = content_text.strip()
            try:
                result = json.loads(text, strict=False)
            except json.JSONDecodeError:
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                result = json.loads(text.strip(), strict=False)

            result["reasoning"] = thinking_text
            return result

        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"  [retry {attempt+1}/{MAX_RETRIES}] compare_pair failed: {e}")
            else:
                print(f"  [FAILED after {MAX_RETRIES} attempts] compare_pair: {e}")
                return None


async def run_comparisons(
    pairs: list[tuple[dict, dict]],
    weights: list[float],
    reasoning_budget: int,
    output_path: Path,
) -> None:
    """Run all pairwise comparisons (both orderings) and save results.

    Each pair produces one CSV row with results from both orderings.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Build all tasks: for each pair, run comparison in both orders
    # Tasks are interleaved: [pair0_low_first, pair0_high_first, pair1_low_first, ...]
    tasks: list = []
    for low, high in pairs:
        # Order 1: low=A, high=B
        tasks.append(
            compare_pair(
                semaphore, client, low["question"],
                low["answer"], high["answer"], reasoning_budget,
            )
        )
        # Order 2: high=A, low=B (flipped)
        tasks.append(
            compare_pair(
                semaphore, client, high["question"],
                high["answer"], low["answer"], reasoning_budget,
            )
        )

    print(f"\nRunning {len(tasks)} comparisons ({len(pairs)} pairs x 2 orderings)...")
    results = await tqdm_asyncio.gather(*tasks, desc="Comparing", unit="cmp")

    await client.close()

    # Process results into CSV rows — one row per pair
    rows: list[dict] = []
    n_skipped = 0
    for pair_idx, ((low, high), weight) in enumerate(zip(pairs, weights)):
        result_low_first = results[pair_idx * 2]      # A=low, B=high
        result_high_first = results[pair_idx * 2 + 1]  # A=high, B=low

        # Skip pairs where either comparison failed
        if result_low_first is None or result_high_first is None:
            n_skipped += 1
            continue

        # Order 1: A=low, B=high -> winner "B" means high won
        picked_high_lowfirst = result_low_first["winner"] == "B"
        # Order 2: A=high, B=low -> winner "A" means high won
        picked_high_highfirst = result_high_first["winner"] == "A"

        # Consensus: True if both say high, False if both say low, "disagree" if mixed
        if picked_high_lowfirst and picked_high_highfirst:
            model_picked_high = "True"
        elif not picked_high_lowfirst and not picked_high_highfirst:
            model_picked_high = "False"
        else:
            model_picked_high = "disagree"

        rows.append({
            "pair_idx": pair_idx,
            "question_id": low["question_id"],
            "low_prompt_variant": low["prompt_variant"],
            "low_sample_idx": low["sample_idx"],
            "low_total": low["total"],
            "high_prompt_variant": high["prompt_variant"],
            "high_sample_idx": high["sample_idx"],
            "high_total": high["total"],
            "score_diff": high["total"] - low["total"],
            "weight": weight,
            "picked_high_lowfirst": picked_high_lowfirst,
            "picked_high_highfirst": picked_high_highfirst,
            "model_picked_high": model_picked_high,
            "confidence_lowfirst": result_low_first.get("confidence", ""),
            "confidence_highfirst": result_high_first.get("confidence", ""),
            "reason_lowfirst": result_low_first.get("reason", ""),
            "reason_highfirst": result_high_first.get("reason", ""),
            "reasoning_lowfirst": result_low_first.get("reasoning", ""),
            "reasoning_highfirst": result_high_first.get("reasoning", ""),
            "timestamp": datetime.now().isoformat(),
        })

    if n_skipped:
        print(f"\nWarning: {n_skipped}/{len(pairs)} pairs skipped due to failed API calls.")

    # Write CSV
    fieldnames = [
        "pair_idx", "question_id",
        "low_prompt_variant", "low_sample_idx", "low_total",
        "high_prompt_variant", "high_sample_idx", "high_total",
        "score_diff", "weight",
        "picked_high_lowfirst", "picked_high_highfirst", "model_picked_high",
        "confidence_lowfirst", "confidence_highfirst",
        "reason_lowfirst", "reason_highfirst",
        "reasoning_lowfirst", "reasoning_highfirst",
        "timestamp",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Summary
    total_pairs = len(rows)
    agreed = sum(1 for r in rows if r["model_picked_high"] != "disagree")
    high_won_both = sum(1 for r in rows if r["model_picked_high"] == "True")
    low_won_both = sum(1 for r in rows if r["model_picked_high"] == "False")
    disagreed = sum(1 for r in rows if r["model_picked_high"] == "disagree")

    print(f"\n{'=' * 50}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"  Total pairs:                      {total_pairs}")
    print(f"  Agreement (same both orderings):   {agreed}/{total_pairs} ({100*agreed/total_pairs:.0f}%)")
    print(f"  High-score won both orderings:     {high_won_both}/{total_pairs} ({100*high_won_both/total_pairs:.0f}%)")
    print(f"  Low-score won both orderings:      {low_won_both}/{total_pairs} ({100*low_won_both/total_pairs:.0f}%)")
    print(f"  Disagreed across orderings:        {disagreed}/{total_pairs} ({100*disagreed/total_pairs:.0f}%)")
    print(f"{'=' * 50}")
    print(f"\nResults saved to {output_path}")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pairwise comparison of philosophical answers using Claude 4.5 Sonnet"
    )
    parser.add_argument(
        "--low-threshold", type=int, default=30,
        help="Maximum total score for 'low' bucket (default: 30)",
    )
    parser.add_argument(
        "--high-threshold", type=int, default=33,
        help="Minimum total score for 'high' bucket (default: 33)",
    )
    parser.add_argument(
        "--n-pairs", type=int, default=20,
        help="Number of pairs to sample (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--reasoning-budget", type=int, default=1500,
        help="Anthropic extended thinking budget in tokens (default: 1500)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: data/v2/pairwise_comparisons.csv)",
    )
    parser.add_argument(
        "--csv-name", type=str, default=None,
        help="Name of the output CSV file (without path, e.g. 'my_comparisons.csv'). Overrides default filename but keeps directory structure.",
    )
    parser.add_argument(
        "--grades-suffix", type=str, default=None,
        help="Grades file suffix, e.g. 'anthropic_claude-sonnet-4-5' to use grades_anthropic_claude-sonnet-4-5.jsonl (default: grades.jsonl)",
    )
    parser.add_argument(
        "--no-stratify", action="store_true",
        help="Disable stratified sampling (use simple random sampling instead)",
    )
    parser.add_argument(
        "--cross-variant-only", action="store_true",
        help="Only sample pairs where low and high have different prompt variants",
    )

    args = parser.parse_args()

    base_dir = Path(__file__).parent
    runs_dir = base_dir / "data" / "v2" / "runs"

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    # Load all entries
    print("Loading entries from all runs...")
    entries = load_all_entries(runs_dir, args.grades_suffix)
    if not entries:
        raise ValueError(
            f"No entries loaded. Check that grades files exist in {runs_dir}. "
            f"grades_suffix={args.grades_suffix}"
        )
    print(f"Loaded {len(entries)} graded entries")

    # Score distribution
    totals = [e["total"] for e in entries]
    print(f"Score range: {min(totals)}-{max(totals)}, mean: {sum(totals)/len(totals):.1f}")

    low_count = sum(1 for t in totals if t <= args.low_threshold)
    high_count = sum(1 for t in totals if t >= args.high_threshold)
    print(f"Low (<={args.low_threshold}): {low_count}, High (>={args.high_threshold}): {high_count}")

    # Find pairs
    stratified = not args.no_stratify
    pairs, weights = find_pairs(
        entries, args.low_threshold, args.high_threshold, args.n_pairs, args.seed,
        stratified=stratified, cross_variant_only=args.cross_variant_only,
    )
    print(f"Sampled {len(pairs)} pairs")

    # Print config
    print(f"\n{'=' * 50}")
    print("PAIRWISE COMPARISON")
    print(f"{'=' * 50}")
    print(f"  Low threshold:    <={args.low_threshold}")
    print(f"  High threshold:   >={args.high_threshold}")
    print(f"  Pairs:            {len(pairs)}")
    print(f"  Sampling:         {'stratified by variant pair' if stratified else 'simple random'}")
    print(f"  Cross-variant:    {'only' if args.cross_variant_only else 'all'}")
    print(f"  Reasoning budget: {args.reasoning_budget}")
    print(f"  Seed:             {args.seed}")
    print(f"  Grader:           Claude 4.5 Sonnet")
    print(f"{'=' * 50}")

    # Run comparisons
    if args.output:
        output_path = Path(args.output)
    elif args.csv_name:
        output_path = base_dir / "data" / "v2" / args.csv_name
    else:
        output_path = base_dir / "data" / "v2" / "pairwise_comparisons.csv"
    await run_comparisons(pairs, weights, args.reasoning_budget, output_path)


if __name__ == "__main__":
    asyncio.run(main())
