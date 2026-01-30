#!/usr/bin/env python3
"""
Extract philosophical arguments from PDF papers using Claude 4.5 Opus with extended thinking.

Usage:
    conda activate evals
    python extract_arguments.py
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF
from anthropic import AsyncAnthropic
from dotenv import load_dotenv

load_dotenv()

# Max concurrent API requests
MAX_CONCURRENT = 15


def pdf_to_text(pdf_path: Path) -> str:
    """Convert a PDF file to plain text."""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text_parts.append(page.get_text())
    doc.close()
    return "\n".join(text_parts)


async def extract_argument(
    client: AsyncAnthropic,
    semaphore: asyncio.Semaphore,
    paper_text: str,
    prompt: str,
    paper_name: str,
    output_dir: Path,
) -> dict:
    """
    Send a philosophy paper to Claude 4.5 Opus with extended thinking enabled
    and extract a 600-word argument.
    """
    async with semaphore:
        # Format the document with the required tags
        user_message = f"<philosophy_research_paper>\n{paper_text}\n</philosophy_research_paper>\n\n{prompt}"

        print(f"Processing: {paper_name}...")

        try:
            response = await client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=16000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 2000,
                },
                messages=[
                    {
                        "role": "user",
                        "content": user_message,
                    }
                ],
            )

            # Extract the text response (skip thinking blocks)
            result_text = ""
            thinking_text = ""
            for block in response.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    result_text = block.text

            # Parse the JSON response
            json_text = result_text.strip()
            if json_text.startswith("```json"):
                json_text = json_text[7:]
            if json_text.startswith("```"):
                json_text = json_text[3:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            json_text = json_text.strip()

            result = json.loads(json_text)
            result["source_paper"] = paper_name
            result["thinking_summary"] = thinking_text
            result["usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        except json.JSONDecodeError as e:
            result = {
                "error": f"Failed to parse JSON: {e}",
                "raw_response": result_text,
                "source_paper": paper_name,
            }
        except Exception as e:
            result = {
                "error": str(e),
                "source_paper": paper_name,
            }

        # Save individual result
        paper_stem = Path(paper_name).stem
        output_filename = f"{paper_stem}.json"
        output_file = output_dir / output_filename
        result["output_filename"] = output_filename
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)

        status = "ERROR" if "error" in result else "OK"
        print(f"  Saved: {output_file.name} [{status}]")

        return result


async def main():
    # Setup paths
    base_dir = Path(__file__).parent
    papers_dir = base_dir / "papers"
    prompt_file = base_dir / "paper_summarizer_prompt.txt"

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_dir / "output" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the prompt
    with open(prompt_file, "r") as f:
        prompt = f.read().strip()

    # Get all PDF files
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in the papers directory.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")
    print(f"Output directory: {output_dir}")
    print(f"Max concurrent requests: {MAX_CONCURRENT}\n")

    # Initialize the async client and semaphore
    client = AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    # Convert PDFs to text and create tasks
    tasks = []
    for pdf_path in pdf_files:
        print(f"Converting {pdf_path.name} to text...")
        paper_text = pdf_to_text(pdf_path)
        tasks.append(
            extract_argument(client, semaphore, paper_text, prompt, pdf_path.name, output_dir)
        )

    # Run all extractions concurrently (limited by semaphore)
    print("\nSending papers to Claude 4.5 Opus with extended thinking enabled...")
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results for combined output
    output_data = []
    success_count = 0
    error_count = 0

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            output_data.append({
                "error": str(result),
                "source_paper": pdf_files[i].name,
            })
            error_count += 1
        else:
            output_data.append(result)
            if "error" in result:
                error_count += 1
            else:
                success_count += 1

    # Save combined results
    combined_file = output_dir / "all_arguments.json"
    with open(combined_file, "w") as f:
        json.dump(output_data, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Total papers: {len(pdf_files)}")
    print(f"Successful: {success_count}")
    print(f"Errors: {error_count}")
    print(f"\nCombined output: {combined_file}")


if __name__ == "__main__":
    asyncio.run(main())
