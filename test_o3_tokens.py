#!/usr/bin/env python3
"""
Test script to figure out how to get reasoning tokens and response tokens from OpenAI o3.

Usage:
    python test_o3_tokens.py
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()


async def test_o3_tokens():
    """Test getting reasoning and response tokens from o3 model using responses API."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    
    client = AsyncOpenAI(api_key=api_key)
    
    # Simple test prompt
    test_prompt = "What is the capital of France? Explain your reasoning."
    
    print("Testing OpenAI responses API...")
    print(f"Prompt: {test_prompt}\n")
    
    # Use the responses API (same as generate_answers.py)
    response = await client.responses.create(
        model="o3-mini",
        input=[{"role": "user", "content": test_prompt}],
        temperature=1.0,
    )
    
    print("=" * 60)
    print("RESPONSE OBJECT:")
    print("=" * 60)
    print(json.dumps(response.model_dump(), indent=2, default=str))
    print()
    
    # Check for token info
    print("=" * 60)
    print("TOKEN INFORMATION:")
    print("=" * 60)
    if hasattr(response, 'usage') and response.usage:
        usage = response.usage
        print(f"Usage object: {usage}")
        print(f"\nUsage attributes:")
        for attr in dir(usage):
            if not attr.startswith('_'):
                val = getattr(usage, attr, None)
                if not callable(val):
                    print(f"  {attr}: {val}")
    else:
        print("No usage information in responses API")
    
    print("\n" + "=" * 60)
    print("RESPONSE CONTENT:")
    print("=" * 60)
    print(response.output_text)


if __name__ == "__main__":
    asyncio.run(test_o3_tokens())
