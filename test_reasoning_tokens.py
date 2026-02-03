#!/usr/bin/env python3
"""Test script to find reasoning tokens in OpenRouter API response."""

import asyncio
import json
import os

import httpx
from dotenv import load_dotenv

load_dotenv()


async def test_reasoning_tokens():
    """Test where reasoning tokens appear in the response."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set")

    model = "z-ai/glm-4.7"
    provider = "parasail/fp8"
    
    # Simple test prompt
    prompt = "What is the nature of consciousness? Provide a brief philosophical answer."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "provider": {"order": [provider]},
    }

    print(f"Testing model: {model}")
    print(f"Provider: {provider}")
    print(f"Prompt: {prompt}\n")
    print("=" * 80)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    print("\nFULL RESPONSE:")
    print(json.dumps(data, indent=2))
    
    print("\n" + "=" * 80)
    print("\nANALYZING STRUCTURE:")
    print(f"Top-level keys: {list(data.keys())}")
    
    if "choices" in data:
        print(f"\nNumber of choices: {len(data['choices'])}")
        choice = data["choices"][0]
        print(f"Choice keys: {list(choice.keys())}")
        
        if "message" in choice:
            message = choice["message"]
            print(f"Message keys: {list(message.keys())}")
            print(f"\nMessage content: {message.get('content', 'N/A')[:200]}...")
            
            # Check for reasoning in message
            if "reasoning" in message:
                print(f"\n✓ FOUND reasoning in message: {message['reasoning'][:200]}...")
            if "reasoning_content" in message:
                print(f"\n✓ FOUND reasoning_content in message: {message['reasoning_content'][:200]}...")
    
    if "usage" in data:
        print(f"\nUsage keys: {list(data['usage'].keys())}")
        usage = data["usage"]
        for key, value in usage.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(test_reasoning_tokens())
