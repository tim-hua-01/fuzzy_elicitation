"""OpenAI provider with async and batch support."""

import asyncio
import json
import os
import tempfile
import time
from typing import Any

from openai import AsyncOpenAI

from .base import BaseProvider, GenerationResult


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI API with batching support."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """Initialize OpenAI provider.

        Args:
            model: Model name (e.g., "gpt-4o", "gpt-4o-mini")
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
        """
        super().__init__(model, max_retries, base_delay, max_delay)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
        self.client = AsyncOpenAI(api_key=self.api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def _generate_impl(
        self, 
        prompt: str, 
        system: str | None = None, 
        temperature: float = 1.0,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        """Generate a response using OpenAI API.
        
        Uses the responses API (client.responses.create) which works for both
        GPT-5 and GPT-4o models.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Build API call parameters for responses API
        api_params = {
            "model": self.model,
            "input": messages,
        }
        
        # GPT-5 models don't support temperature parameter
        if "gpt-5" not in self.model.lower():
            api_params["temperature"] = temperature
        
        # Add reasoning effort if specified (for GPT-5 models)
        if reasoning_effort is not None:
            api_params["reasoning"] = {"effort": reasoning_effort}

        response = await self.client.responses.create(**api_params)

        # Extract content from response using output_text
        if not hasattr(response, 'output_text') or not response.output_text:
            raise RuntimeError("No content in output from OpenAI")
        
        content = response.output_text
        if content == "":
            raise RuntimeError("Empty content returned from OpenAI")
        
        usage = None
        if hasattr(response, 'usage') and response.usage:
            usage = {
                "prompt_tokens": getattr(response.usage, 'input_tokens', 0),
                "completion_tokens": getattr(response.usage, 'output_tokens', 0),
                "total_tokens": getattr(response.usage, 'total_tokens', 0),
            }

        return GenerationResult(content=content, model=self.model, usage=usage)

    def supports_batching(self) -> bool:
        return True

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
        poll_interval: float = 30.0,
    ) -> list[GenerationResult]:
        """Generate responses using OpenAI Batch API.

        Args:
            requests: List of dicts with 'prompt', 'system' (optional), 'custom_id', 'temperature' (optional), 'reasoning_effort' (optional)
            poll_interval: How often to poll for completion (seconds)

        Returns:
            List of GenerationResults in the same order as requests
        """
        # Create JSONL content
        jsonl_lines = []
        for req in requests:
            messages = []
            if req.get("system"):
                messages.append({"role": "system", "content": req["system"]})
            messages.append({"role": "user", "content": req["prompt"]})

            body = {
                "model": self.model,
                "messages": messages,
            }
            
            # GPT-5 models don't support temperature parameter
            if "gpt-5" not in self.model.lower():
                body["temperature"] = req.get("temperature", 1.0)
            
            # Add reasoning effort if specified
            if req.get("reasoning_effort"):
                body["reasoning"] = {"effort": req["reasoning_effort"]}

            line = {
                "custom_id": req["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
            jsonl_lines.append(json.dumps(line))

        jsonl_content = "\n".join(jsonl_lines)

        # Upload file - need sync client for file upload
        from openai import OpenAI

        sync_client = OpenAI(api_key=self.api_key)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            tmp.write(jsonl_content)
            tmp_path = tmp.name

        try:
            with open(tmp_path, "rb") as f:
                batch_file = sync_client.files.create(file=f, purpose="batch")

            # Create batch job
            batch_job = sync_client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            print(f"  Batch job created: {batch_job.id}")

            # Poll until complete
            while True:
                batch_job = sync_client.batches.retrieve(batch_job.id)
                status = batch_job.status

                if status == "completed":
                    print(f"  Batch completed: {batch_job.request_counts}")
                    break
                elif status in ("failed", "expired", "cancelled"):
                    raise RuntimeError(f"Batch job {status}: {batch_job.errors}")
                else:
                    print(f"  Batch status: {status}, waiting {poll_interval}s...")
                    await asyncio.sleep(poll_interval)

            # Download results
            if not batch_job.output_file_id:
                raise RuntimeError("Batch completed but no output file")

            result_content = sync_client.files.content(batch_job.output_file_id)
            result_text = result_content.text

            # Parse results
            results_by_id: dict[str, GenerationResult] = {}
            for line in result_text.strip().split("\n"):
                if not line:
                    continue
                result = json.loads(line)
                custom_id = result["custom_id"]
                response_body = result.get("response", {}).get("body", {})

                if result.get("error"):
                    # Handle error responses
                    results_by_id[custom_id] = GenerationResult(
                        content=f"ERROR: {result['error']}",
                        model=self.model,
                    )
                else:
                    choices = response_body.get("choices", [])
                    content = (
                        choices[0]["message"]["content"] if choices else "NO CONTENT"
                    )
                    usage_data = response_body.get("usage")
                    usage = None
                    if usage_data:
                        usage = {
                            "prompt_tokens": usage_data.get("prompt_tokens", 0),
                            "completion_tokens": usage_data.get("completion_tokens", 0),
                            "total_tokens": usage_data.get("total_tokens", 0),
                        }
                    results_by_id[custom_id] = GenerationResult(
                        content=content, model=self.model, usage=usage
                    )

            # Return results in original order
            return [results_by_id[req["custom_id"]] for req in requests]

        finally:
            import os as os_module

            os_module.unlink(tmp_path)
