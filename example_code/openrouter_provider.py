"""OpenRouter provider for various models."""

import os

import httpx

from .base import BaseProvider, GenerationResult


class OpenRouterProvider(BaseProvider):
    """Provider for OpenRouter API (access to many models)."""

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """Initialize OpenRouter provider.

        Args:
            model: Model name (e.g., "meta-llama/llama-3-70b-instruct")
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
        """
        super().__init__(model, max_retries, base_delay, max_delay)
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key not provided and OPENROUTER_API_KEY not set"
            )

    @property
    def provider_name(self) -> str:
        return "openrouter"

    async def _generate_impl(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 1.0,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        """Generate a response using OpenRouter API.

        Args:
            prompt: User message
            system: Optional system prompt
            temperature: Sampling temperature
            reasoning_effort: Reasoning effort level ("high", "medium", "low", "minimal", "none")
                            or None to use model defaults
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/deliberative-alignment",
            "X-Title": "Deliberative Alignment Data Generation",
        }

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        # Add reasoning config if specified
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                self.BASE_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

        message_data = data["choices"][0]["message"]
        content = message_data.get("content", "")

        # Extract reasoning tokens if available
        reasoning = message_data.get("reasoning") or message_data.get("reasoning_content")

        usage = None
        if "usage" in data:
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }
            # Include reasoning token count if available
            if "completion_tokens_details" in data["usage"]:
                details = data["usage"]["completion_tokens_details"]
                if "reasoning_tokens" in details:
                    usage["reasoning_tokens"] = details["reasoning_tokens"]

        return GenerationResult(
            content=content,
            model=self.model,
            usage=usage,
            reasoning=reasoning,
        )

    def supports_batching(self) -> bool:
        return False
