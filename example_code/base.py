"""Base provider interface for LLM API calls."""

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationResult:
    """Result from a single generation call."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    reasoning: str | None = None  # Chain-of-thought/reasoning tokens if available


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        max_retries: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        """Initialize the provider.

        Args:
            model: The model identifier (without provider prefix)
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)
        """
        assert model and isinstance(model, str), "model must be a non-empty string"
        assert max_retries > 0, "max_retries must be positive"
        assert base_delay > 0, "base_delay must be positive"
        assert max_delay >= base_delay, "max_delay must be >= base_delay"

        self.model = model
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    @abstractmethod
    async def _generate_impl(
        self, 
        prompt: str, 
        system: str | None = None, 
        temperature: float = 1.0,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        """Implementation of the generation call.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Sampling temperature
            reasoning_effort: Optional reasoning effort (OpenAI only: "low", "medium", "high")

        Returns:
            GenerationResult with the response

        Raises:
            Exception: On API errors (will be retried by generate())
        """
        pass

    async def generate(
        self, 
        prompt: str, 
        system: str | None = None, 
        temperature: float = 1.0,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        """Generate a response with exponential backoff retry.

        Args:
            prompt: The user prompt
            system: Optional system prompt
            temperature: Sampling temperature
            reasoning_effort: Optional reasoning effort (OpenAI only: "low", "medium", "high")

        Returns:
            GenerationResult with the response

        Raises:
            Exception: After max_retries attempts
        """
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await self._generate_impl(prompt, system, temperature, reasoning_effort)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter
                    delay = min(
                        self.base_delay * (2**attempt) + random.uniform(0, 1),
                        self.max_delay,
                    )
                    print(
                        f"  Retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: {e}"
                    )
                    await asyncio.sleep(delay)

        raise last_exception  # type: ignore

    @abstractmethod
    def supports_batching(self) -> bool:
        """Whether this provider supports batch processing."""
        pass

    async def generate_batch(
        self,
        requests: list[dict[str, Any]],
        poll_interval: float = 30.0,
    ) -> list[GenerationResult]:
        """Generate responses for a batch of requests.

        Only available for providers that support batching.

        Args:
            requests: List of request dicts with 'prompt', 'system', 'custom_id'
            poll_interval: How often to poll for batch completion (seconds)

        Returns:
            List of GenerationResults in the same order as requests

        Raises:
            NotImplementedError: If provider doesn't support batching
        """
        raise NotImplementedError("This provider does not support batching")

    @property
    def full_model_name(self) -> str:
        """Get the full model name including provider prefix."""
        return f"{self.provider_name}/{self.model}"

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name (e.g., 'openai', 'anthropic')."""
        pass
