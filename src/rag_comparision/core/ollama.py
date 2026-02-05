"""Ollama client for direct LLM interactions."""

import json
from collections.abc import Generator

from rag_comparision.config import (
    DEFAULT_MODEL_FALLBACK,
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
)

try:
    from langchain_core.messages import AIMessageChunk
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
    AIMessageChunk = None

try:
    import httpx

    HAS_HTTPX = True
except ImportError:
    httpx = None
    HAS_HTTPX = False

try:
    from urllib.error import URLError
    from urllib.request import urlopen

    HAS_URLLIB = True
except ImportError:
    urlopen = None
    URLError = None
    HAS_URLLIB = False


class OllamaResponse:
    """Response from Ollama API."""

    def __init__(
        self,
        content: str,
        model: str,
        base_url: str,
        reasoning_content: str | None = None,
        response_metadata: dict | None = None,
    ):
        """Initialize Ollama response.

        Args:
            content: Response content text
            model: Model name used
            base_url: Base URL of Ollama instance
            reasoning_content: Reasoning/thinking content if reasoning mode enabled
            response_metadata: Full response metadata from Ollama
        """
        self.content = content
        self.model = model
        self.base_url = base_url
        self.reasoning_content = reasoning_content
        self.response_metadata = response_metadata or {}

    def to_dict(self) -> dict:
        """Convert response to dictionary."""
        result = {
            'content': self.content,
            'model': self.model,
            'base_url': self.base_url,
        }
        if self.reasoning_content:
            result['reasoning_content'] = self.reasoning_content
        if self.response_metadata:
            result['response_metadata'] = self.response_metadata
        return result


class OllamaStreamChunk:
    """A single chunk from streaming Ollama response."""

    def __init__(
        self,
        content: str,
        reasoning_content: str | None = None,
        done: bool = False,
        response_metadata: dict | None = None,
    ):
        """Initialize stream chunk.

        Args:
            content: Content chunk text
            reasoning_content: Reasoning/thinking chunk if reasoning mode enabled
            done: Whether this is the final chunk
            response_metadata: Response metadata from Ollama
        """
        self.content = content
        self.reasoning_content = reasoning_content
        self.done = done
        self.response_metadata = response_metadata or {}

    def to_dict(self) -> dict:
        """Convert chunk to dictionary."""
        result = {
            'content': self.content,
            'done': self.done,
        }
        if self.reasoning_content:
            result['reasoning_content'] = self.reasoning_content
        if self.response_metadata:
            result['response_metadata'] = self.response_metadata
        return result


class OllamaClient:
    """Client for interacting with Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL_FALLBACK,
        base_url: str = OLLAMA_BASE_URL,
        reasoning: bool | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
    ):
        """Initialize Ollama client.

        Args:
            model: Ollama model name
            base_url: Ollama API base URL
            reasoning: Enable reasoning/thinking mode for supported models.
                True: Captures reasoning in additional_kwargs.reasoning_content
                False: Disables reasoning
                None: Uses model default behavior
            temperature: Sampling temperature (0.0 to 1.0)
            num_predict: Max number of tokens to generate
        """
        if ChatOllama is None:
            raise ImportError(
                'langchain-ollama is not installed. '
                'Install it with: pip install langchain-ollama'
            )
        self.model = model
        self.base_url = base_url
        self.reasoning = reasoning
        self.temperature = temperature
        self.num_predict = num_predict
        self._llm: ChatOllama | None = None

    def _get_llm(self) -> ChatOllama:
        """Get or create ChatOllama instance.

        Returns:
            ChatOllama instance
        """
        # Create instance if it doesn't exist
        # Note: reasoning can be passed per-call to invoke/stream, so we don't need
        # to recreate the instance when reasoning changes
        if self._llm is None:
            llm_kwargs = {
                'model': self.model,
                'base_url': self.base_url,
            }
            if self.reasoning is not None:
                llm_kwargs['reasoning'] = self.reasoning
            if self.temperature is not None:
                llm_kwargs['temperature'] = self.temperature
            if self.num_predict is not None:
                llm_kwargs['num_predict'] = self.num_predict

            self._llm = ChatOllama(**llm_kwargs)
        return self._llm

    def test_connection(self) -> dict:
        """Test connection to Ollama by listing available models.

        Returns:
            Dictionary with connection status and available models

        Raises:
            Exception: If connection fails
        """
        # Use Ollama's /api/tags endpoint to list models
        tags_url = f'{self.base_url}/api/tags'

        if not HAS_HTTPX and not HAS_URLLIB:
            raise ImportError(
                'Neither httpx nor urllib is available. '
                'Install httpx with: pip install httpx'
            )

        try:
            if HAS_HTTPX:
                # Use httpx if available (preferred)
                with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
                    response = client.get(tags_url)
                    response.raise_for_status()
                    data = response.json()
            else:
                # Fallback to urllib
                with urlopen(tags_url, timeout=int(OLLAMA_TIMEOUT)) as response:
                    data = json.loads(response.read().decode())

            # Extract model names
            models = []
            if 'models' in data:
                models = [model.get('name', '') for model in data['models']]

            return {
                'status': 'success',
                'base_url': self.base_url,
                'available_models': models,
                'model_count': len(models),
                'configured_model': self.model,
                'model_available': self.model in models if models else False,
            }
        except Exception as e:
            raise ConnectionError(
                f'Failed to connect to Ollama at {self.base_url}: {str(e)}'
            ) from e

    def chat(self, prompt: str, reasoning: bool | None = None) -> OllamaResponse:
        """Send a chat message to Ollama.

        Args:
            prompt: User prompt
            reasoning: Override reasoning setting for this call

        Returns:
            OllamaResponse with model response

        Raises:
            Exception: If request fails
        """
        llm = self._get_llm()
        invoke_kwargs = {}
        # Pass reasoning to invoke if provided (overrides instance default)
        if reasoning is not None:
            invoke_kwargs['reasoning'] = reasoning

        response = llm.invoke(prompt, **invoke_kwargs)

        # Extract reasoning content from additional_kwargs if available
        reasoning_content = None
        if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
            reasoning_content = response.additional_kwargs.get('reasoning_content')

        return OllamaResponse(
            content=response.content,
            model=self.model,
            base_url=self.base_url,
            reasoning_content=reasoning_content,
            response_metadata=getattr(response, 'response_metadata', {}),
        )

    def stream_chat(
        self, prompt: str, reasoning: bool | None = None
    ) -> Generator[OllamaStreamChunk, None, None]:
        """Stream a chat message from Ollama.

        Args:
            prompt: User prompt
            reasoning: Override reasoning setting for this call

        Yields:
            OllamaStreamChunk objects with incremental content

        Raises:
            Exception: If request fails
        """
        llm = self._get_llm()
        stream_kwargs = {}
        # Pass reasoning to stream if provided (overrides instance default)
        if reasoning is not None:
            stream_kwargs['reasoning'] = reasoning

        for chunk in llm.stream(prompt, **stream_kwargs):
            if AIMessageChunk is None or not isinstance(chunk, AIMessageChunk):
                # Fallback for non-AIMessageChunk responses
                chunk_content = getattr(chunk, 'content', str(chunk))
                if chunk_content:
                    yield OllamaStreamChunk(
                        content=chunk_content,
                        done=False,
                    )
                continue

            # Extract content
            chunk_content = chunk.content or ''

            # Extract reasoning content from additional_kwargs
            # Yield reasoning_content as-is - let the dashboard handle accumulation
            chunk_reasoning = None
            if hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
                chunk_reasoning = chunk.additional_kwargs.get('reasoning_content')

            # Check if this is the final chunk
            is_done = False
            response_metadata = {}
            if hasattr(chunk, 'response_metadata'):
                response_metadata = chunk.response_metadata or {}
                is_done = response_metadata.get('done', False)

            yield OllamaStreamChunk(
                content=chunk_content,
                reasoning_content=chunk_reasoning,
                done=is_done,
                response_metadata=response_metadata if is_done else {},
            )

    @staticmethod
    def is_available() -> bool:
        """Check if langchain-ollama is available."""
        return ChatOllama is not None
