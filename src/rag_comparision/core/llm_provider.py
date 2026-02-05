"""LLM provider abstraction for switchable LLM backends (Gemini/Ollama)."""

from collections.abc import Generator
from typing import Any

from rag_comparision.config import (
    DEFAULT_MODEL_FALLBACK,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
)

# Try to import Gemini
try:
    from langchain_google_genai import ChatGoogleGenerativeAI

    HAS_GEMINI = True
except ImportError:
    try:
        # Fallback: try older import path
        from langchain_google_genai import GoogleGenerativeAI as ChatGoogleGenerativeAI

        HAS_GEMINI = True
    except ImportError:
        ChatGoogleGenerativeAI = None  # type: ignore
        HAS_GEMINI = False

# Try to import Ollama
try:
    from langchain_ollama import ChatOllama

    HAS_OLLAMA = True
except ImportError:
    ChatOllama = None  # type: ignore
    HAS_OLLAMA = False

try:
    from langchain_core.messages import AIMessageChunk

    HAS_AIMESSAGE_CHUNK = True
except ImportError:
    AIMessageChunk = None  # type: ignore
    HAS_AIMESSAGE_CHUNK = False


class LLMResponse:
    """Response from LLM provider."""

    def __init__(
        self,
        content: str,
        model: str,
        provider: str,
        reasoning_content: str | None = None,
        response_metadata: dict | None = None,
    ):
        """Initialize LLM response.

        Args:
            content: Response content text
            model: Model name used
            provider: LLM provider name ("gemini" or "ollama")
            reasoning_content: Reasoning/thinking content if reasoning mode enabled
            response_metadata: Full response metadata from LLM
        """
        self.content = content
        self.model = model
        self.provider = provider
        self.reasoning_content = reasoning_content
        self.response_metadata = response_metadata or {}

    def to_dict(self) -> dict:
        """Convert response to dictionary."""
        result = {
            'content': self.content,
            'model': self.model,
            'provider': self.provider,
        }
        if self.reasoning_content:
            result['reasoning_content'] = self.reasoning_content
        if self.response_metadata:
            result['response_metadata'] = self.response_metadata
        return result


class LLMStreamChunk:
    """A single chunk from streaming LLM response."""

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
            response_metadata: Response metadata from LLM
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


class LLMProvider:
    """Abstraction layer for switchable LLM providers (Gemini/Ollama)."""

    def __init__(
        self,
        provider: str | None = None,
        model: str = DEFAULT_MODEL_FALLBACK,
        gemini_api_key: str | None = None,
        ollama_base_url: str = OLLAMA_BASE_URL,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning: bool | None = None,
    ):
        """Initialize LLM provider.

        Args:
            provider: LLM provider name ("gemini" or "ollama"). If None, uses config default.
            model: Model name to use
            gemini_api_key: Google API key for Gemini (required if provider is "gemini")
            ollama_base_url: Base URL for Ollama API (required if provider is "ollama")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Max number of tokens to generate (Gemini) or num_predict (Ollama)
            reasoning: Enable reasoning/thinking mode for supported models (Ollama-specific).
                True: Captures reasoning in additional_kwargs.reasoning_content
                False: Disables reasoning
                None: Uses model default behavior
        """
        # Use config default if provider not specified
        self.provider = (provider or LLM_PROVIDER).lower()

        if self.provider == 'gemini':
            if not HAS_GEMINI:
                raise ImportError(
                    'langchain-google-genai is not installed. '
                    'Install it with: pip install langchain-google-genai'
                )
            if not gemini_api_key:
                import os

                gemini_api_key = os.getenv('GOOGLE_API_KEY') or os.getenv(
                    'GOOGLE_AI_API_KEY'
                )
            if not gemini_api_key:
                raise ValueError(
                    'Gemini API key is required. Set GOOGLE_API_KEY or '
                    'GOOGLE_AI_API_KEY environment variable, or pass gemini_api_key parameter.'
                )

            # Initialize Gemini LLM
            llm_kwargs: dict[str, Any] = {
                'model': model,
                'google_api_key': gemini_api_key,
            }
            if temperature is not None:
                llm_kwargs['temperature'] = temperature
            if max_tokens is not None:
                llm_kwargs['max_output_tokens'] = max_tokens

            self._llm = ChatGoogleGenerativeAI(**llm_kwargs)
            self.model = model
            self.base_url = None

        elif self.provider == 'ollama':
            if not HAS_OLLAMA:
                raise ImportError(
                    'langchain-ollama is not installed. '
                    'Install it with: pip install langchain-ollama'
                )

            # Initialize Ollama LLM
            llm_kwargs: dict[str, Any] = {
                'model': model,
                'base_url': ollama_base_url,
            }
            if temperature is not None:
                llm_kwargs['temperature'] = temperature
            if max_tokens is not None:
                llm_kwargs['num_predict'] = max_tokens
            if reasoning is not None:
                llm_kwargs['reasoning'] = reasoning

            self._llm = ChatOllama(**llm_kwargs)
            self.model = model
            self.base_url = ollama_base_url

        else:
            raise ValueError(
                f'Unknown LLM provider: {self.provider}. '
                "Supported providers: 'gemini', 'ollama'"
            )

    def chat(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Send a chat message to the LLM.

        Args:
            prompt: User prompt
            **kwargs: Additional provider-specific arguments

        Returns:
            LLMResponse with model response

        Raises:
            Exception: If request fails
        """
        response = self._llm.invoke(prompt, **kwargs)

        # Extract content - handle both string and list formats (Gemini can return lists)
        content = response.content
        if isinstance(content, list):
            # If content is a list (Gemini format), extract text from each item
            # Format: [{'type': 'text', 'text': '...'}, ...]
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    # Extract text from dict items
                    if 'text' in item:
                        text_parts.append(str(item['text']))
                    elif 'content' in item:
                        text_parts.append(str(item['content']))
                elif isinstance(item, str):
                    text_parts.append(item)
                else:
                    text_parts.append(str(item))
            content = ''.join(text_parts)
        elif not isinstance(content, str):
            # Fallback: convert to string if it's not already
            content = str(content)

        # Extract reasoning content if available (Ollama-specific)
        reasoning_content = None
        if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
            reasoning_content = response.additional_kwargs.get('reasoning_content')

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider,
            reasoning_content=reasoning_content,
            response_metadata=getattr(response, 'response_metadata', {}),
        )

    def stream_chat(
        self, prompt: str, **kwargs: Any
    ) -> Generator[LLMStreamChunk, None, None]:
        """Stream a chat message from the LLM.

        Args:
            prompt: User prompt
            **kwargs: Additional provider-specific arguments

        Yields:
            LLMStreamChunk objects with incremental content

        Raises:
            Exception: If request fails
        """
        for chunk in self._llm.stream(prompt, **kwargs):
            if not HAS_AIMESSAGE_CHUNK or not isinstance(chunk, AIMessageChunk):
                # Fallback for non-AIMessageChunk responses
                chunk_content = getattr(chunk, 'content', str(chunk))
                if chunk_content:
                    yield LLMStreamChunk(
                        content=chunk_content,
                        done=False,
                    )
                continue

            # Extract content - handle both string and list formats (Gemini can return lists)
            chunk_content_raw = chunk.content or ''
            if isinstance(chunk_content_raw, list):
                # If content is a list (Gemini format), extract text from each item
                text_parts = []
                for item in chunk_content_raw:
                    if isinstance(item, dict):
                        if 'text' in item:
                            text_parts.append(str(item['text']))
                        elif 'content' in item:
                            text_parts.append(str(item['content']))
                    elif isinstance(item, str):
                        text_parts.append(item)
                    else:
                        text_parts.append(str(item))
                chunk_content = ''.join(text_parts)
            elif not isinstance(chunk_content_raw, str):
                chunk_content = str(chunk_content_raw)
            else:
                chunk_content = chunk_content_raw

            # Extract reasoning content from additional_kwargs (Ollama-specific)
            chunk_reasoning = None
            if hasattr(chunk, 'additional_kwargs') and chunk.additional_kwargs:
                chunk_reasoning = chunk.additional_kwargs.get('reasoning_content')

            # Check if this is the final chunk
            is_done = False
            response_metadata = {}
            if hasattr(chunk, 'response_metadata'):
                response_metadata = chunk.response_metadata or {}
                is_done = response_metadata.get('done', False)

            yield LLMStreamChunk(
                content=chunk_content,
                reasoning_content=chunk_reasoning,
                done=is_done,
                response_metadata=response_metadata if is_done else {},
            )

    def get_langchain_llm(self) -> Any:
        """Get the underlying LangChain LLM instance.

        Returns:
            LangChain LLM instance (ChatGoogleGenerativeAI or ChatOllama)
        """
        return self._llm

    @staticmethod
    def is_gemini_available() -> bool:
        """Check if Gemini is available."""
        return HAS_GEMINI

    @staticmethod
    def is_ollama_available() -> bool:
        """Check if Ollama is available."""
        return HAS_OLLAMA

    @staticmethod
    def get_available_providers() -> list[str]:
        """Get list of available LLM providers.

        Returns:
            List of available provider names
        """
        providers = []
        if HAS_GEMINI:
            providers.append('gemini')
        if HAS_OLLAMA:
            providers.append('ollama')
        return providers
