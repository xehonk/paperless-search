"""
Provider factory and exports
"""

import os
import logging
from typing import Optional
from .base_provider import BaseProvider, EmbeddingTaskType
from .ollama import OllamaProvider
from .google_ai import GoogleProvider

logger = logging.getLogger(__name__)


def create_provider(
    provider_type: Optional[str] = None,
    ollama_url: Optional[str] = None,
    embedding_model: Optional[str] = None,
    llm_model: Optional[str] = None,
    rerank_model: Optional[str] = None,
    google_api_key: Optional[str] = None
) -> BaseProvider:
    """
    Factory function to create appropriate provider.

    Args:
        provider_type: "ollama" or "google" (defaults to PROVIDER env var)
        ollama_url: Ollama server URL (for Ollama provider)
        embedding_model: Model name for embeddings
        llm_model: Model name for text generation
        rerank_model: Model name for reranking (Ollama only)
        google_api_key: API key for Google (for Google provider)

    Returns:
        Provider instance

    Raises:
        ValueError: If provider_type is invalid or required config is missing
    """
    if provider_type is None:
        provider_type = os.getenv('PROVIDER', 'ollama').lower()

    logger.info(f"Creating provider: {provider_type}")

    if provider_type == 'google':
        # Google AI provider
        api_key = google_api_key or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable required for Google provider. "
                "Get your API key at https://ai.google.dev/"
            )

        embedding_model = embedding_model or os.getenv('GOOGLE_EMBEDDING_MODEL', 'gemini-embedding-001')
        llm_model = llm_model or os.getenv('GOOGLE_LLM_MODEL', 'gemini-3.8-flash')

        logger.info(f"Creating Google provider with embedding={embedding_model}, llm={llm_model}")

        return GoogleProvider(
            api_key=api_key,
            embedding_model=embedding_model,
            llm_model=llm_model
        )

    elif provider_type == 'ollama':
        # Ollama provider
        ollama_url = ollama_url or os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434')
        embedding_model = embedding_model or os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
        llm_model = llm_model or os.getenv('LLM_MODEL', 'llama3.2:latest')
        rerank_model = rerank_model or os.getenv('RERANK_MODEL', 'bge-reranker-v2-m3')

        logger.info(f"Creating Ollama provider at {ollama_url}")

        return OllamaProvider(
            ollama_url=ollama_url,
            embedding_model=embedding_model,
            llm_model=llm_model,
            rerank_model=rerank_model
        )

    else:
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Valid options are 'ollama' or 'google'"
        )


# Export public API
__all__ = ['create_provider', 'BaseProvider', 'EmbeddingTaskType', 'OllamaProvider', 'GoogleProvider']
