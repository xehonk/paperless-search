"""
Base provider abstraction for AI services (embeddings + LLM generation)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from enum import Enum


class EmbeddingTaskType(Enum):
    """Task type for embeddings (Google AI specific, ignored by Ollama)"""
    RETRIEVAL_DOCUMENT = "retrieval_document"  # For indexing documents
    RETRIEVAL_QUERY = "retrieval_query"  # For search queries


class BaseProvider(ABC):
    """Abstract base class for AI providers (embeddings + LLM generation)"""

    @abstractmethod
    def get_embedding(
        self,
        text: str,
        task_type: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_QUERY
    ) -> Optional[List[float]]:
        """
        Get embedding for a single text.

        Args:
            text: Text to embed
            task_type: Type of embedding task (used by Google, ignored by Ollama)

        Returns:
            Embedding vector or None on failure
        """
        pass

    @abstractmethod
    def get_embeddings_batch(
        self,
        texts: List[str],
        task_type: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        batch_size: int = 15
    ) -> List[Optional[List[float]]]:
        """
        Get embeddings for multiple texts with batching.

        Args:
            texts: List of texts to embed
            task_type: Type of embedding task (used by Google, ignored by Ollama)
            batch_size: Number of texts to process per batch

        Returns:
            List of embedding vectors

        Raises:
            Exception: If embedding generation fails
        """
        pass

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using LLM.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0.0 - 1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific options (e.g., logprobs for Ollama)

        Returns:
            Dict with keys:
                - 'response': Generated text
                - 'done': Whether generation is complete
                - 'thinking': Optional reasoning text (for compatible models)
                - Other provider-specific fields

        Raises:
            Exception: If generation fails
        """
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """
        List available models.

        Returns:
            List of model names/IDs

        Note:
            Returns empty list if listing fails
        """
        pass

    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        """
        Get embedding vector dimensions.

        Returns:
            Number of dimensions in embedding vectors

        Note:
            May test with a sample embedding if not known statically
        """
        pass
