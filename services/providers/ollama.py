"""
Ollama provider implementation
"""

import logging
import requests
from typing import List, Optional, Dict, Any
from .base_provider import BaseProvider, EmbeddingTaskType

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider):
    """Ollama provider implementation"""

    def __init__(
        self,
        ollama_url: str,
        embedding_model: str,
        llm_model: str,
        rerank_model: str
    ):
        """
        Initialize Ollama provider.

        Args:
            ollama_url: Ollama server URL
            embedding_model: Model for embeddings (e.g., nomic-embed-text)
            llm_model: Model for text generation (e.g., llama3.2:latest)
            rerank_model: Model for reranking (e.g., bge-reranker-v2-m3)
        """
        self.ollama_url = ollama_url.rstrip('/')
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.rerank_model = rerank_model

    def get_embedding(
        self,
        text: str,
        task_type: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_QUERY
    ) -> Optional[List[float]]:
        """
        Get embedding for single text.

        Note: task_type is ignored for Ollama (Google-specific parameter)
        """
        try:
            # Use batch endpoint with single item for consistency
            response = requests.post(
                f"{self.ollama_url}/api/embed",
                json={
                    "model": self.embedding_model,
                    "input": [text[:8000]]  # Single item array, truncate long texts
                },
                timeout=60
            )
            response.raise_for_status()
            embeddings = response.json().get('embeddings', [])

            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            raise ValueError("Ollama API returned empty embeddings response")

        except Exception as e:
            logger.error(f"Failed to get Ollama embedding: {e}")
            raise

    def get_embeddings_batch(
        self,
        texts: List[str],
        task_type: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        batch_size: int = 15
    ) -> List[Optional[List[float]]]:
        """
        Get embeddings in batches.

        Note: task_type is ignored for Ollama (Google-specific parameter)
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            truncated = [text[:8000] for text in batch]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            logger.info(f"Processing Ollama embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

            try:
                response = requests.post(
                    f"{self.ollama_url}/api/embed",
                    json={
                        "model": self.embedding_model,
                        "input": truncated
                    },
                    timeout=120  # 120 seconds for batch
                )
                response.raise_for_status()
                batch_embeddings = response.json().get('embeddings', [])

                # Ensure we got the correct number of embeddings
                if len(batch_embeddings) != len(batch):
                    error_msg = f"Batch {batch_num}: Expected {len(batch)} embeddings but got {len(batch_embeddings)}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Batch {batch_num}/{total_batches} complete ({len(all_embeddings)}/{len(texts)} total)")

            except Exception as e:
                logger.error(f"Ollama batch embedding failed: {e}")
                raise

        # Final check
        if len(all_embeddings) != len(texts):
            error_msg = f"Expected {len(texts)} embeddings but got {len(all_embeddings)}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f"Successfully generated {len(all_embeddings)} Ollama embeddings")
        return all_embeddings

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama.

        Supports Ollama-specific kwargs:
            - logprobs: bool - Return log probabilities
            - top_logprobs: int - Number of top logprobs to return
            - model_override: str - Override the default LLM model
        """
        # Allow overriding the model (e.g., for reranking with different model)
        model = kwargs.pop('model_override', self.llm_model)

        api_request = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }

        # Add Ollama-specific options
        if 'logprobs' in kwargs and kwargs['logprobs']:
            api_request["logprobs"] = True
            if 'top_logprobs' in kwargs:
                api_request["top_logprobs"] = kwargs['top_logprobs']

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=api_request,
                timeout=120
            )
            response.raise_for_status()
            return response.json()

        except Exception as e:
            logger.error(f"Ollama text generation failed: {e}")
            raise

    def list_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            response = requests.get(
                f"{self.ollama_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            models = response.json().get('models', [])
            return [m['name'] for m in models]

        except Exception as e:
            logger.warning(f"Could not list Ollama models: {e}")
            return []

    def get_embedding_dimensions(self) -> int:
        """Get embedding dimensions by testing or using known model dimensions"""
        # Try to get actual dimensions by testing
        test_embedding = self.get_embedding("test")
        if test_embedding:
            dims = len(test_embedding)
            logger.info(f"Detected Ollama embedding dimensions: {dims}")
            return dims

        # Fallback to known dimensions for common models
        model_dims = {
            'nomic-embed-text': 768,
            'mxbai-embed-large': 1024,
            'bge-large': 1024,
            'bge-base': 768,
            'all-minilm': 384,
            'qwen3-embedding': 1024,
            'qwen2-embedding': 1024,
        }

        for model_key, dims in model_dims.items():
            if model_key in self.embedding_model.lower():
                logger.info(f"Using known dimensions for {model_key}: {dims}")
                return dims

        # Default fallback
        logger.warning(f"Unknown embedding model {self.embedding_model}, defaulting to 768 dimensions")
        return 768
