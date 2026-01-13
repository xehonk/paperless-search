"""
Google AI provider implementation using Gemini models
"""

import logging
from typing import List, Optional, Dict, Any
from google import genai
from google.genai import types
from .base_provider import BaseProvider, EmbeddingTaskType

logger = logging.getLogger(__name__)


class GoogleProvider(BaseProvider):
    """Google AI provider implementation"""

    def __init__(
        self,
        api_key: str,
        embedding_model: str = "gemini-embedding-001",
        llm_model: str = "gemini-3-flash-preview"
    ):
        """
        Initialize Google AI provider.

        Args:
            api_key: Google AI API key
            embedding_model: Embedding model (default: gemini-embedding-001, 3072 dims)
            llm_model: LLM model for generation (default: gemini-3-flash-preview)
        """
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Create client
        self.client = genai.Client(api_key=api_key)

        logger.info(f"Initialized Google AI provider with embedding={embedding_model}, llm={llm_model}")

    def get_embedding(
        self,
        text: str,
        task_type: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_QUERY
    ) -> Optional[List[float]]:
        """Get embedding for single text with proper task type"""
        try:
            # Convert task type to Google's format
            task_type_str = "RETRIEVAL_QUERY" if task_type == EmbeddingTaskType.RETRIEVAL_QUERY else "RETRIEVAL_DOCUMENT"

            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text[:8000],  # Truncate long texts
                config=types.EmbedContentConfig(task_type=task_type_str)
            )

            # For single text, result.embeddings[0].values contains the vector
            return result.embeddings[0].values

        except Exception as e:
            logger.error(f"Failed to get Google embedding: {e}")
            raise

    def get_embeddings_batch(
        self,
        texts: List[str],
        task_type: EmbeddingTaskType = EmbeddingTaskType.RETRIEVAL_DOCUMENT,
        batch_size: int = 100  # Google supports larger batches
    ) -> List[Optional[List[float]]]:
        """Get embeddings in batches with proper task type"""
        all_embeddings = []

        # Convert task type to Google's format
        task_type_str = "RETRIEVAL_QUERY" if task_type == EmbeddingTaskType.RETRIEVAL_QUERY else "RETRIEVAL_DOCUMENT"

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            truncated = [text[:8000] for text in batch]
            batch_num = (i // batch_size) + 1
            total_batches = (len(texts) + batch_size - 1) // batch_size

            logger.info(f"Processing Google embedding batch {batch_num}/{total_batches} ({len(batch)} texts)")

            try:
                result = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=truncated,  # Use 'contents' (plural) for multiple texts
                    config=types.EmbedContentConfig(task_type=task_type_str)
                )

                # Extract .values from each ContentEmbedding object in result.embeddings
                batch_embeddings = [emb.values for emb in result.embeddings]

                # Ensure we got the correct number of embeddings
                if len(batch_embeddings) != len(batch):
                    error_msg = f"Batch {batch_num}: Expected {len(batch)} embeddings but got {len(batch_embeddings)}"
                    logger.error(error_msg)
                    raise Exception(error_msg)

                all_embeddings.extend(batch_embeddings)
                logger.debug(f"Batch {batch_num}/{total_batches} complete ({len(all_embeddings)}/{len(texts)} total)")

            except Exception as e:
                logger.error(f"Google batch embedding failed: {e}")
                raise

        # Final check
        if len(all_embeddings) != len(texts):
            error_msg = f"Expected {len(texts)} embeddings but got {len(all_embeddings)}"
            logger.error(error_msg)
            raise Exception(error_msg)

        logger.info(f"Successfully generated {len(all_embeddings)} Google embeddings")
        return all_embeddings

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using Gemini.

        Note: Ignores Ollama-specific kwargs like logprobs, top_logprobs
        """
        # Remove Ollama-specific kwargs
        kwargs.pop('logprobs', None)
        kwargs.pop('top_logprobs', None)
        kwargs.pop('model_override', None)

        try:
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
            )

            # Extract text from response, handling thought_signature and other non-text parts
            # Access candidates.content.parts directly to avoid warning
            text_parts = []
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    parts = candidate.content.parts
                    # Check that parts is not None before iterating
                    if parts is not None:
                        for part in parts:
                            # Only include text parts, skip thought_signature and others
                            if hasattr(part, 'text') and part.text:
                                text_parts.append(part.text)

            # Combine all text parts
            response_text = ''.join(text_parts).strip()

            # Fallback to response.text if we couldn't extract text
            if not response_text:
                response_text = response.text if hasattr(response, 'text') else ''

            # If still no text, log the issue for debugging
            if not response_text:
                logger.warning(f"Empty response from Gemini (may be blocked by safety filters or refusal)")
                # Check for block reasons
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        logger.warning(f"Finish reason: {candidate.finish_reason}")
                    if hasattr(candidate, 'safety_ratings'):
                        logger.warning(f"Safety ratings: {candidate.safety_ratings}")

            # Return in Ollama-compatible format
            return {
                "response": response_text,
                "done": True,
                "done_reason": "stop"
            }

        except Exception as e:
            logger.error(f"Google text generation failed: {e}")
            raise

    def list_models(self) -> List[str]:
        """List available Google models"""
        try:
            models = self.client.models.list()
            model_names = [m.name for m in models]
            logger.info(f"Found {len(model_names)} Google models")
            return model_names

        except Exception as e:
            logger.warning(f"Could not list Google models: {e}")
            return []

    def get_embedding_dimensions(self) -> int:
        """
        Get embedding dimensions.

        gemini-embedding-001 produces 3072-dimensional vectors
        """
        # Known dimension for gemini-embedding-001
        if 'gemini-embedding-001' in self.embedding_model:
            return 3072

        # Fallback: test with actual embedding
        logger.info("Testing Google embedding dimensions...")
        test_embedding = self.get_embedding("test")
        if test_embedding:
            dims = len(test_embedding)
            logger.info(f"Detected Google embedding dimensions: {dims}")
            return dims

        # Default fallback (gemini-embedding-001 standard)
        logger.warning(f"Unknown embedding model {self.embedding_model}, defaulting to 3072 dimensions")
        return 3072
