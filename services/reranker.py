"""
Cross-encoder reranking for search results
"""

import logging
import math
import os
from typing import List, Tuple, Optional
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import re
from models import SearchResult
from providers import BaseProvider, EmbeddingTaskType

logger = logging.getLogger(__name__)

# Load prompt templates
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory"""
    with open(os.path.join(PROMPTS_DIR, filename), 'r') as f:
        return f.read()


class Reranker:
    """Reranks search results using cross-encoder or similarity scoring"""

    def __init__(self, provider: BaseProvider):
        self.provider = provider
        self.use_cross_encoder = self._check_cross_encoder_available()
        self.rerank_mode = None  # Track which mode is working
        self.failed_llm_calls = 0  # Track failures to auto-switch modes

        # Configurable max_tokens for different operations (from env vars)
        self.dedicated_max_tokens = int(os.getenv('RERANK_DEDICATED_MAX_TOKENS', '10'))
        self.llm_max_tokens = int(os.getenv('RERANK_LLM_MAX_TOKENS', '500'))
        self.explain_max_tokens = int(os.getenv('RERANK_EXPLAIN_MAX_TOKENS', '1500'))
        self.max_workers = int(os.getenv('RERANK_MAX_WORKERS', '20'))

        logger.info(f"Reranker config: max_workers={self.max_workers}, max_tokens: dedicated={self.dedicated_max_tokens}, llm={self.llm_max_tokens}, explain={self.explain_max_tokens}")

    def _check_cross_encoder_available(self) -> bool:
        """Check if cross-encoder model is available"""
        try:
            models = self.provider.list_models()
            model_names = [name.lower() for name in models]

            # Check if any model contains 'rerank' in its name
            available = any('rerank' in name for name in model_names)
            if available:
                logger.info(f"Cross-encoder/rerank model available")
            else:
                logger.info(f"No dedicated rerank model found, will use LLM-based reranking")
            return available

        except Exception as e:
            logger.warning(f"Could not check available models: {e}")
            return False

    def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 10,
        recency_weight: float = 0.0,
        recency_decay_days: int = 365
    ) -> List[SearchResult]:
        """
        Rerank search results using LLM-based scoring with optional recency boost.

        Args:
            query: Search query
            results: List of search results
            top_k: Number of results to return
            recency_weight: Weight for recency boost (0.0 = no boost, 1.0 = full boost)
            recency_decay_days: Days for 50% decay in recency boost

        Returns top_k results sorted by rerank score.
        """
        if not results:
            return []

        # Always use LLM-based reranking (cross-encoder method handles both dedicated and LLM)
        try:
            reranked = self._rerank_with_cross_encoder(query, results, top_k, recency_weight, recency_decay_days)
            logger.info(f"Reranked {len(results)} results to top {len(reranked)} (recency_weight={recency_weight})")
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original results")
            # Return top_k results with original scores
            for result in results[:top_k]:
                result.rerank_score = result.score
                if recency_weight > 0:
                    result.recency_boost = self._calculate_recency_boost(result, recency_weight, recency_decay_days)
                    result.rerank_score = result.rerank_score * (1.0 + result.recency_boost)
            return results[:top_k]

    def rerank_with_progress(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
        recency_weight: float = 0.0,
        recency_decay_days: int = 365
    ):
        """
        Generator that yields results as they are reranked.
        Uses parallel execution and yields results as they complete for progressive UI updates.
        The caller should maintain their own counter for progress tracking.
        """
        if not results:
            return

        import time
        start_time = time.time()
        max_workers = min(self.max_workers, len(results))
        logger.info(f"Starting parallel progressive reranking of {len(results)} results with {max_workers} workers")

        # Track completion stats
        completion_count = 0
        failures = 0
        successes = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scoring tasks
            submit_start = time.time()
            future_to_index = {
                executor.submit(
                    self._score_single_result,
                    query,
                    result,
                    recency_weight,
                    recency_decay_days
                ): idx for idx, result in enumerate(results)
            }
            submit_time = time.time() - submit_start
            logger.info(f"Submitted {len(results)} tasks in {submit_time:.3f}s, waiting for completions...")

            # Yield results as they complete (in completion order, not original order)
            for future in as_completed(future_to_index):
                result, score, success = future.result()

                if success and score is not None:
                    successes += 1
                else:
                    failures += 1

                completion_count += 1
                # Just yield the result - let caller track progress
                yield result

        # Update failure counter
        self.failed_llm_calls += failures
        self.failed_llm_calls = max(0, self.failed_llm_calls - successes)

        elapsed = time.time() - start_time
        avg_time = elapsed / len(results) if results else 0
        logger.info(f"Parallel progressive reranking completed: {completion_count} results in {elapsed:.2f}s (avg {avg_time:.2f}s/doc)")


    def _score_single_result(
        self,
        query: str,
        result: SearchResult,
        recency_weight: float,
        recency_decay_days: int
    ) -> Tuple[SearchResult, Optional[float], bool]:
        """
        Score a single result with the reranker (thread-safe helper).
        Returns: (result, score, success)
        """
        import time
        import threading
        start_time = time.time()
        thread_id = threading.current_thread().name

        try:
            logger.info(f"[{thread_id}] Starting rerank for doc {result.paperless_id}")

            # Build context from result
            context = self._build_result_context(result)

            # Try to score with LLM/reranker
            score = self._score_with_reranker(query, context)
            elapsed = time.time() - start_time
            logger.info(f"[{thread_id}] Doc {result.paperless_id} LLM rerank score: {score} (took {elapsed:.2f}s)")

            if score is None:
                # Scoring failed, use original score
                logger.warning(f"Doc {result.paperless_id}: LLM returned None, using initial score {result.score:.4f}")
                result.rerank_score = result.score
                return (result, None, False)
            else:
                # Convert 0-1 normalized score to 0-10 scale for display/adjustment
                result.rerank_score = score * 10.0

                # Apply recency boost if enabled
                if recency_weight > 0:
                    result.recency_boost = self._calculate_recency_boost(result, recency_weight, recency_decay_days)
                    result.rerank_score = result.rerank_score * (1.0 + result.recency_boost)
                else:
                    result.recency_boost = 0.0

                return (result, result.rerank_score, True)

        except Exception as e:
            logger.warning(f"Failed to rerank result {result.chunk_id}: {e}")
            result.rerank_score = result.score
            result.recency_boost = 0.0
            return (result, None, False)

    def _rerank_with_cross_encoder(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
        recency_weight: float = 0.0,
        recency_decay_days: int = 365
    ) -> List[SearchResult]:
        """
        Rerank using cross-encoder model via Ollama with optional recency boost.
        Supports both dedicated reranker models and LLM-based reranking.
        Uses parallel execution for faster processing.
        """
        import time
        start_time = time.time()

        # Check if we should switch to similarity mode due to failures
        if self.failed_llm_calls > 10:
            logger.warning(f"Too many LLM reranking failures ({self.failed_llm_calls}), switching to similarity mode")
            self.use_cross_encoder = False
            return self._rerank_with_similarity(query, results, top_k, recency_weight, recency_decay_days)

        scored_results = []
        successful_scores = []
        failures = 0
        successes = 0

        # Parallel execution with ThreadPoolExecutor
        # Use configured max_workers (or fewer if we have less results)
        max_workers = min(self.max_workers, len(results))
        logger.info(f"Starting parallel reranking of {len(results)} results with {max_workers} workers")

        parallel_start = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all scoring tasks
            submit_start = time.time()
            future_to_result = {
                executor.submit(
                    self._score_single_result,
                    query,
                    result,
                    recency_weight,
                    recency_decay_days
                ): result for result in results
            }
            submit_time = time.time() - submit_start
            logger.info(f"Submitted {len(results)} tasks in {submit_time:.3f}s")

            # Collect results as they complete
            for future in as_completed(future_to_result):
                result, score, success = future.result()
                scored_results.append(result)

                if success and score is not None:
                    successful_scores.append(score)
                    successes += 1
                else:
                    failures += 1

        parallel_time = time.time() - parallel_start
        total_time = time.time() - start_time
        avg_time = parallel_time / len(results) if results else 0
        logger.info(f"Parallel reranking completed in {parallel_time:.2f}s (avg {avg_time:.2f}s/doc, total {total_time:.2f}s)")

        # Update failure counter (thread-safe after parallel execution completes)
        self.failed_llm_calls += failures
        self.failed_llm_calls = max(0, self.failed_llm_calls - successes)

        # Check if all scores are the same (indicates reranking not working)
        if len(successful_scores) > 5:
            score_variance = max(successful_scores) - min(successful_scores)
            if score_variance < 0.01:  # All scores nearly identical
                logger.warning("Reranking scores show no variance, likely not working properly")
                # Keep the scores but warn the user

        # Sort by rerank score and return top_k
        scored_results.sort(key=lambda x: x.rerank_score, reverse=True)
        return scored_results[:top_k]

    def _rerank_with_similarity(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int,
        recency_weight: float = 0.0,
        recency_decay_days: int = 365
    ) -> List[SearchResult]:
        """
        Rerank using embedding similarity (bi-encoder approach) with optional recency boost.
        Faster than cross-encoder but less accurate.
        """
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)

            # Get embeddings for all result contexts
            contexts = [self._build_result_context(r) for r in results]
            context_embeddings = self._get_embeddings_batch(contexts)

            # Calculate cosine similarity
            for result, context_embedding in zip(results, context_embeddings):
                if query_embedding and context_embedding:
                    similarity = self._cosine_similarity(query_embedding, context_embedding)
                    # Combine with original score (weighted average)
                    result.rerank_score = 0.7 * similarity + 0.3 * result.score
                else:
                    result.rerank_score = result.score

                # Apply recency boost if enabled
                if recency_weight > 0:
                    result.recency_boost = self._calculate_recency_boost(result, recency_weight, recency_decay_days)
                    result.rerank_score = result.rerank_score * (1.0 + result.recency_boost)
                else:
                    result.recency_boost = 0.0

            # Sort and return top_k
            results.sort(key=lambda x: x.rerank_score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Similarity reranking failed: {e}")
            for result in results[:top_k]:
                result.rerank_score = result.score
                result.recency_boost = 0.0
            return results[:top_k]

    def _score_with_reranker(self, query: str, context: str) -> float:
        """
        Score query-context relevance using available reranker method.
        Tries multiple approaches in order:
        1. Dedicated reranker model (if available)
        2. LLM-based scoring via text generation (only for non-reranker models)
        3. Returns None on failure (caller will use similarity/original score)
        """
        # Try Method 1: Check if this is a dedicated reranker model
        if self.use_cross_encoder:
            score = self._score_with_dedicated_reranker(query, context)
            # For dedicated rerankers, don't fall back to LLM method as they
            # are classification models and don't understand general prompts
            return score

        # Try Method 2: LLM-based scoring (only for general LLMs)
        score = self._score_relevance_with_llm(query, context)
        return score

    def _score_with_dedicated_reranker(self, query: str, context: str) -> float:
        """
        Use a dedicated reranker model (e.g., Qwen3-Reranker).
        Uses the proper format and extracts probability from logprobs.
        """
        try:
            # Qwen3-Reranker format (yes/no classification)
            instruction = "Given a web search query, retrieve relevant passages that answer the query"
            template = load_prompt("reranker_dedicated.txt")
            prompt = template.format(
                instruction=instruction,
                query=query,
                document=context[:512]
            )

            logger.info(f"=== Reranker API Request ===")
            logger.info(f"Prompt length: {len(prompt)} chars")
            logger.info(f"Prompt: {prompt[:300]}...")

            # Use provider with Ollama-specific logprobs parameters
            data = self.provider.generate_text(
                prompt=prompt,
                temperature=0.0,
                max_tokens=self.dedicated_max_tokens,
                logprobs=True,  # Ollama-specific
                top_logprobs=10  # Ollama-specific
            )

            # Debug: check what's in the response
            logger.info(f"=== Reranker API Response ===")
            logger.info(f"Response keys: {list(data.keys())}")
            logger.info(f"Response text: '{data.get('response', '')}'")
            logger.info(f"Done: {data.get('done', 'unknown')}")
            logger.info(f"Done reason: {data.get('done_reason', 'unknown')}")
            logger.info(f"Has logprobs: {'logprobs' in data}")

            # Extract logprobs from response
            if "logprobs" in data and data["logprobs"]:
                first_token_logprobs = data["logprobs"][0]  # First (and only) token

                # Look for 'yes' and 'no' in top_logprobs
                token_probs = {}
                if "top_logprobs" in first_token_logprobs:
                    for token_info in first_token_logprobs["top_logprobs"]:
                        token_text = token_info.get("token", "").lower().strip()
                        logprob = token_info.get("logprob", -100)
                        token_probs[token_text] = logprob

                    logger.debug(f"Available tokens: {list(token_probs.keys())[:5]}")

                # Get logprobs for yes/no (with fallback attempts)
                yes_logprob = token_probs.get("yes", token_probs.get("yes.", token_probs.get("▁yes", -100)))
                no_logprob = token_probs.get("no", token_probs.get("no.", token_probs.get("▁no", -100)))

                # Check if we found valid yes/no tokens
                if yes_logprob != -100 or no_logprob != -100:
                    # Convert log probabilities to probabilities using softmax
                    yes_prob = math.exp(yes_logprob)
                    no_prob = math.exp(no_logprob)

                    # Normalize to get final probability
                    total = yes_prob + no_prob
                    if total > 0:
                        score = yes_prob / total
                        logger.debug(f"Reranker score from logprobs: {score:.4f} (yes: {yes_logprob:.2f}, no: {no_logprob:.2f})")
                        return score
                else:
                    logger.warning(f"Could not find yes/no tokens in logprobs, available: {list(token_probs.keys())[:10]}")
                    return None
            else:
                logger.warning(f"No logprobs in response from reranker")
                return None

        except Exception as e:
            logger.warning(f"Dedicated reranker method failed: {e}")

        return None  # Fall back to similarity method

    def _score_relevance_with_llm(self, query: str, context: str) -> float:
        """Use LLM to score query-context relevance via text generation"""
        context_excerpt = context[:400]
        template = load_prompt("reranker_llm_scoring.txt")
        prompt = template.format(query=query, document=context_excerpt)

        try:
            logger.debug(f"LLM scoring with context: '{context_excerpt[:100]}...'")

            # Use provider for LLM scoring
            response_data = self.provider.generate_text(
                prompt=prompt,
                temperature=0.0,
                max_tokens=self.llm_max_tokens
            )

            output = response_data.get('response', '').strip()
            logger.info(f"LLM raw output: '{output}'")

            # Extract number from output (expecting 0-10)
            import re
            match = re.search(r'(\d+(?:\.\d+)?)', output)
            if match:
                raw_score = float(match.group())
                score = raw_score / 10.0  # Normalize 0-10 to 0-1
                # Clamp to valid range
                score = max(0.0, min(1.0, score))
                logger.info(f"LLM score: {raw_score}/10 -> normalized: {score:.4f}")
                return score

            logger.warning(f"Could not parse relevance score from LLM output: '{output}'")

        except requests.exceptions.Timeout:
            logger.warning(f"LLM scoring timeout (120s) for model {self.rerank_model}")
        except requests.exceptions.RequestException as e:
            logger.warning(f"LLM scoring request failed: {e}")
        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}")

        return None  # Let caller decide what to do

    def explain_score(self, query: str, result_context: str) -> str:
        """
        Generate an explanation for why a document received its relevance score.
        This is called on-demand from the UI, not during batch reranking.

        Args:
            query: The search query
            result_context: The document context (built from title, content, etc.)

        Returns:
            str: Explanation text from the LLM
        """
        context_excerpt = result_context[:400]
        template = load_prompt("explain_relevance.txt")
        prompt = template.format(query=query, document=context_excerpt)

        try:
            # Use provider for explanation generation
            response_data = self.provider.generate_text(
                prompt=prompt,
                temperature=0.0,
                max_tokens=self.explain_max_tokens
            )

            output = response_data.get('response', '').strip()
            logger.info(f"Explain output: {output[:200]}...")

            # Extract score and reason
            import re
            score_match = re.search(r'Score:\s*(\d+)', output, re.IGNORECASE)
            reason_match = re.search(r'Reason:\s*(.+?)(?:\n|$)', output, re.IGNORECASE | re.DOTALL)

            if score_match and reason_match:
                score = score_match.group(1)
                reason = reason_match.group(1).strip()
                return f"Score: {score}\nReason: {reason}"

            # If we found just the reason, return it
            if reason_match:
                return reason_match.group(1).strip()

            # Otherwise return the full output (might be formatted differently)
            return output if output else "Unable to generate explanation"

        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return f"Error generating explanation: {str(e)}"

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            # Use provider for embeddings
            embedding = self.provider.get_embedding(
                text=text,
                task_type=EmbeddingTaskType.RETRIEVAL_QUERY
            )
            return embedding if embedding else []

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return []

    def _get_embedding_old(self, text: str) -> List[float]:
        """Old embedding method (kept for reference)"""
        try:
            response = requests.post(
                f"REMOVED/api/embeddings",
                json={
                    "model": self.embedding_model,
                    "prompt": text[:8000]  # Truncate long texts
                },
                timeout=60  # Increased from 10s to 60s for slower models
            )
            response.raise_for_status()
            return response.json().get('embedding', [])

        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            return []

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _calculate_recency_boost(
        self,
        result: SearchResult,
        recency_weight: float,
        recency_decay_days: int
    ) -> float:
        """
        Calculate recency boost based on document creation date.

        Args:
            result: Search result with created date
            recency_weight: Maximum boost weight (0.0 - 1.0)
            recency_decay_days: Days for 50% decay (half-life)

        Returns:
            Boost multiplier (0.0 - recency_weight)
        """
        if not result.created_date or recency_weight == 0:
            return 0.0

        try:
            # Parse created date
            if isinstance(result.created_date, str):
                created_date = datetime.fromisoformat(result.created_date.replace('Z', '+00:00'))
            else:
                created_date = result.created_date

            # Ensure created_date is timezone-aware (assume UTC if naive)
            if created_date.tzinfo is None:
                created_date = created_date.replace(tzinfo=timezone.utc)

            # Calculate age in days
            now = datetime.now(timezone.utc)
            age_days = (now - created_date).days

            # Apply exponential decay: decay_factor = 0.5^(age_days / decay_days)
            # This gives 50% at decay_days, 25% at 2*decay_days, etc.
            decay_factor = 0.5 ** (age_days / recency_decay_days)

            # Scale by recency_weight
            boost = recency_weight * decay_factor

            if boost > 0.01:  # Only log meaningful boosts
                logger.debug(f"Recency boost: age={age_days}d, decay_factor={decay_factor:.3f}, boost={boost:.3f}")

            return boost

        except Exception as e:
            logger.warning(f"Failed to calculate recency boost: {e}")
            return 0.0

    def _build_result_context(self, result: SearchResult) -> str:
        """Build context string from search result for reranking"""
        parts = [
            f"Title: {result.title}",
        ]

        if result.summary:
            parts.append(f"Summary: {result.summary}")

        if result.tags:
            parts.append(f"Tags: {', '.join(result.tags)}")

        if result.correspondent_name:
            parts.append(f"From: {result.correspondent_name}")

        # Add content excerpt
        content_excerpt = result.content[:500] if result.content else ""
        if content_excerpt:
            parts.append(f"Content: {content_excerpt}")

        return " | ".join(parts)
