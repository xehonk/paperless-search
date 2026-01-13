"""
Query processing with LLM-based expansion and understanding
"""

import re
import json
import logging
import os
from typing import Dict, Any, List
import requests
from models import ExpandedQuery
from providers import BaseProvider

logger = logging.getLogger(__name__)

# Load prompt templates
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt(filename: str) -> str:
    """Load a prompt template from the prompts directory"""
    with open(os.path.join(PROMPTS_DIR, filename), 'r') as f:
        return f.read()


class QueryProcessor:
    """Processes and expands search queries using LLM"""

    def __init__(self, provider: BaseProvider, default_language: str = "English"):
        self.provider = provider
        self.default_language = default_language

    def expand_query(self, query: str, enable_expansion: bool = True) -> ExpandedQuery:
        """
        Expand the search query using LLM by adding synonyms and related terms.
        Filters must be applied manually by the user.
        """
        if not enable_expansion:
            return ExpandedQuery(
                original_query=query,
                expanded_query=query,
                extracted_filters={},
                keywords=[]
            )

        try:
            # Build prompt for query understanding
            prompt = self._build_query_expansion_prompt(query)

            logger.info(f"=== Query Expansion Request ===")
            logger.info(f"Prompt length: {len(prompt)} chars")
            logger.info(f"Prompt preview: {prompt[:200]}...")

            # Call provider's LLM
            response_data = self.provider.generate_text(
                prompt=prompt,
                temperature=0.1,
                max_tokens=1024
            )

            llm_output = response_data.get('response', '').strip()
            thinking_output = response_data.get('thinking', '').strip()

            logger.info(f"=== Query Expansion Response ===")
            logger.info(f"Response keys: {list(response_data.keys())}")
            logger.info(f"Response text length: {len(llm_output)} chars")
            logger.info(f"Response text: '{llm_output[:500]}'")
            logger.info(f"Thinking length: {len(thinking_output)} chars")
            logger.info(f"Thinking text: '{thinking_output[:500]}'")
            logger.info(f"Done: {response_data.get('done', 'unknown')}")
            logger.info(f"Done reason: {response_data.get('done_reason', 'unknown')}")

            # If response is empty but thinking has content, use thinking field
            # (qwen3 models output everything in thinking field)
            if not llm_output and thinking_output:
                logger.info("Using thinking field output (qwen3 model)")
                llm_output = thinking_output

            # Parse LLM output
            expanded = self._parse_llm_output(query, llm_output)

            if expanded.expanded_query != query:
                logger.info(f"Query expanded: '{query}' -> '{expanded.expanded_query}'")
            else:
                logger.warning(f"Query not expanded (LLM returned same query): '{query}'")

            return expanded

        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return ExpandedQuery(
                original_query=query,
                expanded_query=query,
                extracted_filters={},
                keywords=[]
            )

    def _build_query_expansion_prompt(self, query: str) -> str:
        """Build prompt for LLM query expansion"""
        template = load_prompt("query_expansion.txt")
        return template.format(query=query, language=self.default_language)

    def _parse_llm_output(self, original_query: str, llm_output: str) -> ExpandedQuery:
        """Parse LLM JSON output into ExpandedQuery"""
        try:
            # Extract JSON from output (LLM might add extra text)
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                expanded_query = parsed.get('expanded_query', original_query)

                return ExpandedQuery(
                    original_query=original_query,
                    expanded_query=expanded_query,
                    extracted_filters={},
                    keywords=[]
                )

        except Exception as e:
            logger.warning(f"Failed to parse LLM output: {e}")

        # Fallback: use original query
        return ExpandedQuery(
            original_query=original_query,
            expanded_query=original_query,
            extracted_filters={},
            keywords=[]
        )

    def extract_highlights(self, text: str, query: str, max_highlights: int = 3) -> List[str]:
        """
        Extract text snippets that match the query for highlighting.
        """
        if not text or not query:
            return []

        highlights = []
        query_terms = query.lower().split()

        # Split text into sentences
        sentences = re.split(r'[.!?]\s+', text)

        for sentence in sentences:
            sentence_lower = sentence.lower()
            # Check if any query term appears in sentence
            if any(term in sentence_lower for term in query_terms):
                # Truncate long sentences
                if len(sentence) > 200:
                    # Find the query term position
                    for term in query_terms:
                        if term in sentence_lower:
                            pos = sentence_lower.index(term)
                            start = max(0, pos - 100)
                            end = min(len(sentence), pos + 100)
                            snippet = sentence[start:end]
                            if start > 0:
                                snippet = "..." + snippet
                            if end < len(sentence):
                                snippet = snippet + "..."
                            highlights.append(snippet)
                            break
                else:
                    highlights.append(sentence)

                if len(highlights) >= max_highlights:
                    break

        return highlights
