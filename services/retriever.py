"""
Multi-stage retrieval from Qdrant with hybrid search
"""

import logging
from typing import List, Dict, Any, Optional
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
)
from models import SearchResult, ExpandedQuery
from providers import BaseProvider, EmbeddingTaskType

logger = logging.getLogger(__name__)


class Retriever:
    """Handles document retrieval from Qdrant"""

    def __init__(
        self,
        qdrant_url: str,
        collection_name: str,
        provider: BaseProvider
    ):
        self.qdrant_client = QdrantClient(url=qdrant_url)
        self.collection_name = collection_name
        self.provider = provider

    def retrieve(
        self,
        expanded_query: ExpandedQuery,
        top_k: int = 50,
        hybrid_alpha: float = 0.5
    ) -> List[SearchResult]:
        """
        Retrieve documents using hybrid search (keyword + semantic).

        Args:
            expanded_query: Query with expansion and filters
            top_k: Number of results to retrieve
            hybrid_alpha: Balance between keyword (0.0) and semantic (1.0) search

        Returns:
            List of search results
        """
        try:
            # Get query embedding for semantic search
            query_embedding = self._get_query_embedding(expanded_query.expanded_query)

            # Build Qdrant filter from extracted filters
            qdrant_filter = self._build_filter(expanded_query.extracted_filters)

            # Perform hybrid search
            if hybrid_alpha > 0.0 and query_embedding:
                # Use vector search
                results = self._vector_search(
                    query_embedding,
                    qdrant_filter,
                    top_k
                )
            else:
                # Use keyword search only (via metadata filtering)
                # Note: Qdrant doesn't have native BM25, so we do best-effort text matching
                results = self._keyword_search(
                    expanded_query.expanded_query,
                    qdrant_filter,
                    top_k
                )

            logger.info(f"Retrieved {len(results)} results for query: {expanded_query.original_query}")
            return results

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []

    def _get_query_embedding(self, query: str) -> Optional[List[float]]:
        """Get embedding for search query"""
        try:
            # Use provider's embedding with RETRIEVAL_QUERY task type
            embedding = self.provider.get_embedding(
                text=query,
                task_type=EmbeddingTaskType.RETRIEVAL_QUERY
            )

            if embedding:
                logger.debug(f"Generated query embedding with {len(embedding)} dimensions")
                return embedding
            else:
                logger.error("No embedding returned from provider")
                return None

        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            return None

    def _vector_search(
        self,
        query_embedding: List[float],
        qdrant_filter: Optional[Filter],
        top_k: int
    ) -> List[SearchResult]:
        """Perform vector similarity search in Qdrant with automatic deduplication by document ID"""
        try:
            # Use query_points_groups to get one result per document (grouped by paperless_id)
            # Note: In qdrant-client 1.16+, search_groups was replaced with query_points_groups
            logger.info(f"Searching Qdrant with filter: {qdrant_filter}, limit={top_k}")
            search_result = self.qdrant_client.query_points_groups(
                collection_name=self.collection_name,
                query=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k,  # Number of unique documents to return
                group_by="paperless_id",  # Group by document ID
                group_size=1,  # Only return best chunk per document
                with_payload=True
            )

            logger.info(f"Qdrant returned {len(search_result.groups)} groups")
            results = []
            # query_points_groups returns groups, each containing the best chunk for that document
            for group in search_result.groups:
                # Each group has a list of hits (we requested group_size=1, so just one)
                if group.hits:
                    point = group.hits[0]  # Best chunk for this document
                    result = self._point_to_search_result(point)
                    if result:
                        results.append(result)

            logger.info(f"Retrieved {len(results)} unique documents (grouped by paperless_id)")
            return results

        except Exception as e:
            logger.error(f"Vector search with grouping failed: {e}")
            return []

    def _keyword_search(
        self,
        query: str,
        qdrant_filter: Optional[Filter],
        top_k: int
    ) -> List[SearchResult]:
        """
        Perform keyword search (best-effort text matching).
        Note: Qdrant doesn't have built-in BM25, so we do filtering + scroll.
        """
        try:
            # Scroll through collection with filters
            points, _ = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
                with_vectors=False
            )

            results = []
            query_terms = set(query.lower().split())

            for point in points:
                result = self._point_to_search_result(point)
                if result:
                    # Simple keyword matching score
                    content_lower = result.content.lower()
                    title_lower = result.title.lower()

                    # Count query term matches
                    matches = sum(1 for term in query_terms if term in content_lower or term in title_lower)
                    result.score = matches / len(query_terms) if query_terms else 0.0

                    results.append(result)

            # Sort by score
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []

    def _build_filter(self, extracted_filters: Dict[str, Any]) -> Optional[Filter]:
        """Build Qdrant filter from extracted filters"""
        if not extracted_filters:
            return None

        logger.info(f"Building filter from: {extracted_filters}")
        conditions = []

        # Tags filter
        if 'tags' in extracted_filters:
            tags = extracted_filters['tags']
            tag_logic = extracted_filters.get('tag_logic', 'any')  # Default to 'any'
            logger.info(f"Processing tags filter: {tags} with logic: {tag_logic}")

            if isinstance(tags, list) and tags:
                if tag_logic == 'all':
                    # Match ALL tags - each tag must be present
                    for tag in tags:
                        tag_condition = FieldCondition(
                            key="tags",
                            match=MatchAny(any=[tag])
                        )
                        logger.info(f"Adding tag condition (all): key='tags', match=MatchAny(any=[{tag}])")
                        conditions.append(tag_condition)
                else:
                    # Match ANY of the tags (default)
                    tag_condition = FieldCondition(
                        key="tags",
                        match=MatchAny(any=tags)
                    )
                    logger.info(f"Adding tag condition (any): key='tags', match=MatchAny(any={tags})")
                    conditions.append(tag_condition)

        # Correspondent filter
        if 'correspondent_name' in extracted_filters:
            correspondent = extracted_filters['correspondent_name']
            logger.info(f"Processing correspondent filter: '{correspondent}'")
            correspondent_condition = FieldCondition(
                key="correspondent_name",
                match=MatchValue(value=correspondent)
            )
            logger.info(f"Adding correspondent condition: key='correspondent_name', match=MatchValue(value='{correspondent}')")
            conditions.append(correspondent_condition)

        # Date range filter
        if 'created_date' in extracted_filters:
            date_range = extracted_filters['created_date']
            if isinstance(date_range, dict):
                # Range filter for dates (convert ISO strings to Unix timestamps if needed)
                from datetime import datetime
                range_dict = {}

                if 'gte' in date_range:
                    val = date_range['gte']
                    # Convert ISO date string to Unix timestamp if it's a string
                    if isinstance(val, str):
                        dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                        range_dict['gte'] = int(dt.timestamp())
                    else:
                        range_dict['gte'] = val

                if 'lte' in date_range:
                    val = date_range['lte']
                    # Convert ISO date string to Unix timestamp if it's a string
                    if isinstance(val, str):
                        # For 'lte', add 23:59:59 to include the entire day
                        if len(val) == 10:  # YYYY-MM-DD format
                            val = val + 'T23:59:59'
                        dt = datetime.fromisoformat(val.replace('Z', '+00:00'))
                        range_dict['lte'] = int(dt.timestamp())
                    else:
                        range_dict['lte'] = val

                if range_dict:
                    logger.info(f"Applying date filter: {range_dict}")
                    conditions.append(
                        FieldCondition(
                            key="created_date",
                            range=Range(**range_dict)
                        )
                    )

        if not conditions:
            logger.info("No filter conditions, returning None")
            return None

        # Combine conditions with AND logic
        final_filter = Filter(must=conditions)
        logger.info(f"Final filter: {len(conditions)} conditions combined with AND logic")
        return final_filter

    def _timestamp_to_iso(self, timestamp) -> Optional[str]:
        """Convert Unix timestamp to ISO date string"""
        if timestamp is None:
            return None
        if isinstance(timestamp, str):
            return timestamp  # Already a string
        if isinstance(timestamp, int):
            from datetime import datetime
            try:
                return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
            except Exception as e:
                logger.warning(f"Failed to convert timestamp {timestamp}: {e}")
                return None
        return None

    def _point_to_search_result(self, point) -> Optional[SearchResult]:
        """Convert Qdrant point to SearchResult"""
        try:
            payload = point.payload
            score = point.score if hasattr(point, 'score') else 0.0

            # Convert timestamps to ISO strings for API response
            created_date = self._timestamp_to_iso(payload.get('created_date'))
            modified_date = self._timestamp_to_iso(payload.get('modified_date'))

            return SearchResult(
                paperless_id=payload.get('paperless_id', 0),
                chunk_id=payload.get('id', str(point.id)),
                chunk_index=payload.get('chunk_index', 0),
                title=payload.get('title', ''),
                content=payload.get('content', ''),
                summary=payload.get('summary'),
                tags=payload.get('tags', []),
                correspondent_name=payload.get('correspondent_name'),
                created_date=created_date,
                modified_date=modified_date,
                modified=modified_date,  # For recency calculation
                score=score,
                highlights=[]
            )

        except Exception as e:
            logger.warning(f"Failed to convert point to result: {e}")
            return None
