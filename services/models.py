"""
Data models for Paperless Qdrant Search
"""

from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """A chunk of a document for indexing"""
    id: str
    paperless_id: int
    chunk_index: int
    chunk_type: str  # 'metadata', 'content'
    title: str
    content: str
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    correspondent_name: Optional[str] = None
    created_date: Optional[int] = None  # Unix timestamp
    modified_date: Optional[int] = None  # Unix timestamp
    added_date: Optional[int] = None  # Unix timestamp
    archive_serial_number: Optional[int] = None
    original_filename: Optional[str] = None
    total_chunks: int = 1


class SearchQuery(BaseModel):
    """Search query from user"""
    query: str
    top_k: Optional[int] = 10
    retrieval_top_k: Optional[int] = None  # Number of results to retrieve before reranking
    filters: Optional[Dict[str, Any]] = None
    enable_reranking: Optional[bool] = True
    enable_query_expansion: Optional[bool] = True
    hybrid_alpha: Optional[float] = 0.5  # 0=keyword, 1=semantic
    recency_weight: Optional[float] = 0.0  # 0.0=no boost, 1.0=full boost
    recency_decay_days: Optional[int] = 365  # Days for 50% decay
    rerank_influence: Optional[float] = 0.2  # How much reranking adjusts score (0.0-1.0, max ±0.2 by default)


class ExplainRequest(BaseModel):
    """Request to explain a rerank score"""
    query: str
    title: str
    content: str
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    correspondent_name: Optional[str] = None


class ExplainResponse(BaseModel):
    """Response with explanation"""
    explanation: str


class ExpandedQuery(BaseModel):
    """Query after LLM expansion"""
    original_query: str
    expanded_query: str
    extracted_filters: Dict[str, Any] = Field(default_factory=dict)
    keywords: List[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    """A single search result"""
    paperless_id: int
    chunk_id: str
    chunk_index: int
    title: str
    content: str
    summary: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    correspondent_name: Optional[str] = None
    created_date: Optional[Union[str, int]] = None  # ISO datetime string or Unix timestamp
    modified_date: Optional[Union[str, int]] = None  # ISO datetime string or Unix timestamp
    modified: Optional[Union[str, int]] = None  # Deprecated - kept for backward compatibility
    score: float
    rerank_score: Optional[float] = None
    rerank_reason: Optional[str] = None  # LLM's explanation for the rerank score
    recency_boost: Optional[float] = 0.0
    final_score: Optional[float] = None  # Blended score from initial + rerank
    highlights: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Complete search response"""
    query: str
    expanded_query: Optional[str] = None
    results: List[SearchResult]
    total_results: int
    retrieval_time_ms: float
    rerank_time_ms: Optional[float] = None
    total_time_ms: float


class PaperlessDocument(BaseModel):
    """Paperless document metadata"""
    id: int
    title: str
    content: Optional[str] = None
    tags: List[Any] = Field(default_factory=list)
    correspondent: Optional[Any] = None
    created: Optional[str] = None
    modified: Optional[str] = None
    added: Optional[str] = None
    archive_serial_number: Optional[int] = None
    original_file_name: Optional[str] = None


class IndexingStats(BaseModel):
    """Statistics about indexing operation"""
    documents_processed: int
    chunks_created: int
    time_taken_seconds: float
    errors: List[str] = Field(default_factory=list)
