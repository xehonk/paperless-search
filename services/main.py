"""
Main FastAPI application for Paperless Qdrant Search
Serves search API and web UI (indexing runs in separate service)
"""

import os
import logging
import time
import json
import asyncio
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from models import SearchQuery, SearchResponse, ExplainRequest, ExplainResponse
from retriever import Retriever
from reranker import Reranker
from query_processor import QueryProcessor

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
PAPERLESS_URL = os.getenv('PAPERLESS_URL', 'http://localhost:8000')
PAPERLESS_PUBLIC_URL = os.getenv('PAPERLESS_PUBLIC_URL', '').strip() or PAPERLESS_URL
PAPERLESS_API_KEY = os.getenv('PAPERLESS_API_KEY')

# Validate required configuration
if not PAPERLESS_API_KEY:
    logger.error("PAPERLESS_API_KEY environment variable is required")
    import sys
    sys.exit(1)

QDRANT_URL = os.getenv('QDRANT_URL', 'http://qdrant:6333')
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'documents')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
RERANK_MODEL = os.getenv('RERANK_MODEL', 'bge-reranker-v2-m3')
LLM_MODEL = os.getenv('LLM_MODEL', 'llama3.2:latest')
DEFAULT_QUERY_LANGUAGE = os.getenv('DEFAULT_QUERY_LANGUAGE', 'English')
ENABLE_QUERY_EXPANSION = os.getenv('ENABLE_QUERY_EXPANSION', 'true').lower() == 'true'
ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
RETRIEVAL_TOP_K = int(os.getenv('RETRIEVAL_TOP_K', '50'))
RERANK_TOP_K = int(os.getenv('RERANK_TOP_K', '10'))
HYBRID_ALPHA = float(os.getenv('HYBRID_ALPHA', '0.5'))
RERANK_INFLUENCE = float(os.getenv('RERANK_INFLUENCE', '0.2'))

# Global components
retriever: Optional[Retriever] = None
reranker: Optional[Reranker] = None
query_processor: Optional[QueryProcessor] = None


def initialize_components():
    """Initialize search components (indexing runs in separate service)"""
    global retriever, reranker, query_processor

    logger.info("Initializing search components...")

    # Create provider
    from providers import create_provider
    provider = create_provider()

    # Initialize retriever
    retriever = Retriever(
        qdrant_url=QDRANT_URL,
        collection_name=QDRANT_COLLECTION,
        provider=provider
    )

    # Initialize reranker
    reranker = Reranker(provider=provider)

    # Initialize query processor
    query_processor = QueryProcessor(
        provider=provider,
        default_language=DEFAULT_QUERY_LANGUAGE
    )

    logger.info("Search components initialized successfully")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    # Startup
    logger.info("=" * 60)
    logger.info("Paperless Qdrant Search Web UI starting...")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Ollama URL: {OLLAMA_URL}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Rerank model: {RERANK_MODEL}")
    logger.info(f"LLM model: {LLM_MODEL}")
    logger.info("Note: Indexing runs in separate service")
    logger.info("=" * 60)

    initialize_components()

    logger.info("Web UI ready on http://0.0.0.0:8080")

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Paperless Qdrant Search",
    description="Google-like search for Paperless documents with RAG",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static files
app.mount("/static", StaticFiles(directory="web-ui/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="web-ui")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the search UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/search", response_model=SearchResponse)
async def search(query: SearchQuery):
    """
    Search documents with multi-stage retrieval and reranking
    """
    if not query.query or not query.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.time()

    try:
        # Step 1: Query expansion
        logger.info(f"Search query: '{query.query}'")
        expansion_enabled = query.enable_query_expansion if query.enable_query_expansion is not None else ENABLE_QUERY_EXPANSION
        expanded_query = query_processor.expand_query(query.query, expansion_enabled)

        # Merge UI filters with LLM-extracted filters (UI filters take precedence)
        if query.filters:
            expanded_query.extracted_filters.update(query.filters)
            logger.info(f"Applied UI filters: {query.filters}")

        # Step 2: Retrieval
        retrieval_start = time.time()
        top_k = query.top_k or RERANK_TOP_K

        # Allow user to override retrieval_top_k
        if query.retrieval_top_k is not None:
            retrieval_limit = query.retrieval_top_k
            logger.info(f"Using user-specified retrieval_top_k={retrieval_limit}")
        else:
            # Qdrant groups by paperless_id, so we get unique documents directly
            retrieval_limit = RETRIEVAL_TOP_K if ENABLE_RERANKING else top_k
            logger.info(f"Using calculated retrieval_limit={retrieval_limit} unique documents")

        hybrid_alpha = query.hybrid_alpha if query.hybrid_alpha is not None else HYBRID_ALPHA

        results = retriever.retrieve(
            expanded_query=expanded_query,
            top_k=retrieval_limit,
            hybrid_alpha=hybrid_alpha
        )
        retrieval_time = (time.time() - retrieval_start) * 1000
        logger.info(f"Retrieved {len(results)} unique documents (Qdrant-level deduplication)")

        # Step 3: Reranking
        rerank_time = None
        reranking_enabled = query.enable_reranking if query.enable_reranking is not None else ENABLE_RERANKING

        # Get recency parameters
        recency_weight = query.recency_weight if query.recency_weight is not None else 0.0
        recency_decay_days = query.recency_decay_days if query.recency_decay_days is not None else 365
        rerank_influence = query.rerank_influence if query.rerank_influence is not None else RERANK_INFLUENCE

        if reranking_enabled:
            rerank_start = time.time()
            # Rerank all results (not just top_k, we'll select top_k after blending)
            results = reranker.rerank(
                query.query,
                results,
                len(results),  # Rerank all deduplicated results
                recency_weight=recency_weight,
                recency_decay_days=recency_decay_days
            )
            rerank_time = (time.time() - rerank_start) * 1000
        elif recency_weight > 0:
            # Apply recency boost even without reranking
            for result in results:
                result.recency_boost = reranker._calculate_recency_boost(result, recency_weight, recency_decay_days)
                result.rerank_score = result.score * (1.0 + result.recency_boost)

        # Step 3.5: Blend initial and rerank scores
        logger.info(f"Blending scores with rerank_influence={rerank_influence}")
        for result in results:
            if result.rerank_score is not None:
                # Normalize rerank_score (0-10) to -1 to +1 range
                normalized_rerank = (result.rerank_score - 5.0) / 5.0
                # Calculate adjustment based on influence
                adjustment = normalized_rerank * rerank_influence
                # Apply to initial score
                result.final_score = max(0.0, min(1.0, result.score + adjustment))
                logger.debug(f"Doc {result.paperless_id}: initial={result.score:.4f}, rerank={result.rerank_score:.2f}/10, adjustment={adjustment:+.4f}, final={result.final_score:.4f}")
            else:
                # No reranking, use initial score
                result.final_score = result.score
                logger.debug(f"Doc {result.paperless_id}: initial={result.score:.4f}, no rerank, final={result.final_score:.4f}")

        # Sort by final score and take top_k
        results.sort(key=lambda x: x.final_score, reverse=True)

        # Log top results before limiting
        logger.info(f"Top {min(top_k, len(results))} results after sorting:")
        for i, result in enumerate(results[:top_k]):
            rerank_str = f"{result.rerank_score:.2f}/10" if result.rerank_score is not None else "N/A"
            logger.info(f"  #{i+1}: Doc {result.paperless_id} - final={result.final_score:.4f}, initial={result.score:.4f}, rerank={rerank_str}")

        results = results[:top_k]

        # Step 4: Add highlights
        for result in results:
            result.highlights = query_processor.extract_highlights(
                result.content,
                query.query,
                max_highlights=2
            )

        total_time = (time.time() - start_time) * 1000

        return SearchResponse(
            query=query.query,
            expanded_query=expanded_query.expanded_query if expansion_enabled else None,
            results=results[:top_k],
            total_results=len(results),
            retrieval_time_ms=retrieval_time,
            rerank_time_ms=rerank_time,
            total_time_ms=total_time
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/search/stream")
async def search_stream(query: SearchQuery, request: Request):
    """
    Search with progressive result streaming via SSE
    """
    async def generate():
        try:
            start_time = time.time()

            # Step 1: Query expansion
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'query_expansion', 'message': 'Expanding query...'})}\n\n"
            await asyncio.sleep(0)  # Force flush the stream

            expansion_enabled = query.enable_query_expansion if query.enable_query_expansion is not None else ENABLE_QUERY_EXPANSION
            expanded_query = query_processor.expand_query(query.query, expansion_enabled)

            # Merge UI filters with LLM-extracted filters (UI filters take precedence)
            if query.filters:
                expanded_query.extracted_filters.update(query.filters)
                logger.info(f"Applied UI filters: {query.filters}")

            yield f"data: {json.dumps({'type': 'progress', 'stage': 'query_expansion', 'message': 'Query expanded', 'done': True})}\n\n"

            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("Client disconnected during query expansion")
                return

            # Step 2: Retrieval
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'retrieval', 'message': 'Retrieving documents...'})}\n\n"
            await asyncio.sleep(0)  # Force flush the stream

            retrieval_start = time.time()
            top_k = query.top_k or RERANK_TOP_K

            if query.retrieval_top_k is not None:
                retrieval_limit = query.retrieval_top_k
            else:
                retrieval_limit = RETRIEVAL_TOP_K if ENABLE_RERANKING else top_k

            hybrid_alpha = query.hybrid_alpha if query.hybrid_alpha is not None else HYBRID_ALPHA

            results = retriever.retrieve(
                expanded_query=expanded_query,
                top_k=retrieval_limit,
                hybrid_alpha=hybrid_alpha
            )
            retrieval_time = (time.time() - retrieval_start) * 1000
            logger.info(f"Retrieval completed in {retrieval_time:.0f}ms")

            yield f"data: {json.dumps({'type': 'progress', 'stage': 'retrieval', 'message': f'Retrieved {len(results)} documents', 'done': True, 'count': len(results)})}\n\n"

            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("Client disconnected after retrieval")
                return

            # Send initial results WITHOUT highlights (fast)
            logger.info("Preparing initial results to send...")
            prep_start = time.time()

            recency_weight = query.recency_weight if query.recency_weight is not None else 0.0
            rerank_influence = query.rerank_influence if query.rerank_influence is not None else RERANK_INFLUENCE

            # Set initial final_score = score for all results (no highlights yet)
            for result in results:
                result.final_score = result.score
                result.highlights = []  # No highlights yet

            logger.info(f"Prepared {len(results)} results in {(time.time() - prep_start) * 1000:.0f}ms")

            # Send ALL results to client - client will sort and display top_k
            serialize_start = time.time()
            initial_response = SearchResponse(
                query=query.query,
                expanded_query=expanded_query.expanded_query if expansion_enabled else None,
                results=results,  # Send all results without highlights
                total_results=len(results),
                retrieval_time_ms=retrieval_time,
                rerank_time_ms=None,
                total_time_ms=(time.time() - start_time) * 1000
            )
            logger.info(f"Serialized response in {(time.time() - serialize_start) * 1000:.0f}ms")

            send_start = time.time()
            json_data = json.dumps({'type': 'initial_results', 'data': initial_response.model_dump()})
            logger.info(f"JSON size: {len(json_data)} bytes ({len(json_data)/1024:.1f} KB)")
            yield f"data: {json_data}\n\n"
            await asyncio.sleep(0)  # Force flush the stream
            logger.info(f"Sent initial_results in {(time.time() - send_start) * 1000:.0f}ms")

            # Step 3: Reranking with progressive updates
            reranking_enabled = query.enable_reranking if query.enable_reranking is not None else ENABLE_RERANKING
            recency_decay_days = query.recency_decay_days if query.recency_decay_days is not None else 365
            rerank_time = None

            if reranking_enabled:
                rerank_start = time.time()

                yield f"data: {json.dumps({'type': 'progress', 'stage': 'reranking', 'message': 'Starting reranking...', 'current': 0, 'total': len(results)})}\n\n"
                await asyncio.sleep(0)  # Force flush the stream

                # Rerank with progress updates
                total_docs = len(results)
                completed_count = 0
                for reranked_result in reranker.rerank_with_progress(
                    query.query,
                    results,
                    len(results),
                    recency_weight=recency_weight,
                    recency_decay_days=recency_decay_days
                ):
                    completed_count += 1

                    # Check if client disconnected
                    if await request.is_disconnected():
                        logger.info(f"Client disconnected during reranking at {completed_count}/{total_docs}")
                        return

                    # Calculate final score with blending
                    if reranked_result.rerank_score is not None:
                        normalized_rerank = (reranked_result.rerank_score - 5.0) / 5.0
                        adjustment = normalized_rerank * rerank_influence
                        reranked_result.final_score = max(0.0, min(1.0, reranked_result.score + adjustment))
                    else:
                        reranked_result.final_score = reranked_result.score

                    # Send individual result update
                    yield f"data: {json.dumps({'type': 'result_update', 'paperless_id': reranked_result.paperless_id, 'rerank_score': reranked_result.rerank_score, 'final_score': reranked_result.final_score, 'recency_boost': reranked_result.recency_boost})}\n\n"

                    # Update progress
                    yield f"data: {json.dumps({'type': 'progress', 'stage': 'reranking', 'message': f'Reranking {completed_count}/{total_docs}', 'current': completed_count, 'total': total_docs})}\n\n"

                rerank_time = (time.time() - rerank_start) * 1000
                yield f"data: {json.dumps({'type': 'progress', 'stage': 'reranking', 'message': 'Reranking complete', 'done': True})}\n\n"

            # Step 4: Generate highlights for top_k results only (after reranking)
            yield f"data: {json.dumps({'type': 'progress', 'stage': 'finalize', 'message': 'Generating highlights...'})}\n\n"
            await asyncio.sleep(0)  # Force flush the stream

            # Sort by final score and get top_k
            results.sort(key=lambda x: x.final_score, reverse=True)
            top_results = results[:top_k]

            # Generate highlights for top_k only
            highlights_map = {}
            for result in top_results:
                result.highlights = query_processor.extract_highlights(
                    result.content,
                    query.query,
                    max_highlights=2
                )
                highlights_map[result.paperless_id] = result.highlights

            # Send highlights
            yield f"data: {json.dumps({'type': 'highlights', 'highlights': highlights_map})}\n\n"

            total_time = (time.time() - start_time) * 1000

            # Send completion
            yield f"data: {json.dumps({'type': 'done', 'total_time_ms': total_time, 'rerank_time_ms': rerank_time})}\n\n"

        except Exception as e:
            logger.error(f"Stream search failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.get("/api/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "qdrant_url": QDRANT_URL,
        "ollama_url": OLLAMA_URL
    }


@app.get("/api/paperless/tags")
async def get_paperless_tags():
    """Fetch all tags from Paperless (with pagination)"""
    try:
        import requests
        all_tags = []
        page = 1

        while True:
            response = requests.get(
                f"{PAPERLESS_URL}/api/tags/?page={page}&page_size=100",
                headers={"Authorization": f"Token {PAPERLESS_API_KEY}"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Extract tag data including colors and document counts
            tags = [{
                "id": tag["id"],
                "name": tag["name"],
                "color": tag.get("color", "#808080"),
                "document_count": tag.get("document_count", 0)
            } for tag in data.get("results", [])]

            all_tags.extend(tags)

            # Check if there are more pages
            if not data.get("next"):
                break
            page += 1

        logger.info(f"Fetched {len(all_tags)} tags from Paperless")
        return {"tags": all_tags}
    except Exception as e:
        logger.error(f"Failed to fetch tags from Paperless: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch tags: {str(e)}")


@app.get("/api/paperless/correspondents")
async def get_paperless_correspondents():
    """Fetch all correspondents from Paperless (with pagination)"""
    try:
        import requests
        all_correspondents = []
        page = 1

        while True:
            response = requests.get(
                f"{PAPERLESS_URL}/api/correspondents/?page={page}&page_size=100",
                headers={"Authorization": f"Token {PAPERLESS_API_KEY}"},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Extract correspondent data including document counts
            correspondents = [{
                "id": corr["id"],
                "name": corr["name"],
                "document_count": corr.get("document_count", 0)
            } for corr in data.get("results", [])]

            all_correspondents.extend(correspondents)

            # Check if there are more pages
            if not data.get("next"):
                break
            page += 1

        logger.info(f"Fetched {len(all_correspondents)} correspondents from Paperless")
        return {"correspondents": all_correspondents}
    except Exception as e:
        logger.error(f"Failed to fetch correspondents from Paperless: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch correspondents: {str(e)}")


@app.get("/api/config")
async def get_config():
    """Get current configuration"""
    return {
        "paperless_url": PAPERLESS_PUBLIC_URL,  # Use public URL for UI links
        "embedding_model": EMBEDDING_MODEL,
        "rerank_model": RERANK_MODEL,
        "llm_model": LLM_MODEL,
        "enable_query_expansion": ENABLE_QUERY_EXPANSION,
        "enable_reranking": ENABLE_RERANKING,
        "retrieval_top_k": RETRIEVAL_TOP_K,
        "rerank_top_k": RERANK_TOP_K,
        "hybrid_alpha": HYBRID_ALPHA,
        "rerank_influence": RERANK_INFLUENCE
    }


@app.post("/api/explain", response_model=ExplainResponse)
async def explain(request: ExplainRequest):
    """
    Generate an explanation for why a document received its rerank score
    """
    try:
        # Build context from the request (same format as in reranker)
        parts = [f"Title: {request.title}"]

        if request.summary:
            parts.append(f"Summary: {request.summary}")

        if request.tags:
            parts.append(f"Tags: {', '.join(request.tags)}")

        if request.correspondent_name:
            parts.append(f"From: {request.correspondent_name}")

        # Add content excerpt
        content_excerpt = request.content[:500] if request.content else ""
        if content_excerpt:
            parts.append(f"Content: {content_excerpt}")

        context = " | ".join(parts)

        # Get explanation from reranker
        explanation = reranker.explain_score(request.query, context)

        return ExplainResponse(explanation=explanation)

    except Exception as e:
        logger.error(f"Explanation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


if __name__ == "__main__":
    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
