"""
Document indexer for Paperless-ngx to Qdrant
"""

import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    Distance,
    VectorParams,
    OptimizersConfigDiff,
)
from models import DocumentChunk, PaperlessDocument, IndexingStats
from providers import BaseProvider, EmbeddingTaskType

logger = logging.getLogger(__name__)


def iso_to_timestamp(iso_date: Optional[str]) -> Optional[int]:
    """Convert ISO date string to Unix timestamp"""
    if not iso_date:
        return None
    try:
        # Handle various ISO formats
        dt = datetime.fromisoformat(iso_date.replace('Z', '+00:00'))
        return int(dt.timestamp())
    except Exception as e:
        logger.warning(f"Failed to convert date '{iso_date}': {e}")
        return None


class DocumentIndexer:
    """Indexes Paperless documents into Qdrant"""

    def __init__(
        self,
        paperless_url: str,
        paperless_api_key: str,
        qdrant_url: str,
        collection_name: str,
        provider: BaseProvider,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        require_reviewed: bool = True
    ):
        self.paperless_url = paperless_url.rstrip('/')
        self.paperless_api_key = paperless_api_key
        self.collection_name = collection_name
        self.provider = provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.require_reviewed = require_reviewed

        self.headers = {
            'Authorization': f'Token {paperless_api_key}',
            'Content-Type': 'application/json'
        }

        self.qdrant_client = QdrantClient(url=qdrant_url)

        # State file
        self.state_file = Path('/app/data/last_poll.json')

        # Cache for tags and correspondents (ID -> name mappings)
        self.tags_cache = {}  # {tag_id: tag_name}
        self.correspondents_cache = {}  # {correspondent_id: correspondent_name}

        # Retry configuration
        self.max_retries = 10
        self.retry_backoff_minutes = [0, 5, 15, 60, 240, 720]  # 0min, 5min, 15min, 1hr, 4hr, 12hr

    def initialize_collection(self, vector_size: int):
        """Initialize or update Qdrant collection"""
        try:
            # Check if collection exists
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if self.collection_name in collection_names:
                logger.info(f"Collection '{self.collection_name}' already exists")
                return

            # Create new collection
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                ),
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )

            # Create payload indexes for filtering
            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="paperless_id",
                field_schema="integer"
            )

            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="tags",
                field_schema="keyword"
            )

            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="correspondent_name",
                field_schema="keyword"
            )

            self.qdrant_client.create_payload_index(
                collection_name=self.collection_name,
                field_name="created_date",
                field_schema="integer"  # Use integer for Unix timestamps
            )

            logger.info(f"Collection '{self.collection_name}' created successfully")

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            raise

    def poll_and_index(self) -> IndexingStats:
        """Poll Paperless for new/updated documents and index them"""
        start_time = time.time()
        stats = IndexingStats(
            documents_processed=0,
            chunks_created=0,
            time_taken_seconds=0.0,
            errors=[]
        )

        # Load tags and correspondents cache before indexing
        logger.info("Loading tags and correspondents from Paperless...")
        self._load_tags_cache()
        self._load_correspondents_cache()

        try:
            # Load state (last poll time + failed documents)
            state = self._load_state()
            last_poll = state.get('last_poll')
            failed_docs = state.get('failed_documents', {})

            # Step 1: Retry failed documents with backoff
            retry_count = 0
            for doc_id, failure_info in list(failed_docs.items()):
                if self._should_retry(failure_info):
                    retry_count += 1
                    success = self._retry_document(doc_id, failure_info, stats)
                    if success:
                        # Remove from failed list
                        del failed_docs[doc_id]
                        self._save_state(last_poll, failed_docs)
                    else:
                        # Update failure info
                        self._save_state(last_poll, failed_docs)

            if retry_count > 0:
                logger.info(f"Retried {retry_count} previously failed documents")

            # Step 2: Fetch new/updated documents from Paperless
            first_page_params = {
                'page_size': 50,
                'ordering': 'modified',
                'truncate_content': 'false'
            }

            if last_poll:
                first_page_params['modified__gt'] = last_poll

            if self.require_reviewed:
                first_page_params['custom_field_query'] = json.dumps(["Reviewed", "exact", True])

            url = f"{self.paperless_url}/api/documents/"
            all_documents = []

            while url:
                try:
                    response = requests.get(
                        url,
                        headers=self.headers,
                        params=first_page_params if url == f"{self.paperless_url}/api/documents/" else None,
                        timeout=30
                    )
                    response.raise_for_status()

                    data = response.json()
                    documents = data.get('results', [])
                    all_documents.extend(documents)

                    url = data.get('next')

                except Exception as e:
                    logger.error(f"Failed to fetch documents: {e}")
                    stats.errors.append(f"Fetch error: {str(e)}")
                    break

            if not all_documents:
                if retry_count == 0:
                    logger.info("No new documents to index")
                # Update last poll time to now
                new_poll_time = datetime.now(timezone.utc).isoformat()
                self._save_state(new_poll_time, failed_docs)
                stats.time_taken_seconds = time.time() - start_time
                return stats

            logger.info(f"Processing {len(all_documents)} new/updated documents...")

            # Sort by modified time for incremental progress
            all_documents.sort(key=lambda d: d.get('modified', ''))

            # Step 3: Process new/updated documents
            for doc_data in all_documents:
                doc_id = doc_data.get('id')
                doc_modified = doc_data.get('modified')

                try:
                    doc = PaperlessDocument(**doc_data)
                    chunks = self._process_document(doc)

                    if chunks:
                        self._index_chunks(chunks)
                        stats.documents_processed += 1
                        stats.chunks_created += len(chunks)

                        # Remove from failed list if it was there
                        if str(doc_id) in failed_docs:
                            del failed_docs[str(doc_id)]

                        # Save checkpoint after each successful document
                        if doc_modified:
                            self._save_state(doc_modified, failed_docs)

                except Exception as e:
                    logger.error(f"Failed to process document {doc_id}: {e}")
                    stats.errors.append(f"Doc {doc_id}: {str(e)}")

                    # Track failure for retry
                    self._record_failure(doc_id, str(e), failed_docs)
                    self._save_state(last_poll, failed_docs)

            stats.time_taken_seconds = time.time() - start_time
            logger.info(f"Indexing complete: {stats.documents_processed} docs, {stats.chunks_created} chunks in {stats.time_taken_seconds:.2f}s")

            return stats

        except Exception as e:
            logger.error(f"Polling failed: {e}")
            stats.errors.append(f"Poll error: {str(e)}")
            stats.time_taken_seconds = time.time() - start_time
            return stats

    def _process_document(self, doc: PaperlessDocument) -> List[DocumentChunk]:
        """Process a document into chunks"""
        try:
            # Extract metadata
            tags = self._extract_tags(doc.tags)
            correspondent_name = self._extract_correspondent(doc.correspondent)

            # Chunk the content
            content = doc.content or ""
            chunks = []

            if not content:
                # Create single metadata chunk
                chunks.append(DocumentChunk(
                    id=f"paperless_{doc.id}_chunk_0",
                    paperless_id=doc.id,
                    chunk_index=0,
                    chunk_type='metadata',
                    title=doc.title,
                    content=doc.title,
                    summary=None,
                    tags=tags,
                    correspondent_name=correspondent_name,
                    created_date=iso_to_timestamp(doc.created),
                    modified_date=iso_to_timestamp(doc.modified),
                    added_date=iso_to_timestamp(doc.added),
                    archive_serial_number=doc.archive_serial_number,
                    original_filename=doc.original_file_name,
                    total_chunks=1
                ))
            else:
                # Create metadata chunk + content chunks
                # Chunk 0: Metadata + summary
                summary = self._create_summary(content, doc.title, tags)
                metadata_text = f"Title: {doc.title}"
                if tags:
                    metadata_text += f" | Tags: {', '.join(tags)}"
                if correspondent_name:
                    metadata_text += f" | From: {correspondent_name}"
                if summary:
                    metadata_text += f" | Summary: {summary}"

                chunks.append(DocumentChunk(
                    id=f"paperless_{doc.id}_chunk_0",
                    paperless_id=doc.id,
                    chunk_index=0,
                    chunk_type='metadata',
                    title=doc.title,
                    content=metadata_text,
                    summary=summary,
                    tags=tags,
                    correspondent_name=correspondent_name,
                    created_date=iso_to_timestamp(doc.created),
                    modified_date=iso_to_timestamp(doc.modified),
                    added_date=iso_to_timestamp(doc.added),
                    archive_serial_number=doc.archive_serial_number,
                    original_filename=doc.original_file_name
                ))

                # Content chunks
                content_chunks = self._chunk_text(content)
                for i, chunk_text in enumerate(content_chunks, start=1):
                    chunks.append(DocumentChunk(
                        id=f"paperless_{doc.id}_chunk_{i}",
                        paperless_id=doc.id,
                        chunk_index=i,
                        chunk_type='content',
                        title=doc.title,
                        content=chunk_text,
                        summary=summary,
                        tags=tags,
                        correspondent_name=correspondent_name,
                        created_date=iso_to_timestamp(doc.created),
                        modified_date=iso_to_timestamp(doc.modified),
                        added_date=iso_to_timestamp(doc.added),
                        archive_serial_number=doc.archive_serial_number,
                        original_filename=doc.original_file_name
                    ))

            # Set total_chunks for all chunks
            total = len(chunks)
            for chunk in chunks:
                chunk.total_chunks = total

            return chunks

        except Exception as e:
            logger.error(f"Failed to process document {doc.id}: {e}")
            return []

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0
        min_chunk_size = 100  # Minimum meaningful chunk size

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Only include chunks that are large enough to be meaningful
            # Tiny final chunks (< min_chunk_size) create noise in vector search
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            elif len(chunks) > 0:
                # Merge tiny final chunk with previous chunk instead of discarding
                chunks[-1] = chunks[-1] + chunk
            else:
                # Edge case: first chunk is tiny (shouldn't happen with min_chunk_size=100)
                chunks.append(chunk)

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def _create_summary(self, content: str, title: str, tags: List[str]) -> Optional[str]:
        """Create a brief summary (just use first 200 chars for now)"""
        # Simple summary - just take first 200 characters
        # You could enhance this with LLM summarization
        if len(content) <= 200:
            return content
        return content[:200].strip() + "..."

    def _load_tags_cache(self):
        """Load all tags from Paperless and build ID -> name mapping"""
        try:
            all_tags = []
            page = 1

            while True:
                response = requests.get(
                    f"{self.paperless_url}/api/tags/?page={page}&page_size=100",
                    headers=self.headers,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()

                all_tags.extend(data.get('results', []))

                if not data.get('next'):
                    break
                page += 1

            # Build ID -> name mapping
            self.tags_cache = {tag['id']: tag['name'] for tag in all_tags}
            logger.info(f"Loaded {len(self.tags_cache)} tags into cache")

        except Exception as e:
            logger.error(f"Failed to load tags cache: {e}")
            self.tags_cache = {}

    def _load_correspondents_cache(self):
        """Load all correspondents from Paperless and build ID -> name mapping"""
        try:
            all_correspondents = []
            page = 1

            while True:
                response = requests.get(
                    f"{self.paperless_url}/api/correspondents/?page={page}&page_size=100",
                    headers=self.headers,
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()

                all_correspondents.extend(data.get('results', []))

                if not data.get('next'):
                    break
                page += 1

            # Build ID -> name mapping
            self.correspondents_cache = {corr['id']: corr['name'] for corr in all_correspondents}
            logger.info(f"Loaded {len(self.correspondents_cache)} correspondents into cache")

        except Exception as e:
            logger.error(f"Failed to load correspondents cache: {e}")
            self.correspondents_cache = {}

    def _extract_tags(self, tags: List[Any]) -> List[str]:
        """Extract tag names from mixed list (handles IDs, dicts, and strings)"""
        tag_names = []
        for tag in tags:
            if isinstance(tag, int):
                # Tag ID - lookup in cache
                tag_name = self.tags_cache.get(tag)
                if tag_name:
                    tag_names.append(tag_name)
            elif isinstance(tag, dict):
                tag_names.append(tag.get('name', ''))
            elif isinstance(tag, str):
                tag_names.append(tag)
        return [t for t in tag_names if t]

    def _extract_correspondent(self, correspondent: Any) -> Optional[str]:
        """Extract correspondent name (handles IDs, dicts, and strings)"""
        if not correspondent:
            return None
        if isinstance(correspondent, int):
            # Correspondent ID - lookup in cache
            return self.correspondents_cache.get(correspondent)
        if isinstance(correspondent, dict):
            return correspondent.get('name')
        elif isinstance(correspondent, str):
            return correspondent
        return None

    def _index_chunks(self, chunks: List[DocumentChunk]):
        """Generate embeddings and index chunks in Qdrant"""
        try:
            # Delete existing chunks for this document
            paperless_id = chunks[0].paperless_id
            from qdrant_client.models import Filter, FieldCondition, MatchValue

            self.qdrant_client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="paperless_id",
                            match=MatchValue(value=paperless_id)
                        )
                    ]
                )
            )

            # Generate embeddings for all chunks
            texts = [chunk.content for chunk in chunks]
            embeddings = self._generate_embeddings_batch(texts)

            # Create points
            points = []
            for chunk, embedding in zip(chunks, embeddings):
                if embedding:
                    point = PointStruct(
                        id=hash(chunk.id) & 0x7FFFFFFF,  # Convert to positive int
                        vector=embedding,
                        payload=chunk.dict()
                    )
                    points.append(point)

            # Upsert to Qdrant in batches to avoid payload size limits
            if points:
                batch_size = 20  # Reduced batch size for large documents with many chunks
                total_batches = (len(points) + batch_size - 1) // batch_size

                logger.info(f"Upserting {len(points)} points for document {paperless_id} in {total_batches} batch(es)")

                for i in range(0, len(points), batch_size):
                    batch = points[i:i + batch_size]
                    batch_num = (i // batch_size) + 1

                    logger.debug(f"Upserting batch {batch_num}/{total_batches} ({len(batch)} points)")

                    self.qdrant_client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )

                logger.info(f"Successfully indexed {len(points)} chunks for document {paperless_id}")
            else:
                raise Exception(f"No valid embeddings generated for document {paperless_id}")

        except Exception as e:
            logger.error(f"Failed to index chunks: {e}")
            raise

    def _generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts using provider with batching"""
        if not texts:
            return []

        try:
            # Use provider's batch embedding with RETRIEVAL_DOCUMENT task type
            embeddings = self.provider.get_embeddings_batch(
                texts=texts,
                task_type=EmbeddingTaskType.RETRIEVAL_DOCUMENT,
                batch_size=15  # Default batch size (Ollama uses 15, Google can use more)
            )

            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            # Raise exception to trigger retry
            raise Exception(f"Embedding generation failed: {str(e)}")

    def _should_retry(self, failure_info: Dict[str, Any]) -> bool:
        """Check if enough time has passed to retry a failed document"""
        retry_count = failure_info.get('retry_count', 0)

        # Give up after max retries
        if retry_count >= self.max_retries:
            return False

        # Get backoff time in minutes
        backoff_index = min(retry_count, len(self.retry_backoff_minutes) - 1)
        backoff_minutes = self.retry_backoff_minutes[backoff_index]

        # Check if enough time has passed
        last_attempt = failure_info.get('last_attempt')
        if not last_attempt:
            return True  # No last attempt, retry immediately

        try:
            last_attempt_dt = datetime.fromisoformat(last_attempt.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            elapsed_minutes = (now - last_attempt_dt).total_seconds() / 60

            should_retry = elapsed_minutes >= backoff_minutes
            if should_retry:
                logger.debug(f"Document ready for retry (attempt {retry_count + 1}, waited {elapsed_minutes:.1f}min)")
            return should_retry

        except Exception as e:
            logger.warning(f"Failed to parse last_attempt time: {e}")
            return True

    def _retry_document(self, doc_id: str, failure_info: Dict[str, Any], stats: IndexingStats) -> bool:
        """Retry indexing a failed document"""
        try:
            logger.info(f"Retrying document {doc_id} (attempt {failure_info.get('retry_count', 0) + 1})")

            # Fetch document from Paperless
            response = requests.get(
                f"{self.paperless_url}/api/documents/{doc_id}/",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()

            doc_data = response.json()
            doc = PaperlessDocument(**doc_data)
            chunks = self._process_document(doc)

            if chunks:
                self._index_chunks(chunks)
                stats.documents_processed += 1
                stats.chunks_created += len(chunks)
                logger.info(f"Successfully indexed document {doc_id} on retry")
                return True

        except Exception as e:
            logger.error(f"Retry failed for document {doc_id}: {e}")
            stats.errors.append(f"Retry Doc {doc_id}: {str(e)}")

            # Update failure info
            failure_info['retry_count'] = failure_info.get('retry_count', 0) + 1
            failure_info['last_attempt'] = datetime.now(timezone.utc).isoformat()
            failure_info['last_error'] = str(e)

            return False

    def _record_failure(self, doc_id: int, error: str, failed_docs: Dict[str, Dict]):
        """Record a document failure for retry"""
        doc_id_str = str(doc_id)
        now = datetime.now(timezone.utc).isoformat()

        if doc_id_str in failed_docs:
            # Existing failure, increment count
            failed_docs[doc_id_str]['retry_count'] = failed_docs[doc_id_str].get('retry_count', 0) + 1
            failed_docs[doc_id_str]['last_attempt'] = now
            failed_docs[doc_id_str]['last_error'] = error
        else:
            # New failure
            failed_docs[doc_id_str] = {
                'first_failed': now,
                'last_attempt': now,
                'retry_count': 0,
                'last_error': error
            }

        retry_count = failed_docs[doc_id_str]['retry_count']
        if retry_count >= self.max_retries:
            logger.warning(f"Document {doc_id} has failed {retry_count} times, giving up")
        else:
            backoff_index = min(retry_count, len(self.retry_backoff_minutes) - 1)
            next_retry = self.retry_backoff_minutes[backoff_index]
            logger.info(f"Document {doc_id} will retry in {next_retry} minutes")

    def _load_state(self) -> Dict[str, Any]:
        """Load state including last poll time and failed documents"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load state file: {e}")
        return {'last_poll': None, 'failed_documents': {}}

    def _save_state(self, last_poll: Optional[str], failed_docs: Dict[str, Dict]):
        """Save state including last poll time and failed documents"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'last_poll': last_poll,
                'failed_documents': failed_docs
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logger.debug(f"Saved state: last_poll={last_poll}, {len(failed_docs)} failed docs")
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")

    def _load_last_poll_time(self) -> Optional[str]:
        """Load last poll timestamp from state file"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    return data.get('last_poll')
            except Exception as e:
                logger.error(f"Failed to load state file: {e}")
        return None

    def _save_last_poll_time(self, timestamp: str):
        """Save last poll timestamp to state file"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w') as f:
                json.dump({'last_poll': timestamp}, f)
            logger.info(f"Saved last poll time: {timestamp}")
        except Exception as e:
            logger.error(f"Failed to save state file: {e}")
