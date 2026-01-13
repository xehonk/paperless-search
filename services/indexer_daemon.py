"""
Standalone indexer daemon for Paperless documents
Runs independently from the web UI
"""

import os
import sys
import time
import signal
import logging
import schedule
from indexer import DocumentIndexer
from retriever import Retriever

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment
PAPERLESS_URL = os.getenv('PAPERLESS_URL', 'http://paperless:8000')
PAPERLESS_API_KEY = os.getenv('PAPERLESS_API_KEY')

# Validate required configuration
if not PAPERLESS_API_KEY:
    logger.error("PAPERLESS_API_KEY environment variable is required")
    sys.exit(1)

QDRANT_URL = os.getenv('QDRANT_URL', 'http://qdrant:6333')
QDRANT_COLLECTION = os.getenv('QDRANT_COLLECTION', 'documents')
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://host.docker.internal:11434')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'nomic-embed-text')
POLL_INTERVAL_MINUTES = int(os.getenv('POLL_INTERVAL_MINUTES', '5'))
REQUIRE_REVIEWED = os.getenv('REQUIRE_REVIEWED', 'true').lower() == 'true'
CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '512'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '50'))

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_requested = True


def run_indexing_poll(indexer: DocumentIndexer):
    """Run indexing poll (called by scheduler)"""
    try:
        logger.info("Starting scheduled indexing poll...")
        stats = indexer.poll_and_index()
        logger.info(
            f"Poll complete: {stats.documents_processed} docs, "
            f"{stats.chunks_created} chunks in {stats.time_taken_seconds:.2f}s"
        )
        if stats.errors:
            logger.warning(f"Encountered {len(stats.errors)} errors during indexing")
    except Exception as e:
        logger.error(f"Indexing poll failed: {e}")


def main():
    """Main indexer daemon"""
    logger.info("=" * 60)
    logger.info("Paperless Qdrant Indexer Daemon starting...")
    logger.info(f"Paperless URL: {PAPERLESS_URL}")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Ollama URL: {OLLAMA_URL}")
    logger.info(f"Embedding model: {EMBEDDING_MODEL}")
    logger.info(f"Poll interval: {POLL_INTERVAL_MINUTES} minutes")
    logger.info(f"Require reviewed: {REQUIRE_REVIEWED}")
    logger.info("=" * 60)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Initialize indexer
    try:
        logger.info("Initializing indexer...")

        # Create provider
        from providers import create_provider
        provider = create_provider()

        indexer = DocumentIndexer(
            paperless_url=PAPERLESS_URL,
            paperless_api_key=PAPERLESS_API_KEY,
            qdrant_url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
            provider=provider,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            require_reviewed=REQUIRE_REVIEWED
        )

        # Initialize Qdrant collection
        logger.info("Checking Qdrant collection...")
        retriever = Retriever(
            qdrant_url=QDRANT_URL,
            collection_name=QDRANT_COLLECTION,
            provider=provider
        )
        vector_size = retriever.get_embedding_dimensions()
        indexer.initialize_collection(vector_size)

        logger.info("Indexer initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize indexer: {e}")
        sys.exit(1)

    # Schedule indexing polls
    logger.info(f"Scheduling indexing every {POLL_INTERVAL_MINUTES} minutes")
    schedule.every(POLL_INTERVAL_MINUTES).minutes.do(
        lambda: run_indexing_poll(indexer)
    )

    # Run initial indexing immediately
    logger.info("Running initial indexing poll...")
    run_indexing_poll(indexer)

    # Main loop
    logger.info("Indexer daemon ready, entering main loop")
    while not shutdown_requested:
        schedule.run_pending()
        time.sleep(10)  # Check every 10 seconds

    logger.info("Indexer daemon shutting down")
    sys.exit(0)


if __name__ == "__main__":
    main()
