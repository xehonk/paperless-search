#!/bin/bash
# Rebuild and restart the search-indexer service
# Use this after making code changes to indexing logic

echo "Stopping search-indexer service..."
docker compose stop search-indexer

echo "Rebuilding and restarting search-indexer service..."
docker compose up -d --build search-indexer

echo ""
echo "✅ Done! Check logs with:"
echo "   docker compose logs -f search-indexer"
