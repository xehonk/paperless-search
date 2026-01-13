#!/bin/bash
# Rebuild and restart the search-web service
# Use this after making code changes to search/query/reranking logic

echo "Stopping search-web service..."
docker compose stop search-web

echo "Rebuilding and restarting search-web service..."
docker compose up -d --build search-web

echo ""
echo "✅ Done! Check logs with:"
echo "   docker compose logs -f search-web"
