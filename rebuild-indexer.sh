#!/bin/bash
# Rebuild and restart the search-indexer service
# Use this after making code changes to indexing logic

# Detect docker compose command (v2 plugin vs v1 standalone)
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif docker-compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "❌ Error: Neither 'docker compose' nor 'docker-compose' found"
    exit 1
fi

echo "Using: $DOCKER_COMPOSE"
echo "Stopping search-indexer service..."
$DOCKER_COMPOSE stop search-indexer

echo "Rebuilding and restarting search-indexer service..."
$DOCKER_COMPOSE up -d --build search-indexer

echo ""
echo "✅ Done! Check logs with:"
echo "   $DOCKER_COMPOSE logs -f search-indexer"
