# Paperless Qdrant Search

Semantic search for Paperless-ngx using RAG (Retrieval-Augmented Generation). Combines keyword matching with semantic understanding for better document retrieval.

## Purpose

Traditional keyword search in Paperless requires exact word matches. This implementation uses semantic search to find documents even when search terms differ from document text (e.g., searching "invoice" finds documents containing "bill").

## Features

- Natural language query expansion via LLM
- Hybrid search combining keyword and semantic vector search
- Multi-stage retrieval: initial retrieval (50 candidates) followed by reranking (top 10)
- Cross-encoder reranking for relevance scoring
- Provider support: Google Gemini (cloud) or Ollama (local)
- Web UI included

## Architecture

```
User Query → Query Processor (LLM expansion)
    ↓
Retriever (Hybrid search: keyword + semantic)
    ↓
Reranker (Cross-encoder scoring)
    ↓
Results (sorted by relevance with highlights)
```

Two-stage retrieval: initial search optimizes for recall, reranking optimizes for precision.

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Paperless-ngx instance
- Google AI API key (for Google provider) or Ollama (for local provider)

### Setup

1. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

   Edit `.env`:
   ```bash
   # Paperless connection
   PAPERLESS_URL=http://your-paperless-host:8000
   PAPERLESS_API_KEY=your_paperless_token

   # Google AI provider
   PROVIDER=google
   GOOGLE_API_KEY=your_google_api_key_here

   # Note: Google AI free tier may use data for training
   # For private documents, use paid tier or Ollama provider
   ```

2. **Start services**:
   ```bash
   docker-compose up -d
   ```

3. **Access search UI**:
   - Open http://localhost:7702
   - Indexing runs automatically every 5 minutes

### Ollama (Local Provider)

For local AI without cloud dependencies:

```bash
# Install Ollama models
ollama pull qwen3-embedding:8b
ollama pull ministral-3:latest

# Configure .env
PROVIDER=ollama
EMBEDDING_MODEL=qwen3-embedding:8b
LLM_MODEL=ministral-3:latest
RERANK_MODEL=ministral-3:latest
```

Note: Google AI may provide better performance but requires internet connectivity. Ollama runs locally without external API calls.

## Usage

### Web Interface

Navigate to http://localhost:7702 to search:

- "tax documents from 2023"
- "invoices from Acme Corp"
- "contracts about renewable energy"
- "letters tagged urgent"

Query expansion and reranking can be toggled in the UI.

## Configuration

Key environment variables in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `PROVIDER` | `ollama` | AI provider: `google` or `ollama` |
| `GOOGLE_API_KEY` | - | Google AI API key (if using Google) |
| `PAPERLESS_URL` | - | Your Paperless-ngx URL |
| `PAPERLESS_API_KEY` | - | Paperless API token |
| `POLL_INTERVAL_MINUTES` | `5` | How often to check for new documents |
| `ENABLE_QUERY_EXPANSION` | `true` | Use LLM for query understanding |
| `ENABLE_RERANKING` | `true` | Use reranker for precision |
| `HYBRID_ALPHA` | `0.5` | 0.0=keyword, 1.0=semantic |

See `.env.example` for all options.

## Troubleshooting

**No search results:**
- Check logs: `docker-compose logs search-web`
- Verify Paperless connection in `.env`
- Check indexing: `docker-compose logs search-indexer`

**Poor quality results:**
- Try different provider: `PROVIDER=google` or `PROVIDER=ollama`
- Enable both expansion and reranking
- Increase candidates: `RETRIEVAL_TOP_K=100`
- Adjust hybrid balance: `HYBRID_ALPHA=0.7` for more semantic, `0.3` for more keyword

**Switching providers:**
- Delete Qdrant data: `rm -rf ./data/qdrant/*`
- Restart: `docker-compose restart`
- Note: Vector dimensions differ between providers, requiring re-indexing

## Architecture Notes

Bi-encoder embeddings provide fast search but limited precision. Multi-stage retrieval addresses this:

1. Bi-encoder retrieves 50 candidates (optimized for recall)
2. Cross-encoder reranks top 10 (optimized for precision)

This balances speed and accuracy.

## License

MIT - See [LICENSE](LICENSE) for details

## Credits

- [Qdrant](https://qdrant.tech/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Google Gemini](https://ai.google.dev/) - Cloud AI API
- [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) - Document management
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
