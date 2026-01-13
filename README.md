# Paperless Qdrant Search

**Intelligent semantic search for Paperless-ngx powered by Qdrant vector database and AI**

A production-ready RAG (Retrieval-Augmented Generation) search system that brings Google-like search capabilities to your Paperless documents. Goes beyond simple keyword matching with natural language understanding, multi-stage retrieval, and intelligent reranking.

## 💡 Why This Exists

Ever tried searching for a document in Paperless but couldn't find it because you used slightly different words? You search for "invoice" but the document says "bill". You search for "contract" but it's labeled "agreement".

**Traditional keyword search fails when your search terms don't exactly match the document text.**

This project solves that problem with semantic search - it understands that "invoice" and "bill" mean similar things, that "contract" and "agreement" are related. Now you can find your documents even when you don't remember the exact wording.

## ✨ Features

- 🧠 **Natural Language Queries** - "tax documents from last year" automatically expands and filters
- 🔍 **Hybrid Search** - Combines keyword matching with semantic understanding
- 📊 **Multi-Stage Retrieval** - Wide retrieval (50 candidates) → precise reranking (top 10)
- 🎯 **Cross-Encoder Reranking** - Deep relevance scoring for best results
- ☁️ **Dual Provider Support** - Cloud AI (Google Gemini - faster) or local (Ollama - private)
- 🚀 **Fast & Private** - Runs entirely on your infrastructure
- 💎 **Modern UI** - Clean, Google-inspired search interface

## 🏗️ Architecture

```
User Query → Query Processor (LLM expansion)
    ↓
Retriever (Hybrid search: keyword + semantic)
    ↓
Reranker (Cross-encoder scoring)
    ↓
Results (sorted by relevance with highlights)
```

This follows modern search engine architecture: cast a wide net for recall, then rerank for precision.

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Paperless-ngx instance
- **Google AI API key** (recommended - faster, better results) OR Ollama (local, private)

### Setup

1. **Get Google AI API key** (recommended):
   ```bash
   # Available at https://ai.google.dev/
   # Sign up and generate an API key

   # ⚠️ WARNING: Free tier may use your data for training
   # For private documents, use paid tier or Ollama (local)
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   ```

   Edit `.env`:
   ```bash
   # Paperless connection
   PAPERLESS_URL=http://your-paperless-host:8000
   PAPERLESS_API_KEY=your_paperless_token

   # Use Google AI (recommended - faster and better results)
   PROVIDER=google
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. **Start services**:
   ```bash
   docker-compose up -d
   ```

4. **Access search UI**:
   - Open http://localhost:7702
   - Indexing starts automatically
   - New documents are indexed every 5 minutes

### Alternative: Ollama (Local, Fully Private)

If you prefer local AI without cloud dependencies:

```bash
# Install Ollama models
ollama pull qwen3-embedding:8b      # Best embedding model
ollama pull ministral-3:latest      # Best LLM for queries

# Configure .env
PROVIDER=ollama
EMBEDDING_MODEL=qwen3-embedding:8b
LLM_MODEL=ministral-3:latest
RERANK_MODEL=ministral-3:latest
```

**Performance Note**: Google AI provides slightly better search results and is **significantly faster** than local Ollama, but requires internet connectivity and sends queries to Google's API.

**⚠️ Privacy Warning**: Google AI's **free tier may use your data for training**. For private documents, use Google AI's paid tier (with data protections) or Ollama (fully local and private).

## 📖 Usage

### Web Interface

Navigate to http://localhost:7702 and search naturally:

- "tax documents from 2023"
- "invoices from Acme Corp"
- "contracts about renewable energy"
- "letters tagged urgent"

Toggle query expansion and reranking in the UI for different speed/quality tradeoffs.

### API Integration

```python
import requests

response = requests.post('http://localhost:7702/api/search', json={
    'query': 'tax documents from last year',
    'enable_query_expansion': true,
    'enable_reranking': true,
    'top_k': 10
})

for result in response.json()['results']:
    print(f"{result['title']} - Score: {result['rerank_score']:.2f}")
    print(f"URL: http://paperless/documents/{result['paperless_id']}")
```

## ⚙️ Configuration

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

## 🔧 Troubleshooting

**No search results?**
- Check logs: `docker-compose logs search-web`
- Verify Paperless connection in `.env`
- Check indexing: `docker-compose logs search-indexer`

**Slow search?**
- Disable reranking: `ENABLE_RERANKING=false`
- Reduce candidates: `RETRIEVAL_TOP_K=20`

**Poor quality results?**
- Switch to Google AI: `PROVIDER=google` (better results, much faster)
- Or try better Ollama models: `qwen3-embedding:8b` + `ministral-3:latest`
- Enable both expansion and reranking
- Increase candidates: `RETRIEVAL_TOP_K=100`
- Adjust hybrid balance: `HYBRID_ALPHA=0.7` or `0.8` for more semantic

**Switching providers?**
- Delete Qdrant data: `rm -rf ./data/qdrant/*`
- Restart: `docker-compose restart`
- Vector dimensions differ between providers (requires re-indexing)

## 📊 Why This Architecture?

Traditional search engines use bi-encoder embeddings which are fast but imprecise. We use **multi-stage retrieval**:

1. **Bi-encoder** (fast) retrieves 50 candidates with high recall
2. **Cross-encoder** (slow, accurate) reranks top 10 with high precision

This is how Google, Bing, and modern production search systems work - best of both speed and accuracy.

## 📝 License

MIT - See [LICENSE](LICENSE) for details

## 🙏 Credits

- [Qdrant](https://qdrant.tech/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Google Gemini](https://ai.google.dev/) - Cloud AI API
- [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx) - Document management
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
