# Prompt Templates

This directory contains all LLM prompt templates used by the search system. You can edit these files directly to customize how the system behaves.

## Available Prompts

### 1. `query_expansion.txt`
**Used by:** Query Processor
**Purpose:** Expands user search queries by adding synonyms and related terms

**Variables:**
- `{query}` - The user's original search query
- `{language}` - The default language (from `DEFAULT_QUERY_LANGUAGE` env var)

**Example:**
```
Query: "tax documents"
→ Expanded: "tax documents taxation financial records IRS forms"
```

### 2. `reranker_llm_scoring.txt`
**Used by:** Reranker (LLM-based scoring mode)
**Purpose:** Scores document relevance on a scale of 0-10

**Variables:**
- `{query}` - The user's search query
- `{document}` - Excerpt from the document (up to 400 chars)

**When used:**
- With Google provider (always)
- With Ollama when no dedicated reranker model is available

### 3. `reranker_dedicated.txt`
**Used by:** Reranker (dedicated reranker model mode)
**Purpose:** Yes/no classification for document relevance

**Variables:**
- `{instruction}` - Fixed instruction about the task
- `{query}` - The user's search query
- `{document}` - Excerpt from the document (up to 512 chars)

**When used:**
- With Ollama when a dedicated reranker model is available (e.g., bge-reranker-v2-m3)
- Uses logprobs to extract probability scores

### 4. `explain_relevance.txt`
**Used by:** Reranker (explain_score method)
**Purpose:** Generates human-readable explanations for search result relevance

**Variables:**
- `{query}` - The user's search query
- `{document}` - Excerpt from the document (up to 400 chars)

**When used:**
- When user clicks "Explain" button on a search result
- On-demand, not during batch reranking

**Output format:**
```
Score: 8
Reason: This document directly discusses tax returns and contains relevant financial information.
```

## How to Edit

1. **Edit the prompt file** directly in this directory
2. **Restart the service** to apply changes:
   ```bash
   docker compose restart search-web
   ```

**Note:** For indexer service changes, restart:
```bash
docker compose restart search-indexer
```

## Template Syntax

Prompts use Python's `.format()` syntax for variable substitution:
- `{variable_name}` - Will be replaced with actual values
- `{{` and `}}` - Use double braces to include literal curly braces in output

## Tips

### Query Expansion
- Keep instructions concise - longer prompts = higher latency
- Focus on semantic expansion, not filtering (filtering is disabled)
- Request structured output (JSON) for reliable parsing

### Reranker Scoring
- Be specific about the scoring scale (0-10)
- Request ONLY a number to minimize response length
- Shorter prompts = faster reranking

### Testing Changes
After editing, test with:
```bash
curl -X POST http://localhost:7702/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 5}'
```

Check logs to see the prompts in action:
```bash
docker compose logs -f search-web
```

## Environment Context

Prompts are loaded at **startup time** from these files. Changes require a service restart.

**Provider differences:**
- **Google**: Uses `reranker_llm_scoring.txt` (no dedicated reranker)
- **Ollama**: Uses `reranker_dedicated.txt` if reranker model available, otherwise `reranker_llm_scoring.txt`

## Troubleshooting

### Syntax Errors
If you see errors like `KeyError: 'variable'`, check that:
- Variable names match exactly (case-sensitive)
- All variables in the prompt file are provided by the code

### Empty Responses
If queries return nothing after editing:
- Check that JSON format is valid in `query_expansion.txt`
- Verify the prompt asks for structured output
- Review logs: `docker compose logs search-web`

### Performance Impact
- Longer prompts = higher latency
- For query expansion: ~100-200ms per search
- For reranking: ~20-50ms per document scored
