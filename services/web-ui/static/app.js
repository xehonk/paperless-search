const searchButton = document.getElementById('searchButton');
const contentDiv = document.getElementById('content');
const queryExpansionCheckbox = document.getElementById('queryExpansion');
const rerankingCheckbox = document.getElementById('reranking');
const advancedToggle = document.getElementById('advancedToggle');
const advancedOptions = document.getElementById('advancedOptions');
const toggleIcon = document.getElementById('toggleIcon');

// Slider elements
const recencyWeight = document.getElementById('recencyWeight');
const recencyWeightValue = document.getElementById('recencyWeightValue');
const recencyDecay = document.getElementById('recencyDecay');
const recencyDecayValue = document.getElementById('recencyDecayValue');
const hybridAlpha = document.getElementById('hybridAlpha');
const hybridAlphaValue = document.getElementById('hybridAlphaValue');
const rerankInfluence = document.getElementById('rerankInfluence');
const rerankInfluenceValue = document.getElementById('rerankInfluenceValue');
const retrievalTopK = document.getElementById('retrievalTopK');
const retrievalTopKValue = document.getElementById('retrievalTopKValue');
const topK = document.getElementById('topK');
const topKValue = document.getElementById('topKValue');

let isSearching = false;
let paperlessBaseUrl = 'http://localhost:8000'; // Default fallback

// Fetch configuration on page load
async function loadConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        paperlessBaseUrl = config.paperless_url || paperlessBaseUrl;
        console.log('Loaded Paperless URL:', paperlessBaseUrl);
    } catch (error) {
        console.error('Failed to load config:', error);
    }
}

// Load config when page loads
loadConfig();

// Populate year dropdowns
function populateYearDropdowns() {
    const currentYear = new Date().getFullYear();
    const yearFrom = document.getElementById('yearFrom');
    const yearTo = document.getElementById('yearTo');

    for (let year = currentYear + 2; year >= currentYear - 20; year--) {
        const optionFrom = document.createElement('option');
        optionFrom.value = year;
        optionFrom.textContent = year;
        yearFrom.appendChild(optionFrom);

        const optionTo = document.createElement('option');
        optionTo.value = year;
        optionTo.textContent = year;
        yearTo.appendChild(optionTo);
    }
}

populateYearDropdowns();

// Collapsible filter panels
const tagsToggle = document.getElementById('tagsToggle');
const correspondentsToggle = document.getElementById('correspondentsToggle');
const tagsPanel = document.getElementById('tagsPanel');
const correspondentsPanel = document.getElementById('correspondentsPanel');

tagsToggle.addEventListener('click', () => {
    tagsPanel.classList.toggle('collapsed');
    tagsToggle.classList.toggle('active');

    // Close other panel
    correspondentsPanel.classList.add('collapsed');
    correspondentsToggle.classList.remove('active');
});

correspondentsToggle.addEventListener('click', () => {
    correspondentsPanel.classList.toggle('collapsed');
    correspondentsToggle.classList.toggle('active');

    // Close other panel
    tagsPanel.classList.add('collapsed');
    tagsToggle.classList.remove('active');
});

// Tag logic toggle (All/Any)
const logicButtons = document.querySelectorAll('.logic-btn');
let tagLogic = 'any'; // Default to 'any' (OR logic)

logicButtons.forEach(button => {
    button.addEventListener('click', () => {
        tagLogic = button.getAttribute('data-logic');
        logicButtons.forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    });
});

// Load tags and correspondents
const clearFiltersBtn = document.getElementById('clearFilters');
const yearFrom = document.getElementById('yearFrom');
const yearTo = document.getElementById('yearTo');
const tagsList = document.getElementById('tagsList');
const correspondentsList = document.getElementById('correspondentsList');
const tagSearch = document.getElementById('tagSearch');
const correspondentSearch = document.getElementById('correspondentSearch');

let allTags = [];
let allCorrespondents = [];
let selectedTags = new Set();
let selectedCorrespondent = null;

async function loadTags() {
    try {
        const response = await fetch('/api/paperless/tags');
        const data = await response.json();
        allTags = data.tags;
        renderTags(allTags);
    } catch (error) {
        console.error('Failed to load tags:', error);
        tagsList.innerHTML = '<div class="loading">Failed to load tags</div>';
    }
}

function renderTags(tags) {
    tagsList.innerHTML = '';

    if (tags.length === 0) {
        tagsList.innerHTML = '<div class="loading">No tags found</div>';
        return;
    }

    tags.forEach(tag => {
        const item = document.createElement('div');
        item.className = 'filter-item-checkbox';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `tag-${tag.id}`;
        checkbox.value = tag.name;
        checkbox.checked = selectedTags.has(tag.name);

        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                selectedTags.add(tag.name);
            } else {
                selectedTags.delete(tag.name);
            }
        });

        const label = document.createElement('label');
        label.htmlFor = `tag-${tag.id}`;

        const badge = document.createElement('span');
        badge.className = 'tag-badge';
        badge.textContent = tag.name;
        badge.style.backgroundColor = tag.colour;
        badge.style.color = getContrastColor(tag.colour);

        const count = document.createElement('span');
        count.className = 'doc-count';
        count.textContent = tag.document_count;

        label.appendChild(badge);
        label.appendChild(count);
        item.appendChild(checkbox);
        item.appendChild(label);
        tagsList.appendChild(item);
    });
}

// Calculate contrasting text color for tag badges
function getContrastColor(hexColor) {
    // Convert hex to RGB
    const r = parseInt(hexColor.slice(1, 3), 16);
    const g = parseInt(hexColor.slice(3, 5), 16);
    const b = parseInt(hexColor.slice(5, 7), 16);

    // Calculate relative luminance
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;

    return luminance > 0.5 ? '#000000' : '#ffffff';
}

// Tag search filter
tagSearch.addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase();
    const filtered = allTags.filter(tag =>
        tag.name.toLowerCase().includes(searchTerm)
    );
    renderTags(filtered);
});

async function loadCorrespondents() {
    try {
        const response = await fetch('/api/paperless/correspondents');
        const data = await response.json();
        allCorrespondents = data.correspondents;
        renderCorrespondents(allCorrespondents);
    } catch (error) {
        console.error('Failed to load correspondents:', error);
        correspondentsList.innerHTML = '<div class="loading">Failed to load correspondents</div>';
    }
}

function renderCorrespondents(correspondents) {
    correspondentsList.innerHTML = '';

    if (correspondents.length === 0) {
        correspondentsList.innerHTML = '<div class="loading">No correspondents found</div>';
        return;
    }

    // Add correspondent options
    correspondents.forEach(corr => {
        const item = document.createElement('div');
        item.className = 'filter-item-checkbox';

        const checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.id = `correspondent-${corr.id}`;
        checkbox.value = corr.name;
        checkbox.checked = selectedCorrespondent === corr.name;

        checkbox.addEventListener('change', (e) => {
            // Prevent default to handle manually
            const wasChecked = e.target.checked;

            // Only one correspondent can be selected at a time - uncheck all first
            document.querySelectorAll('#correspondentsList input[type="checkbox"]').forEach(cb => {
                cb.checked = false;
            });

            if (wasChecked) {
                selectedCorrespondent = corr.name;
                e.target.checked = true;
                console.log('Selected correspondent:', selectedCorrespondent);
            } else {
                selectedCorrespondent = null;
                console.log('Cleared correspondent selection');
            }
        });

        const label = document.createElement('label');
        label.htmlFor = `correspondent-${corr.id}`;
        label.innerHTML = `
            <span>${corr.name}</span>
            <span class="doc-count">${corr.document_count}</span>
        `;

        item.appendChild(checkbox);
        item.appendChild(label);
        correspondentsList.appendChild(item);
    });
}

// Correspondent search filter
correspondentSearch.addEventListener('input', (e) => {
    const searchTerm = e.target.value.toLowerCase();
    const filtered = allCorrespondents.filter(corr =>
        corr.name.toLowerCase().includes(searchTerm)
    );
    // Note: renderCorrespondents will preserve selectedCorrespondent state
    renderCorrespondents(filtered);
});

clearFiltersBtn.addEventListener('click', () => {
    yearFrom.value = '';
    yearTo.value = '';
    selectedTags.clear();
    selectedCorrespondent = null;
    renderTags(allTags);
    renderCorrespondents(allCorrespondents);
});

// Load filters on page load
loadTags();
loadCorrespondents();

// Toggle advanced options
advancedToggle.addEventListener('click', () => {
    advancedOptions.classList.toggle('show');
    toggleIcon.textContent = advancedOptions.classList.contains('show') ? '▼' : '▶';
});

// Update slider values
recencyWeight.addEventListener('input', (e) => {
    recencyWeightValue.textContent = e.target.value;
});
recencyDecay.addEventListener('input', (e) => {
    recencyDecayValue.textContent = e.target.value;
});
hybridAlpha.addEventListener('input', (e) => {
    hybridAlphaValue.textContent = e.target.value;
});
rerankInfluence.addEventListener('input', (e) => {
    rerankInfluenceValue.textContent = e.target.value;
});
retrievalTopK.addEventListener('input', (e) => {
    retrievalTopKValue.textContent = e.target.value;
});
topK.addEventListener('input', (e) => {
    topKValue.textContent = e.target.value;
});

// Search on button click
searchButton.addEventListener('click', performSearch);

// Search on Enter key
searchInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

function buildFilters() {
    const filters = {};

    // Year range filter (convert to full date ranges)
    if (yearFrom.value || yearTo.value) {
        filters.created_date = {};
        if (yearFrom.value) {
            filters.created_date.gte = `${yearFrom.value}-01-01`;
        }
        if (yearTo.value) {
            filters.created_date.lte = `${yearTo.value}-12-31`;
        }
    }

    // Tags filter (multiple selection with All/Any logic)
    if (selectedTags.size > 0) {
        filters.tags = Array.from(selectedTags);
        // Note: tagLogic ('all' vs 'any') can be used by backend if supported
        // For now, backend treats tags as OR logic (any)
    }

    // Correspondent filter
    if (selectedCorrespondent !== null) {
        // Empty string means "not assigned"
        filters.correspondent_name = selectedCorrespondent;
    }

    console.log('Built filters:', filters);
    console.log('Selected tags:', selectedTags);
    console.log('Selected correspondent:', selectedCorrespondent);

    return Object.keys(filters).length > 0 ? filters : null;
}

async function performSearch() {
    const query = searchInput.value.trim();

    if (!query || isSearching) {
        return;
    }

    isSearching = true;
    searchButton.disabled = true;
    searchButton.textContent = 'Searching...';

    // Show progress UI (all stages shown as pending/waiting initially)
    contentDiv.innerHTML = `
        <div class="progress-container">
            <div class="progress-stage" id="progress-query">
                <div class="progress-stage-header">
                    <span class="progress-stage-name">
                        <span class="progress-icon">⏳</span>
                        Query Expansion
                    </span>
                    <span class="progress-stage-status">Waiting...</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="progress-stage" id="progress-retrieval">
                <div class="progress-stage-header">
                    <span class="progress-stage-name">
                        <span class="progress-icon">🔍</span>
                        Retrieval
                    </span>
                    <span class="progress-stage-status">Waiting...</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: 0%"></div>
                </div>
            </div>
            <div class="progress-stage" id="progress-reranking">
                <div class="progress-stage-header">
                    <span class="progress-stage-name">
                        <span class="progress-icon">🎯</span>
                        Reranking
                    </span>
                    <span class="progress-stage-status">Waiting...</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-bar-fill" style="width: 0%"></div>
                </div>
            </div>
        </div>
    `;

    try {
        const filters = buildFilters();
        const searchPayload = {
            query: query,
            enable_query_expansion: queryExpansionCheckbox.checked,
            enable_reranking: rerankingCheckbox.checked,
            top_k: parseInt(topK.value),
            retrieval_top_k: parseInt(retrievalTopK.value),
            hybrid_alpha: parseFloat(hybridAlpha.value),
            rerank_influence: parseFloat(rerankInfluence.value),
            recency_weight: parseFloat(recencyWeight.value),
            recency_decay_days: parseInt(recencyDecay.value)
        };

        if (filters) {
            searchPayload.filters = filters;
            console.log('Applying filters:', filters);
        }

        const response = await fetch('/api/search/stream', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(searchPayload)
        });

        if (!response.ok) {
            throw new Error('Search failed');
        }

        // Process SSE stream
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    console.log('SSE received:', data.type, Date.now());
                    handleProgressUpdate(data);
                }
            }
        }

    } catch (error) {
        contentDiv.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">⚠️</div>
                <h3>Search failed</h3>
                <p>${error.message}</p>
            </div>
        `;
    } finally {
        isSearching = false;
        searchButton.disabled = false;
        searchButton.textContent = 'Search';
    }
}

// Store results being progressively updated
let progressiveResults = [];
let resultsContainer = null;
let renderTimeout = null;

function handleProgressUpdate(data) {
    if (data.type === 'progress') {
        console.log('Progress update:', data.stage, data.message, 'done:', data.done, 'current:', data.current, 'total:', data.total);
        // Use requestAnimationFrame to ensure the browser paints updates
        requestAnimationFrame(() => {
            updateProgressStage(data.stage, data.message, data.current, data.total, data.done);
        });
    } else if (data.type === 'initial_results') {
        // Store all results and display top_k
        console.log('Received initial_results:', data.data.results.length, 'results');
        currentQuery = data.data.query;
        progressiveResults = data.data.results || [];
        currentSearchResults = [...progressiveResults];
        renderResults(data.data.query, data.data.expanded_query, data.data.retrieval_time_ms);
        console.log('Rendered initial results');
    } else if (data.type === 'result_update') {
        // Update individual result and re-render
        updateResult(data);
    } else if (data.type === 'highlights') {
        // Add highlights to results
        const highlightsMap = data.highlights;
        for (const [paperlessId, highlights] of Object.entries(highlightsMap)) {
            const result = progressiveResults.find(r => r.paperless_id == paperlessId);
            if (result) {
                result.highlights = highlights;
            }
        }
        // Re-render to show highlights
        renderResults(currentQuery, null, null);
    } else if (data.type === 'done') {
        // Hide progress bars
        const progressContainer = document.querySelector('.progress-container');
        if (progressContainer) {
            progressContainer.style.display = 'none';
        }
    } else if (data.type === 'error') {
        throw new Error(data.message);
    }
}

function updateResult(data) {
    // Find and update the result
    const result = progressiveResults.find(r => r.paperless_id === data.paperless_id);
    if (!result) return;

    // Update scores
    result.rerank_score = data.rerank_score;
    result.final_score = data.final_score;
    result.recency_boost = data.recency_boost;

    // Update currentSearchResults too
    const currentResult = currentSearchResults.find(r => r.paperless_id === data.paperless_id);
    if (currentResult) {
        currentResult.rerank_score = data.rerank_score;
        currentResult.final_score = data.final_score;
        currentResult.recency_boost = data.recency_boost;
    }

    // Re-render immediately with updated scores
    renderResults(currentQuery, null, null);
}

function renderResults(query, expandedQuery, retrievalTimeMs) {
    console.log('renderResults called, results count:', progressiveResults.length);
    const renderStart = Date.now();

    // Sort by final_score
    progressiveResults.sort((a, b) => (b.final_score || b.score) - (a.final_score || a.score));

    // Get top_k
    const currentTopK = parseInt(document.getElementById('topK').value);
    const displayResults = progressiveResults.slice(0, currentTopK);

    // Find or create results container
    if (!resultsContainer) {
        const existingContainer = document.querySelector('.results');
        if (existingContainer) {
            resultsContainer = existingContainer;
        }
    }

    // Build HTML for results
    let html = '';

    // Add stats and expanded query if this is initial render
    if (retrievalTimeMs !== null && retrievalTimeMs !== undefined) {
        html += `<div class="stats">About ${progressiveResults.length} results (${(retrievalTimeMs / 1000).toFixed(2)} seconds)</div>`;

        if (expandedQuery && expandedQuery !== query) {
            html += `
                <div class="expanded-query">
                    <strong>Expanded query:</strong> ${escapeHtml(expandedQuery)}
                </div>
            `;
        }
    }

    html += '<div class="results">';

    displayResults.forEach((result, index) => {
        const paperlessUrl = getPaperlessUrl(result.paperless_id);
        const downloadUrl = getDownloadUrl(result.paperless_id);
        const finalScore = result.final_score || result.score;

        html += `
            <div class="result" data-paperless-id="${result.paperless_id}">
                <a href="${paperlessUrl}" target="_blank" class="result-title">
                    #${index + 1} - ${escapeHtml(result.title)}
                </a>

                <div class="result-meta">
                    ${result.correspondent_name ? `<span class="result-meta-item">👤 ${escapeHtml(result.correspondent_name)}</span>` : ''}
                    ${result.created_date ? `<span class="result-meta-item">📅 ${formatDate(result.created_date)}</span>` : ''}
                </div>

                <div class="ranking-info">
                    <div class="ranking-item">
                        <span class="ranking-label">Final Score:</span>
                        <span class="final-score">${finalScore.toFixed(4)}</span>
                    </div>
                    <div class="ranking-item">
                        <span class="ranking-label">Initial:</span>
                        <span>${result.score.toFixed(4)}</span>
                    </div>
                    ${result.rerank_score != null ? `
                        <div class="ranking-item">
                            <span class="ranking-label">Rerank:</span>
                            <span>
                                <span class="rerank-score">${result.rerank_score.toFixed(2)}/10</span>
                                <button class="explain-btn" onclick="explainScore('${result.paperless_id}', event)" title="Get LLM explanation">
                                    🤔 Explain
                                </button>
                            </span>
                        </div>
                        <div class="ranking-item explanation-container" id="explanation-${result.paperless_id}" style="grid-column: 1 / -1; display: none;">
                            <span class="ranking-label">Reason:</span>
                            <span class="explanation-text" style="font-style: italic; color: #666;"></span>
                        </div>
                    ` : ''}
                    ${result.recency_boost && result.recency_boost > 0 ? `
                        <div class="ranking-item">
                            <span class="ranking-label">Recency Boost:</span>
                            <span class="recency-boost">+${(result.recency_boost * 100).toFixed(1)}%</span>
                        </div>
                    ` : ''}
                    <div class="ranking-item">
                        <span class="ranking-label">Chunk:</span>
                        <span>${result.chunk_index}</span>
                    </div>
                </div>

                <div class="actions" style="margin-top: 12px;">
                    <a href="${paperlessUrl}" target="_blank" class="action-button primary">📄 View in Paperless</a>
                    <a href="${downloadUrl}" target="_blank" class="action-button">⬇️ Download</a>
                </div>

                ${result.tags && result.tags.length > 0 ? `
                    <div class="tags">
                        ${result.tags.map(tag => `<span class="tag">${escapeHtml(tag)}</span>`).join('')}
                    </div>
                ` : ''}

                ${result.highlights && result.highlights.length > 0 ? `
                    <div class="result-content">
                        ${result.highlights.map(h => `<div class="highlight">...${h}...</div>`).join('')}
                    </div>
                ` : ''}

                <div class="metadata-toggle" onclick="toggleMetadata(${result.paperless_id})">
                    <span>📋 Document Metadata</span>
                    <span class="metadata-toggle-icon">▼</span>
                </div>
                <div class="metadata-content" id="metadata-${result.paperless_id}">
                    <div class="metadata-grid">
                        <span class="metadata-label">Document ID:</span>
                        <span class="metadata-value">${result.paperless_id}</span>
                        <span class="metadata-label">Final Score:</span>
                        <span class="metadata-value">${finalScore.toFixed(6)}</span>
                    </div>
                </div>
            </div>
        `;
    });

    html += '</div>';

    // Insert or update results
    const progressContainer = contentDiv.querySelector('.progress-container');
    if (progressContainer) {
        // Remove all siblings after progress container (old results)
        let sibling = progressContainer.nextElementSibling;
        while (sibling) {
            const next = sibling.nextElementSibling;
            sibling.remove();
            sibling = next;
        }
        // Insert after progress container
        progressContainer.insertAdjacentHTML('afterend', html);
    } else {
        contentDiv.innerHTML = html;
    }

    // Update resultsContainer reference
    resultsContainer = contentDiv.querySelector('.results');

    console.log('renderResults completed in', Date.now() - renderStart, 'ms');
}

function updateProgressStage(stage, message, current, total, done) {
    const stageMap = {
        'query_expansion': 'progress-query',
        'retrieval': 'progress-retrieval',
        'reranking': 'progress-reranking',
        'finalize': 'progress-reranking'
    };

    const stageId = stageMap[stage];
    if (!stageId) return;

    const stageEl = document.getElementById(stageId);
    if (!stageEl) return;

    const statusEl = stageEl.querySelector('.progress-stage-status');
    const fillEl = stageEl.querySelector('.progress-bar-fill');
    const iconEl = stageEl.querySelector('.progress-icon');

    if (done) {
        statusEl.textContent = '✓';
        fillEl.style.width = '100%';
        fillEl.classList.remove('indeterminate');
        iconEl.textContent = '✅';
    } else if (current !== undefined && total) {
        const percent = (current / total) * 100;
        statusEl.textContent = message || `${current}/${total}`;
        fillEl.style.width = `${percent}%`;
        fillEl.classList.remove('indeterminate');
    } else {
        // Active stage (indeterminate)
        statusEl.textContent = message || '';
        fillEl.classList.add('indeterminate');
        // Force browser to repaint before next update
        void fillEl.offsetHeight;
    }
}

function getPaperlessUrl(documentId) {
    return `${paperlessBaseUrl}/documents/${documentId}`;
}

function getDownloadUrl(documentId) {
    return `${paperlessBaseUrl}/api/documents/${documentId}/download/`;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Make toggleMetadata globally accessible
window.toggleMetadata = function(paperlessId) {
    const metadataContent = document.getElementById(`metadata-${paperlessId}`);
    const toggle = metadataContent.previousElementSibling;

    if (metadataContent.classList.contains('expanded')) {
        metadataContent.classList.remove('expanded');
        toggle.classList.remove('expanded');
    } else {
        metadataContent.classList.add('expanded');
        toggle.classList.add('expanded');
    }
};

// Store current search results for explain feature
let currentSearchResults = [];
let currentQuery = '';

// Make explainScore globally accessible
window.explainScore = async function(paperlessId, event) {
    if (event) {
        event.preventDefault();
        event.stopPropagation();
    }

    const button = event ? event.target : document.querySelector(`button[data-paperless-id="${paperlessId}"]`);
    const explanationContainer = document.getElementById(`explanation-${paperlessId}`);
    if (!explanationContainer) {
        console.error('Explanation container not found for', paperlessId);
        return;
    }
    const explanationText = explanationContainer.querySelector('.explanation-text');

    // If already showing, toggle off
    if (explanationContainer.style.display !== 'none') {
        explanationContainer.style.display = 'none';
        return;
    }

    // Find the result data
    const result = currentSearchResults.find(r => r.paperless_id == paperlessId);
    if (!result) {
        alert('Could not find result data');
        return;
    }

    // Disable button and show loading
    button.disabled = true;
    explanationText.textContent = 'Loading explanation...';
    explanationContainer.style.display = 'block';

    try {
        const response = await fetch('/api/explain', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: currentQuery,
                title: result.title,
                content: result.content,
                summary: result.summary,
                tags: result.tags || [],
                correspondent_name: result.correspondent_name
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        explanationText.textContent = data.explanation;

    } catch (error) {
        console.error('Explanation failed:', error);
        explanationText.textContent = `Error: ${error.message}`;
    } finally {
        button.disabled = false;
    }
};

// Focus on search input on page load
