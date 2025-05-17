# Dual-Mode AI Search Engine: Usage Guide

**Version:** 1.0 (As of May 2025)

This guide covers how to use the search engine for code and general queries, along with performance details.

## 🔍 How to Use the Search Engine

### 1. **Building the Search Index**
To search, you first need to build an index from your documents.

Run the following command in the `search_project` directory (with virtual environment activated):
```bash
python build_index_script.py --data_path ./sample_data/ --index_file search_index.pkl
```
- **Options:**
  - `--data_path`: Directory with documents (default: `./sample_data/`).
  - `--index_file`: Where to save the index (default: `search_index.pkl`).
  - `--model_name`: Sentence Transformer model (default: `all-MiniLM-L6-v2`). Use `""` or `NONE` to disable dense embeddings.
  - `--force_re_embed`: Force recomputation of dense embeddings.
  - `--log_level`: Set logging level (DEBUG, INFO, WARNING, ERROR; default: INFO).

This indexes code files (e.g., `.py`, `.js`, `.log`) and general documents (e.g., `.txt`, `.md`) from the specified directory.

### 2. **Running the Search Application**
Start the Flask web app:
```bash
python app.py
```
- Access the interface at `http://localhost:5001` (or the address shown in the terminal).
- The UI includes a search bar, mode selection (Code, General, Auto-Detect), and sliders for sparse (`sparse_weight`) and dense (`dense_weight`) scoring.

### 3. **Performing Searches**
- **Code-Specific Search:**
  - Select "Code" mode or let Auto-Detect identify code-related queries.
  - Enter queries like:
    - Code snippets (e.g., `def example_function():`).
    - Error messages (e.g., `TypeError: 'NoneType' object is not subscriptable`).
    - API/framework issues (e.g., `Flask route not found`).
  - Results include code snippets, debug logs, or stack traces with highlighted matches.
  - AI insights (via Gemini API) may suggest fixes or explanations.

- **General Purpose Search:**
  - Select "General" mode or use Auto-Detect for non-code queries.
  - Enter queries like:
    - Topics (e.g., `machine learning basics`).
    - Questions (e.g., `What is BM25?`).
    - Keywords (e.g., `search engine optimization`).
  - Results include articles, summaries, or tools with highlighted snippets.
  - AI insights provide summaries or additional context.

- **Auto-Detect Mode:**
  - Automatically selects Code or General mode based on query content (e.g., code syntax vs. natural language).

- **Smart Autocomplete:**
  - As you type, the Gemini API suggests query completions tailored to the mode (e.g., code syntax for Code mode, topics for General).

### 4. **Search Features**
- **Hybrid Retrieval:** Combines BM25 (keyword-based) and Sentence Transformer (semantic) scores using Reciprocal Rank Fusion (RRF).
- **Highlighting:** Query terms are bolded in results.
- **Adjustable Parameters:** Use UI sliders to balance sparse and dense weights (0 to 1).

## 🚀 Performance
- **Query Speed:** Designed for responses under 300ms on moderately sized datasets (<10k documents) on a mid-tier machine (e.g., 8GB RAM, 4-core CPU).
- **Indexing Efficiency:** Building an index for ~1k documents takes ~1-2 minutes, depending on dense embedding usage.
- **Optimization Notes:**
  - Sparse search (BM25) is faster for keyword-heavy queries.
  - Dense search (Sentence Transformers) improves semantic understanding but is slower; disable via `--model_name NONE` for faster indexing/search.
  - Adjust `max_chunk_size` and `chunk_overlap` in `search_engine.py` to balance precision and speed.