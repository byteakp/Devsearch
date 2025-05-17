# Dual-Mode AI Search Engine

**Version**: 1.0 (May 2025)
A Flask-based search engine with hybrid retrieval (BM25 + Sentence Transformers) and Gemini API integration for code-specific and general-purpose searches. Features smart autocomplete, adjustable scoring, and highlighted results.

## Features
- **Dual Modes**: Code-specific (e.g., code snippets, error logs) and General-purpose (e.g., articles, topics) search with Auto-Detect.
- **Hybrid Retrieval**: Combines BM25 (keyword-based) and Sentence Transformer (semantic) scores via Reciprocal Rank Fusion (RRF).
- **AI Insights**: Gemini API provides query completions and contextual suggestions.
- **Web Interface**: Flask UI with search bar, mode selection, and sliders for sparse/dense scoring weights.
- **Performance**: <300ms query response for <10k documents; ~1-2 min indexing for ~1k documents.

## Preview
![Dual-Mode AI Search Engine Demo](https://ibb.co/39JVfwYz)

![Search Engine Preview](https://ibb.co/4n7B29bx)

## Requirements
- Python 3.8+
- Virtual environment (recommended)
- Dependencies: `flask`, `sentence-transformers`, `rank-bm25`, Gemini API key
- Install via: `pip install -r requirements.txt`

## Setup
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd search_project
   ```
2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Gemini API**:
   - Add your Gemini API key to `config.py` or as an environment variable (`GEMINI_API_KEY`).

## Usage
### 1. Build the Search Index
Index documents (code files like `.py`, `.js`, `.log` or text like `.txt`, `.md`):
```bash
python build_index_script.py --data_path ./sample_data/ --index_file search_index.pkl
```
- **Options**:
  - `--data_path`: Document directory (default: `./sample_data/`).
  - `--index_file`: Index save path (default: `search_index.pkl`).
  - `--model_name`: Sentence Transformer model (default: `all-MiniLM-L6-v2`). Use `NONE` to disable dense embeddings.
  - `--force_re_embed`: Recompute embeddings.
  - `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR; default: INFO).

### 2. Run the Search Application
Start the Flask app:
```bash
python app.py
```
- Access at `http://localhost:5001` (check terminal for exact address).
- UI includes search bar, mode selector (Code, General, Auto-Detect), and scoring sliders.

### 3. Perform Searches
- **Code Search**:
  - Mode: Select "Code" or use Auto-Detect.
  - Queries: Code snippets (e.g., `def example_function():`), errors (e.g., `TypeError: 'NoneType' object is not subscriptable`), or API issues (e.g., `Flask route not found`).
  - Results: Code, logs, or stack traces with highlighted matches; AI suggestions for fixes.
- **General Search**:
  - Mode: Select "General" or use Auto-Detect.
  - Queries: Topics (e.g., `machine learning basics`), questions (e.g., `What is BM25?`), or keywords (e.g., `search engine optimization`).
  - Results: Articles or tools with highlighted snippets; AI summaries.
- **Auto-Detect**: Automatically selects mode based on query (code syntax vs. natural language).
- **Smart Autocomplete**: Gemini API suggests completions tailored to mode.

### 4. Customize Search
- Adjust sparse (`sparse_weight`) and dense (`dense_weight`) scoring via UI sliders (0 to 1).
- Optimize indexing/search speed by disabling dense embeddings (`--model_name NONE`).
- Tune `max_chunk_size` and `chunk_overlap` in `search_engine.py` for precision vs. speed.

## Performance
- **Query Speed**: <300ms for <10k documents on mid-tier hardware (8GB RAM, 4-core CPU).
- **Indexing**: ~1-2 minutes for ~1k documents.
- **Tips**:
  - Use BM25 (sparse) for faster keyword searches.
  - Disable dense embeddings for quicker indexing.
  - Adjust chunk sizes in `search_engine.py` for performance.

## Project Structure
```
search_project/
├── app.py                # Flask web app
├── build_index_script.py # Index builder
├── search_engine.py      # Core search logic
├── sample_data/          # Sample documents
├── templates/            # HTML templates
├── static/               # CSS/JS for UI
└── requirements.txt      # Dependencies
```

## Contributing
- Report issues or suggest features via the repository's issue tracker.
- Submit pull requests with clear descriptions of changes.

## License
MIT License. See `LICENSE` for details.

## Screenshots
Refer to the repository's `/docs` folder or issue tracker for UI screenshots and demo visuals.