# search_project/app.py
import os
import logging
import time # Added for current_date
from flask import Flask, render_template, request, jsonify
from search_engine import DualModeSearchEngine, SearchMode # DocumentType not directly used in app.py routes
from gemini_client import get_autocomplete_suggestions, get_ai_summary_or_fix

# Configure logging for Flask app
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# --- Search Engine Initialization ---
INDEX_FILE_PATH = os.getenv("INDEX_FILE_PATH", "search_index.pkl")
# Default model, can be overridden by index file or DENSE_MODEL_NAME env var
DEFAULT_DENSE_MODEL = "all-MiniLM-L6-v2" 
DENSE_MODEL_NAME_FOR_INIT = os.getenv("DENSE_MODEL_NAME", DEFAULT_DENSE_MODEL)
search_engine_instance = None

def get_search_engine():
    global search_engine_instance
    if search_engine_instance is None:
        logger.info("Initializing search engine...")
        # Pass the model name for initial loading if no index exists or index doesn't specify one
        search_engine_instance = DualModeSearchEngine(dense_model_name=DENSE_MODEL_NAME_FOR_INIT)
        if os.path.exists(INDEX_FILE_PATH):
            try:
                logger.info(f"Loading index from {INDEX_FILE_PATH}...")
                search_engine_instance.load_index(INDEX_FILE_PATH) # load_index might override dense_model_name
                logger.info("Index loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load index from '{INDEX_FILE_PATH}': {e}. Search may use an empty or basic engine.")
        else:
            logger.warning(f"Index file '{INDEX_FILE_PATH}' not found. Search engine will be empty or use default settings. "
                           "Run build_index_script.py to create an index.")
    return search_engine_instance

@app.before_request 
def ensure_engine_loaded():
    get_search_engine() # Ensures engine is loaded before first request

@app.context_processor
def inject_global_vars():
    # Makes these variables available to all templates
    return {
        'GOOGLE_API_KEY_SET': bool(os.getenv("GOOGLE_API_KEY")),
        'current_date': time.strftime("%Y-%m-%d")
    }

@app.route('/')
def index():
    # Values from query parameters (if any, e.g., from a previous search link) or defaults
    context = {
        'query': request.args.get('query', ''),
        'selected_mode': request.args.get('mode', 'AUTO').upper(),
        'sparse_weight': float(request.args.get('sparse_weight', 0.5)), # Default sparse weight
        'dense_weight': float(request.args.get('dense_weight', 0.5)),   # Default dense weight
        'max_results': int(request.args.get('max_results', 10)),      # Default max results
        'search_performed': False # No search on initial index page load
    }
    return render_template('index.html', **context)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '').strip()
    mode_str = request.args.get('mode', 'AUTO').upper()
    
    try:
        search_mode = SearchMode[mode_str]
    except KeyError:
        search_mode = SearchMode.AUTO # Default to AUTO if mode is invalid

    sparse_weight = float(request.args.get('sparse_weight', 0.5))
    dense_weight = float(request.args.get('dense_weight', 0.5))
    max_results = int(request.args.get('max_results', 10))

    engine = get_search_engine()
    results = []
    search_performed = False

    if not query:
        logger.info("Search attempted with empty query.")
        # results remains empty, search_performed is False
    elif not engine.initialized or not engine.documents:
         logger.warning("Search attempted but engine is not initialized or has no documents.")
         # results remains empty, search_performed is True (attempt was made)
         search_performed = True
    else:
        results = engine.search(
            query=query, 
            mode=search_mode, 
            max_results=max_results,
            sparse_weight=sparse_weight,
            dense_weight=dense_weight
        )
        search_performed = True
        
    # Add string version of doc_type for easier template rendering
    for r_item in results:
        r_item.doc_type_str = r_item.doc_type.name

    context = {
        'query': query,
        'results': results,
        'selected_mode': mode_str, # Mode selected by user
        'actual_search_mode': engine.detect_search_mode(query).name if search_mode == SearchMode.AUTO and query else mode_str, # Actual mode used
        'sparse_weight': sparse_weight,
        'dense_weight': dense_weight,
        'max_results': max_results,
        'search_performed': search_performed
    }
    return render_template('results.html', **context)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    partial_query = request.args.get('term', '').strip()
    # Mode selected by user in the UI for autocomplete context
    mode_from_ui = request.args.get('mode', 'AUTO').upper() 
    
    effective_mode_for_gemini = mode_from_ui
    if mode_from_ui == 'AUTO': 
        engine = get_search_engine()
        if partial_query: # If there's text, try to detect mode
            detected_mode = engine.detect_search_mode(partial_query)
            effective_mode_for_gemini = detected_mode.name
        else: # If partial query is empty (e.g. just focused), default to general
            effective_mode_for_gemini = "GENERAL"

    if not partial_query or len(partial_query) < 2:
        return jsonify([])

    suggestions = get_autocomplete_suggestions(partial_query, mode=effective_mode_for_gemini)
    return jsonify(suggestions)


@app.route('/ai_assist', methods=['POST'])
def ai_assist():
    if not os.getenv("GOOGLE_API_KEY"):
        return jsonify({"error": "AI Assistant is disabled. API key not configured."}), 403

    data = request.get_json()
    content = data.get('content')
    query = data.get('query')
    doc_type_str = data.get('doc_type') # This is DocumentType.name (e.g. "CODE", "GENERAL")
    language = data.get('language') # Optional, e.g. "python"

    if not all([content, query, doc_type_str]):
        return jsonify({"error": "Missing required data (content, query, or doc_type) for AI assist."}), 400

    try:
        # doc_type_str is already upper from SearchResult.doc_type_str
        ai_response = get_ai_summary_or_fix(content, query, doc_type_str, language)
        if ai_response:
            return jsonify({"ai_suggestion": ai_response})
        else:
            return jsonify({"ai_suggestion": "AI assistant could not provide a suggestion for this item at this time."})
    except Exception as e:
        logger.error(f"Error in /ai_assist endpoint: {e}")
        return jsonify({"error": "An internal error occurred with the AI assistant."}), 500

if __name__ == '__main__':
    # For production, use a WSGI server like Gunicorn or Waitress
    app.run(debug=False, host='0.0.0.0', port=5001) # debug=False for production