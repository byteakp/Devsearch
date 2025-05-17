# search_project/build_index_script.py
import os
import argparse
import logging
import re # Added for sanitizing relative path for doc_id
from search_engine import ( # Grouped imports for clarity
    Document, 
    DualModeSearchEngine, 
    CodeDocument, 
    GeneralDocument, 
    DocumentType, 
    detect_language
)

# Use a specific logger for this script for better log filtering if needed
script_logger = logging.getLogger("BuildIndexScript") 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Expansive set of extensions for code and general text documents
# Prioritize more specific (e.g. .py) over generic (e.g. .txt if it contains code)
CODE_EXTENSIONS = {
    # Python
    '.py', '.pyw', 
    # JavaScript & TypeScript & Web
    '.js', '.mjs', '.cjs', '.jsx', '.ts', '.tsx', '.vue', '.svelte',
    '.html', '.htm', '.xhtml', '.css', '.scss', '.less', '.sass', 
    # Java & JVM
    '.java', '.kt', '.kts', '.scala', '.groovy', '.gradle',
    # C-family
    '.c', '.h', '.cpp', '.hpp', '.cc', '.hh', '.cs', 
    # Go, Rust
    '.go', '.rs', 
    # Ruby, PHP, Perl
    '.rb', '.php', '.pl', '.pm',
    # Shell & Scripting
    '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat', '.cmd',
    # SQL
    '.sql', 
    # Config & Data interchange
    '.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.conf', '.cfg', '.env',
    # Notebooks (content extraction might need special handling, here treated as text)
    '.ipynb', 
    # Other common scripting/dev related: files without extensions common in dev
    'dockerfile', 'makefile', '.gitattributes', '.gitignore', '.editorconfig', 'gemfile', 'procfile' 
}
GENERAL_TEXT_EXTENSIONS = {
    '.txt', '.md', '.markdown', '.rtf', '.log', '.nfo', '.tex', '.rst', '.text', '.me', '.org', '.csv', '.tsv'
}
# Extensions for files that are typically binary and should be skipped if not explicitly handled
BINARY_SKIP_EXTENSIONS = {
    '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.ico', '.webp', # Images
    '.mp3', '.wav', '.ogg', '.flac', '.aac', '.m4a',                 # Audio
    '.mp4', '.avi', '.mkv', '.mov', '.webm', '.flv',                 # Video
    '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp', # Documents
    '.exe', '.dll', '.so', '.o', '.a', '.lib', '.dylib',      # Executables & Libraries
    '.zip', '.gz', '.tar', '.bz2', '.rar', '.7z', '.xz',      # Archives
    '.pkl', '.dat', '.bin', '.db', '.sqlite', '.img', '.iso', # Data/Binary/Disk Images
    '.class', '.jar', '.war', '.ear',                         # JVM compiled
    '.pyc', '.pyd',                                          # Python compiled
    '.obj', '.pdb',                                          # C/C++ compiled
    '.DS_Store', '.swp', '.swo', '.lock'                     # System/Editor metadata
}
# Default dense model name, also used in search_engine.py
DEFAULT_DENSE_MODEL_NAME = 'all-MiniLM-L6-v2'


def load_documents_from_directory(data_path: str) -> list[Document]:
    documents_loaded: list[Document] = []
    processed_doc_id_counter = 0 # Ensures unique IDs even if filenames repeat in subdirs
    
    if not os.path.isdir(data_path):
        script_logger.error(f"Data path '{data_path}' is not a directory or does not exist. Cannot load documents.")
        return documents_loaded

    script_logger.info(f"Starting document scan in directory: {data_path}")
    for root_dir, _, files_in_dir in os.walk(data_path, topdown=True): # topdown=True is default
        script_logger.debug(f"Scanning subdirectory: {root_dir}")

        # Skip common hidden/system directories like .git, .vscode, .idea, node_modules, venv, etc.
        dir_name_lower = os.path.basename(root_dir.lower())
        if dir_name_lower.startswith('.') or dir_name_lower in ['node_modules', '__pycache__', 'venv', 'env', '.tox', 'target', 'build', 'dist']:
            script_logger.info(f"Skipping directory: {root_dir}")
            continue


        for file_name_only in files_in_dir:
            full_file_path = os.path.join(root_dir, file_name_only)
            
            relative_file_path = os.path.relpath(full_file_path, data_path)
            sanitized_relative_path_for_id = relative_file_path.replace(os.sep, '-').replace(' ', '_')
            sanitized_relative_path_for_id = re.sub(r'[^a-zA-Z0-9_.-]', '', sanitized_relative_path_for_id)
            doc_id = f"docid_{sanitized_relative_path_for_id}_{processed_doc_id_counter}"
            processed_doc_id_counter += 1

            base_name_no_ext, extension_with_dot = os.path.splitext(file_name_only)
            file_name_lower = file_name_only.lower() 
            extension = extension_with_dot.lower()
            
            if not extension and file_name_lower in CODE_EXTENSIONS:
                extension = file_name_lower 
            
            if file_name_only.startswith('.') and extension not in CODE_EXTENSIONS and extension not in GENERAL_TEXT_EXTENSIONS:
                script_logger.debug(f"Skipping hidden file: {full_file_path}")
                continue
            
            if extension in BINARY_SKIP_EXTENSIONS and \
               extension not in CODE_EXTENSIONS and \
               extension not in GENERAL_TEXT_EXTENSIONS:
                script_logger.debug(f"Skipping binary file (type '{extension}'): {full_file_path}")
                continue

            try:
                file_size_mb = os.path.getsize(full_file_path) / (1024 * 1024)
                max_file_size_mb = 10 
                if file_size_mb > max_file_size_mb:
                    script_logger.warning(f"Skipping large file ({file_size_mb:.2f}MB > {max_file_size_mb}MB): {full_file_path}")
                    continue

                with open(full_file_path, 'r', encoding='utf-8', errors='ignore') as file_obj:
                    content = file_obj.read()
                
                if not content.strip(): 
                    script_logger.warning(f"Skipping empty or whitespace-only file: {full_file_path}")
                    continue
                
                doc_type_to_assign: DocumentType
                language_hint: Optional[str] = None 

                if extension in CODE_EXTENSIONS:
                    doc_type_to_assign = DocumentType.CODE
                    if extension and extension.startswith('.'): 
                        language_hint = extension[1:] 
                    elif extension: 
                        language_hint = extension
                elif extension in GENERAL_TEXT_EXTENSIONS:
                    doc_type_to_assign = DocumentType.GENERAL
                    if extension == '.log':
                        content_sample_for_log = content[:2048] 
                        temp_lang_check = detect_language(content_sample_for_log)
                        if temp_lang_check not in ["unknown", "markdown", "text"]: 
                             script_logger.info(f"Log file '{file_name_only}' content suggests language '{temp_lang_check}'. Classifying as CODE.")
                             doc_type_to_assign = DocumentType.CODE
                             language_hint = temp_lang_check
                else: 
                    script_logger.info(f"Ambiguous extension '{extension}' for '{file_name_only}'. Analyzing content...")
                    content_sample_for_unknown = content[:3000] 
                    lang_from_content = detect_language(content_sample_for_unknown)
                    if lang_from_content not in ["unknown", "markdown", "text"]: 
                        doc_type_to_assign = DocumentType.CODE
                        language_hint = lang_from_content
                        script_logger.info(f"  Content analysis suggests CODE (lang: {language_hint}).")
                    else:
                        doc_type_to_assign = DocumentType.GENERAL
                        script_logger.info(f"  Content analysis suggests GENERAL (detected: {lang_from_content}).")
                
                metadata = {
                    "source_file_path": full_file_path, 
                    "filename": file_name_only, 
                    "extension": extension if extension else "none" 
                }
                
                new_doc: Document
                if doc_type_to_assign == DocumentType.CODE:
                    new_doc = CodeDocument(id=doc_id, content=content, metadata=metadata, language=language_hint)
                else: # GENERAL
                    new_doc = GeneralDocument(id=doc_id, content=content, metadata=metadata)
                
                documents_loaded.append(new_doc)
                final_lang = new_doc.metadata.get('language', 'N/A') if new_doc.doc_type == DocumentType.CODE else 'N/A'
                script_logger.info(f"Loaded: '{relative_file_path}' as {new_doc.doc_type.name} (Lang: {final_lang})")

            except OSError as e: 
                script_logger.error(f"OS error reading file {full_file_path}: {e}")
            except Exception as e: 
                script_logger.error(f"Unexpected error processing file {full_file_path}: {e}")
    
    script_logger.info(f"Document scanning complete. Loaded {len(documents_loaded)} documents from '{data_path}'.")
    return documents_loaded

def main():
    parser = argparse.ArgumentParser(
        description="Builds or updates the search index for the Dual-Mode Search Engine.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter 
    )
    parser.add_argument("--data_path", type=str, default="./sample_data/",
                        help="Path to the directory containing documents to index.")
    parser.add_argument("--index_file", type=str, default="search_index.pkl",
                        help="Path to save the built index file.")
    parser.add_argument("--model_name", type=str, default=DEFAULT_DENSE_MODEL_NAME,
                        help="Name of the sentence-transformer model for dense embeddings. "
                             "Set to an empty string ('') or 'NONE' to disable dense embeddings if building from scratch.")
    parser.add_argument("--force_re_embed", action="store_true",
                        help="Force re-embedding of all documents, even if embeddings exist in a loaded index.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level for the script.")
    
    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level.upper()) 
    script_logger.info(f"Log level set to {args.log_level.upper()}")

    if not os.path.isdir(args.data_path):
        script_logger.critical(f"Data path '{args.data_path}' does not exist or is not a directory. Please create it and add documents. Exiting.")
        return

    script_logger.info(f"Starting index build process. Data source: '{args.data_path}', Index file output: '{args.index_file}'")
    
    dense_model_for_engine_init = args.model_name
    if args.model_name.strip().upper() in ['', 'NONE', 'DISABLE', 'FALSE']:
        script_logger.info("Dense embeddings explicitly disabled for this build process via --model_name argument.")
        dense_model_for_engine_init = None 
    
    engine = DualModeSearchEngine(dense_model_name=dense_model_for_engine_init)

    if os.path.exists(args.index_file):
        script_logger.info(f"Existing index file found at '{args.index_file}'. Attempting to load and update.")
        engine.load_index(args.index_file) 
        if not engine.initialized:
            script_logger.warning(f"Failed to properly load existing index from '{args.index_file}'. "
                                  "Will proceed to build a new index with current settings.")
            engine = DualModeSearchEngine(dense_model_name=dense_model_for_engine_init) 
    else:
        script_logger.info(f"No existing index file at '{args.index_file}'. A new index will be created.")

    newly_scanned_docs = load_documents_from_directory(args.data_path)
    
    if not newly_scanned_docs and not engine.documents: 
        script_logger.warning("No documents found in data path and no existing index was loaded (or it was empty). "
                              "The resulting index will be empty.")
    
    if newly_scanned_docs:
        script_logger.info(f"Adding/updating {len(newly_scanned_docs)} documents from data path into the search engine.")
        engine.add_documents(newly_scanned_docs) 
    else:
        script_logger.info("No new documents found in the data path to add or update in the engine.")

    if engine.documents: 
        script_logger.info(f"Building/updating index components for {len(engine.documents)} total chunks...")
        engine.build_index(force_re_embed=args.force_re_embed)
    else:
        script_logger.warning("Engine has no documents after loading and scanning. "
                              "Skipping index component build. An empty (but valid) index will be saved.")
        engine.initialized = True 

    engine.save_index(args.index_file)

    script_logger.info(f"Index building process complete. Index saved to '{args.index_file}'.")
    script_logger.info(f"Final index statistics:")
    script_logger.info(f"  Total chunks/documents indexed: {len(engine.documents)}")
    script_logger.info(f"  Code item chunks: {len(engine.code_docs)}")
    script_logger.info(f"  General item chunks: {len(engine.general_docs)}")
    if engine.dense_model and engine.current_dense_model_name:
        script_logger.info(f"  Dense model active: '{engine.current_dense_model_name}' (Vector Dim: {engine.vector_dim})")
        script_logger.info(f"  Number of dense vectors: {len(engine.doc_vectors_dense)}")
    else:
        script_logger.info("  Dense embeddings are currently disabled or no dense model is loaded.")

if __name__ == "__main__":
    main()