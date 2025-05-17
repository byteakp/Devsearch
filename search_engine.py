# search_project/search_engine.py
"""Core search engine implementation with dual-mode support.
Supports both sparse (BM25) and dense vector representations for hybrid search."""
import os
import re
# json not directly used here, but useful for metadata if it were complex
import time
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import heapq # Used for N-largest if result set is huge
import math
from sentence_transformers import SentenceTransformer 
from sklearn.metrics.pairwise import cosine_similarity 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DENSE_MODEL_NAME = 'all-MiniLM-L6-v2' 

class DocumentType(Enum):
    CODE = 1
    GENERAL = 2

@dataclass
class Document:
    id: str
    content: str
    doc_type: DocumentType # Field for the document type
    metadata: Dict[str, Any] = field(default_factory=dict)
    original_doc_id: Optional[str] = None
    chunk_seq: Optional[int] = None

    def __post_init__(self):
        # Auto-detect language for CODE documents if not provided or unknown
        if self.doc_type == DocumentType.CODE:
            current_lang = self.metadata.get("language")
            # Only detect if language is not set, or was explicitly set to 'unknown' by a hint.
            # Ensure current_lang is treated as a string for .lower() if it exists.
            if current_lang is None or (isinstance(current_lang, str) and current_lang.lower() == "unknown"):
                detected = detect_language(self.content)
                # Only update if detection is valid (not None/empty) and actually found something other than 'unknown'
                if detected and detected.lower() != "unknown": 
                    self.metadata["language"] = detected
            # If 'language' key exists but is an empty string (e.g. language_hint was ""), also try to detect.
            elif not current_lang and isinstance(current_lang, str): 
                 detected_lang_on_empty = detect_language(self.content)
                 if detected_lang_on_empty and detected_lang_on_empty.lower() != "unknown":
                    self.metadata["language"] = detected_lang_on_empty

class CodeDocument(Document):
    def __init__(self, id: str, content: str,
                 language: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 original_doc_id: Optional[str] = None,
                 chunk_seq: Optional[int] = None,
                 doc_type_from_data: Optional[DocumentType] = None, # Catches 'doc_type' from **chunk_data_doc
                 **additional_kwargs_for_metadata): # Catches any other unexpected kwargs

        actual_metadata = metadata.copy() if metadata is not None else {}
        actual_metadata.update(additional_kwargs_for_metadata) 

        if language: # Explicit language parameter takes precedence
            actual_metadata["language"] = language
        
        super().__init__(id=id,
                         content=content,
                         doc_type=DocumentType.CODE, # Explicitly setting the type for CodeDocument
                         metadata=actual_metadata,
                         original_doc_id=original_doc_id,
                         chunk_seq=chunk_seq)

class GeneralDocument(Document):
    def __init__(self, id: str, content: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 original_doc_id: Optional[str] = None,
                 chunk_seq: Optional[int] = None,
                 doc_type_from_data: Optional[DocumentType] = None, # Catches 'doc_type'
                 **additional_kwargs_for_metadata):

        actual_metadata = metadata.copy() if metadata is not None else {}
        actual_metadata.update(additional_kwargs_for_metadata)

        super().__init__(id=id,
                         content=content,
                         doc_type=DocumentType.GENERAL, # Explicitly setting the type for GeneralDocument
                         metadata=actual_metadata,
                         original_doc_id=original_doc_id,
                         chunk_seq=chunk_seq)

@dataclass
class SearchResult:
    doc_id: str 
    score: float
    content: str 
    metadata: Dict[str, Any]
    doc_type: DocumentType 
    highlights: List[str] = field(default_factory=list)
    original_doc_id: Optional[str] = None 
    doc_type_str: str = "" # String representation (e.g., "CODE"), added by app.py

class SearchMode(Enum):
    CODE = 1
    GENERAL = 2
    AUTO = 3

class TokenizationMode(Enum):
    WORD = 1        
    SUBWORD = 2     
    CHARACTER = 3   

def detect_language(code_snippet: str) -> str:
    if not code_snippet or len(code_snippet) < 10: 
        return "unknown"

    language_rules = {
        "python": {
            "patterns": [r"^\s*def\s+\w+\s*\(.*?\):", r"^\s*class\s+\w+(\(.*?\))?:", r"^\s*import\s+[\w.]+", r"^\s*from\s+[\w.]+\s+import"],
            "keywords": ["self", "__init__", "elif", "lambda", "yield", "async def", "await", "with open"],
            "score_mult": 1.2 
        },
        "javascript": {
            "patterns": [r"^\s*function\s*[\w$]*\s*\(", r"\bconst\s+[\w$]+\s*=", r"\blet\s+[\w$]+\s*=", r"^\s*class\s+[\w$]+\s*(extends\s+[\w$]+)?\s*{"],
            "keywords": ["this", "async function", "await", "=>", "document.getElementById", "module.exports", "JSON.parse"],
            "score_mult": 1.1
        },
        "java": {
            "patterns": [r"public\s+class\s+\w+", r"private\s+(static\s+)?\w+\s+\w+;", r"System\.out\.println"],
            "keywords": ["@Override", "ArrayList", "HashMap", "static void main", "throws Exception", "public static final"],
            "score_mult": 1.0
        },
        "html": {
            "patterns": [r"<!DOCTYPE\s+html>", r"<html\s*.*?>", r"<head\s*>", r"<body\s*>", r"<div\s+id="],
            "keywords": ["</head>", "</body>", "<title>", "<script src=", "stylesheet", "href="],
            "score_mult": 1.0
        },
        "css": {
            "patterns": [r"^\s*[\.#@][\w-]+\s*[{,\s]", r"^\s*[\w-]+\s*:", r"@import\s+url"], 
            "keywords": ["font-family:", "background-color:", "!important", "display: flex", "position: absolute"],
            "score_mult": 1.0
        },
        "sql": {
            "patterns": [r"\bSELECT\b[\s\S]+?\bFROM\b", r"\bINSERT\s+INTO\b", r"\bUPDATE\b[\s\S]+?\bSET\b", r"\bCREATE\s+(TABLE|INDEX|VIEW|DATABASE)\b"],
            "keywords": ["JOIN", "WHERE", "GROUP BY", "ORDER BY", "PRIMARY KEY", "VARCHAR", "DATABASE", "ALTER TABLE"],
            "score_mult": 1.1
        },
        "cpp": { # C++
            "patterns": [r"#include\s*<[\w.]+>", r"std::\w+", r"int\s+main\s*\("],
            "keywords": ["cout", "cin", "vector<", "nullptr", "template <", "namespace ", "->"],
            "score_mult": 1.0
        },
        "csharp": { 
            "patterns": [r"public\s+class\s+\w+", r"namespace\s+\w+", r"using\s+System;"],
            # Corrected: string[] args (removed invalid escape \)
            "keywords": ["Console.WriteLine", "async Task", "string[] args", "var ", " new ", "get;", "set;", "yield return"],
             "score_mult": 1.0
        },
        "ruby": {
            "patterns": [r"^\s*def\s+\w+", r"^\s*class\s+\w+\s*<"],
            "keywords": ["puts", "require ", "end\b", "yield", "attr_accessor", ":symbol"],
            "score_mult": 1.0
        },
        "php": {
            "patterns": [r"<\?php", r"\$\w+\s*=", r"function\s+\w+\s*\("],
            "keywords": ["echo ", "isset(", "new PDO(", "namespace ", "use ", "->", "::"],
             "score_mult": 1.0
        },
        "go": {
            "patterns": [r"func\s+\w+\s*\(", r"package\s+\w+", r"import\s*\("],
            "keywords": ["fmt.Println", "make(", "chan ", "defer ", "go func", ":="],
            "score_mult": 1.0
        },
        "rust": {
            "patterns": [r"fn\s+\w+\s*\(", r"let\s+(mut\s+)?\w+", r"impl\s+\w+"],
            "keywords": ["println!", "match ", "struct ", "enum ", "async fn", "->", "::", "use "],
            "score_mult": 1.0
        },
         "markdown": { 
            "patterns": [r"^#+\s+", r"\*\*[\s\S]+?\*\*",r"__[\s\S]+?__", r"\*[\s\S]+?\*", r"_[\s\S]+?_", r"`[\s\S]+?`", r"\[.*?\]\(.*?\)", r"^\s*-\s+", r"^\s*\*\s+", r"^\s*\d+\.\s+"],
            "keywords": ["```"], # Code block fence
            "score_mult": 0.9 
        },
        "text": { # Generic text as a fallback if nothing else matches strongly
            "patterns": [r"\b\w+\b"], # Just needs words
            "keywords": [],
            "score_mult": 0.1 # Very low priority
        }
    }

    snippet_to_check = code_snippet[:4000] 
    lines = snippet_to_check.splitlines()
    if len(lines) > 150: lines = lines[:150] 

    scores = Counter()
    first_few_lines_joined = " ".join(lines[:15])
    
    for lang, rules in language_rules.items():
        lang_score = 0
        for pattern_str in rules.get("patterns", []):
            try:
                target_text_for_pattern = first_few_lines_joined if pattern_str.strip().startswith("^") else snippet_to_check
                if re.search(pattern_str, target_text_for_pattern, re.IGNORECASE | re.MULTILINE):
                    lang_score += 1
            except re.error as e:
                logger.debug(f"Regex error for lang {lang} pattern '{pattern_str}': {e}")
        
        content_sample_for_keywords = " ".join(lines[:20] + lines[-10:]) 
        for keyword in rules.get("keywords", []):
            if re.search(r'\b' + re.escape(keyword) + r'\b', content_sample_for_keywords, re.IGNORECASE) or \
               (not keyword.isalnum() and keyword in content_sample_for_keywords):
                lang_score += 1.5 

        if lang_score > 0:
            scores[lang] = lang_score * rules.get("score_mult", 1.0)

    if lines and lines[0].startswith("#!"):
        shebang_line = lines[0].lower()
        if "python" in shebang_line: scores["python"] += 5
        elif "bash" in shebang_line or "sh" in shebang_line: return "bash" 
        elif "perl" in shebang_line: return "perl"
        elif "node" in shebang_line: scores["javascript"] += 5; return "javascript" 
        elif "ruby" in shebang_line: scores["ruby"] += 5

    if not scores:
        return "unknown"

    best_candidates = scores.most_common(2)
    if not best_candidates: return "unknown"
    best_lang, top_score = best_candidates[0]

    if top_score < 1.5 and best_lang not in ["markdown", "text"]: 
        return "text" if "text" in scores and scores["text"] > 0.5 else "unknown"
    if top_score < 0.5: 
        return "unknown"
        
    return best_lang


class Tokenizer:
    def __init__(self, mode: TokenizationMode = TokenizationMode.WORD):
        self.mode = mode
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set: 
        return {
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", 
            "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", 
            "this", "to", "was", "will", "with", "i", "me", "my", "myself", "we", "our", "ours", "ourselves", 
            "you", "your", "yours", "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", 
            "what", "which", "who", "whom", "do", "does", "did", "am", "is", "are", "was", "were", "has", 
            "have", "had", "can", "could", "should", "would", "may", "might", "must", "com", "org", "net"
        }

    def tokenize(self, text: str, remove_stopwords: bool = True, min_length: int = 2) -> List[str]:
        if not text: return []
        
        processed_text = text.lower() 
        
        tokens: List[str] = []
        if self.mode == TokenizationMode.WORD:
            # Updated regex to capture words starting with letter/underscore, ensuring they have at least min_length if > 1, or any length if just numbers
            # Example: "py" is a word, "a" might be filtered by min_length unless it's a digit like "1"
            # This tries to get meaningful "words" for BM25 or highlighting.
            raw_tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[0-9]+', processed_text)
            tokens = [t for t in raw_tokens if len(t) >= (1 if t.isdigit() else min_length) or len(t) >= min_length]

        elif self.mode == TokenizationMode.SUBWORD: 
            base_words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', processed_text) 
            for word in base_words:
                camel_split = re.sub(r'([A-Z][a-z]+)', r' \1', re.sub(r'([A-Z]+)', r' \1', word)).split()
                temp_subwords = []
                for sub_part in camel_split:
                    temp_subwords.extend(s for s in sub_part.split('_') if s) 
                tokens.extend(s.lower() for s in temp_subwords if s)
            tokens = list(dict.fromkeys(tokens)) # Unique subwords
        elif self.mode == TokenizationMode.CHARACTER:  
            n = 3 
            if len(processed_text) < n : 
                if len(processed_text) >= min_length: tokens = [processed_text]
                else: tokens = []
            else:
                tokens = [processed_text[i:i+n] for i in range(len(processed_text) - n + 1)]
        
        final_tokens = []
        for t in tokens: # Apply min_length again for subword/char and stopwords
            if len(t) >= min_length : # Generic min_length check for all modes after initial tokenization
                if remove_stopwords and self.mode != TokenizationMode.CHARACTER and t in self.stopwords:
                    continue
                final_tokens.append(t)
        return final_tokens


class BM25Vectorizer:
    def __init__(self, k1: float = 1.2, b: float = 0.75): 
        self.k1 = k1
        self.b = b
        self.idf: Dict[str, float] = {}
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.document_count: int = 0
        self.tokenizer = Tokenizer(mode=TokenizationMode.WORD) 

    def fit(self, documents: List[Document]):
        term_doc_counts = Counter() 
        total_doc_length_sum = 0
        
        self.document_count = len(documents)
        if self.document_count == 0: 
            logger.warning("BM25Vectorizer.fit: No documents provided.")
            self.avg_doc_length = 0.0
            return

        for doc in documents:
            terms = self.tokenizer.tokenize(doc.content, remove_stopwords=True, min_length=2)
            self.doc_lengths[doc.id] = len(terms)
            total_doc_length_sum += len(terms)
            for term in set(terms): 
                term_doc_counts[term] += 1
        
        self.avg_doc_length = total_doc_length_sum / self.document_count if self.document_count > 0 else 0.0

        for term, doc_freq in term_doc_counts.items():
            numerator = self.document_count - doc_freq + 0.5
            denominator = doc_freq + 0.5
            self.idf[term] = math.log(numerator / denominator + 1.0) 

    def score(self, query_term_idfs: Dict[str, float], doc_id: str, doc_terms: List[str]) -> float:
        current_doc_bm25_score = 0.0
        doc_len = self.doc_lengths.get(doc_id, self.avg_doc_length) 
        
        if self.avg_doc_length == 0 and doc_len == 0: 
             return 0.0
        safe_avg_doc_length = self.avg_doc_length if self.avg_doc_length > 0 else 1.0 

        term_freq_in_doc = Counter(doc_terms)

        for term_in_query, idf_of_query_term in query_term_idfs.items():
            if term_in_query in term_freq_in_doc:
                freq_in_doc = term_freq_in_doc[term_in_query]
                
                numerator_doc_tf = freq_in_doc * (self.k1 + 1)
                denominator_doc_tf = freq_in_doc + self.k1 * (1 - self.b + self.b * (doc_len / safe_avg_doc_length))
                
                doc_term_score_component = numerator_doc_tf / denominator_doc_tf if denominator_doc_tf != 0 else 0
                
                current_doc_bm25_score += idf_of_query_term * doc_term_score_component
                
        return current_doc_bm25_score


class DualModeSearchEngine:
    def __init__(self, dense_model_name: Optional[str] = DEFAULT_DENSE_MODEL_NAME):
        self.documents: Dict[str, Document] = {} 
        self.doc_vectors_sparse_terms: Dict[str, List[str]] = {} 
        self.doc_vectors_dense: Dict[str, np.ndarray] = {}    
        
        self.bm25 = BM25Vectorizer()
        self.query_tokenizer = Tokenizer(mode=TokenizationMode.WORD) 
        
        self.dense_model: Optional[SentenceTransformer] = None
        self.vector_dim: int = 0
        self.current_dense_model_name: Optional[str] = None

        if dense_model_name: 
            try:
                logger.info(f"Attempting to load dense model: {dense_model_name}")
                self.dense_model = SentenceTransformer(dense_model_name)
                self.vector_dim = self.dense_model.get_sentence_embedding_dimension()
                self.current_dense_model_name = dense_model_name
                logger.info(f"Dense model '{dense_model_name}' loaded successfully. Vector dim: {self.vector_dim}")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model '{dense_model_name}': {e}. Dense search will be disabled for now.")
                self.dense_model = None 
        else:
            logger.info("No dense model name provided at initialization. Dense search disabled.")


        self.code_docs: set[str] = set()    
        self.general_docs: set[str] = set() 
        self.initialized: bool = False      
        self.max_chunk_size: int = 500      
        self.chunk_overlap: int = 50        

    def _chunk_document(self, document: Document) -> List[Document]:
        chunks: List[Document] = []
        content = document.content
        doc_id_base = document.id

        if len(content) <= self.max_chunk_size * 1.1: 
            document.original_doc_id = doc_id_base 
            document.chunk_seq = 0
            return [document]

        logger.info(f"Chunking document {doc_id_base} (length: {len(content)}) into max {self.max_chunk_size} char chunks with {self.chunk_overlap} overlap.")
        
        start = 0
        seq = 0
        while start < len(content):
            end = min(len(content), start + self.max_chunk_size)
            chunk_content = content[start:end]
            chunk_id = f"{doc_id_base}::chunk{seq}" 
            
            # Prepare data for subclass constructors
            # The `doc_type_from_data` key will be caught by the respective __init__ methods
            # and the class will set its own fixed doc_type.
            chunk_doc_data_for_constructor = {
                "id": chunk_id,
                "content": chunk_content,
                "doc_type_from_data": document.doc_type, # Pass parent's type for context if needed by constructor, though it's ignored
                "metadata": document.metadata.copy(), 
                "original_doc_id": doc_id_base,
                "chunk_seq": seq
            }
            
            new_chunk: Document
            if document.doc_type == DocumentType.CODE: 
                if "language" in document.metadata: # Pass language hint if parent had it
                    chunk_doc_data_for_constructor["language"] = document.metadata["language"]
                new_chunk = CodeDocument(**chunk_doc_data_for_constructor)
            else: 
                new_chunk = GeneralDocument(**chunk_doc_data_for_constructor)
            
            chunks.append(new_chunk)
            seq += 1
            
            next_start_pos = start + self.max_chunk_size - self.chunk_overlap
            
            if next_start_pos < len(content) and (len(content) - next_start_pos) < (self.chunk_overlap / 2):
                chunks[-1].content = content[start:] 
                logger.debug(f"Extended last chunk {chunks[-1].id} to cover small remainder.")
                break 
            
            start = next_start_pos 
            
            if start >= len(content): 
                break
        
        return chunks if chunks else [document] 

    def add_document(self, document: Document, chunk_if_needed: bool = True):
        if document.original_doc_id is not None and document.id != document.original_doc_id :
            processed_documents = [document] 
        elif chunk_if_needed:
            processed_documents = self._chunk_document(document)
        else: 
            if document.original_doc_id is None: document.original_doc_id = document.id
            if document.chunk_seq is None: document.chunk_seq = 0
            processed_documents = [document]

        for p_doc in processed_documents:
            if p_doc.id in self.documents: 
                logger.warning(f"Document chunk ID '{p_doc.id}' already exists. Overwriting with new content/metadata.")
            self.documents[p_doc.id] = p_doc 
            if p_doc.doc_type == DocumentType.CODE:
                self.code_docs.add(p_doc.id)
                if p_doc.id in self.general_docs: self.general_docs.remove(p_doc.id) 
            else: 
                self.general_docs.add(p_doc.id)
                if p_doc.id in self.code_docs: self.code_docs.remove(p_doc.id) 
            # The __post_init__ of Document (and thus its subclasses) handles language detection
            logger.debug(f"Stored chunk: ID='{p_doc.id}', OrigID='{p_doc.original_doc_id}', Type='{p_doc.doc_type.name}', Lang='{p_doc.metadata.get('language', 'N/A')}'")

    def add_documents(self, documents: List[Document], chunk_if_needed: bool = True):
        for doc in documents:
            self.add_document(doc, chunk_if_needed=chunk_if_needed)

    def build_index(self, force_re_embed: bool = False): 
        if not self.documents:
            logger.warning("No documents loaded to build index. Index will be empty but initialized.")
            self.initialized = True 
            return
            
        start_time = time.time()
        logger.info(f"Building index for {len(self.documents)} document chunks...")
        
        doc_list = list(self.documents.values()) 
        
        self.bm25.fit(doc_list) 
        self.doc_vectors_sparse_terms.clear() 
        for doc_id, doc_obj in self.documents.items():
             self.doc_vectors_sparse_terms[doc_id] = self.bm25.tokenizer.tokenize(doc_obj.content, remove_stopwords=True, min_length=2)
        logger.info("BM25 index components built.")

        if self.dense_model:
            contents_to_embed: List[str] = []
            doc_ids_for_embedding: List[str] = []
            
            for doc_id, doc in self.documents.items():
                if force_re_embed or doc_id not in self.doc_vectors_dense:
                    contents_to_embed.append(doc.content)
                    doc_ids_for_embedding.append(doc_id)
            
            if not contents_to_embed:
                logger.info("No new or forced documents to embed for dense vectors.")
            else:
                logger.info(f"Generating dense embeddings for {len(contents_to_embed)} chunks (new or forced)...")
                try:
                    batch_size = 128 
                    all_new_embeddings_list: List[np.ndarray] = [] 
                    for i in range(0, len(contents_to_embed), batch_size):
                        batch_contents = contents_to_embed[i:i+batch_size]
                        batch_embeddings = self.dense_model.encode(batch_contents, show_progress_bar=False, convert_to_numpy=True)
                        all_new_embeddings_list.extend(batch_embeddings)
                    
                    for i, doc_id in enumerate(doc_ids_for_embedding):
                        self.doc_vectors_dense[doc_id] = all_new_embeddings_list[i]
                    logger.info(f"Dense embeddings updated for {len(doc_ids_for_embedding)} documents.")
                except Exception as e:
                    logger.error(f"Error generating dense embeddings: {e}. Some dense vectors might be missing or outdated.")
            
            existing_doc_ids = set(self.documents.keys())
            stale_dense_ids = set(self.doc_vectors_dense.keys()) - existing_doc_ids
            for stale_id in stale_dense_ids:
                del self.doc_vectors_dense[stale_id]
            if stale_dense_ids: logger.info(f"Removed {len(stale_dense_ids)} stale dense vectors.")
        else: 
            logger.warning("Dense model not available. Skipping dense vector generation/update.")
            self.doc_vectors_dense.clear() 

        self.initialized = True
        logger.info(f"Index build/update completed in {time.time() - start_time:.2f} seconds.")

    def save_index(self, path: str):
        if not self.initialized:
            logger.error("Cannot save index: Engine not initialized. Call build_index() first.")
            return
        
        dir_name = os.path.dirname(path)
        if dir_name: # Only call makedirs if dirname is not empty (i.e., path includes a directory)
            try:
                os.makedirs(dir_name, exist_ok=True)
            except OSError as e:
                logger.error(f"Error creating directory {dir_name} for index: {e}")
                return # Stop if directory cannot be created
        
        index_data = {
            "documents": self.documents, 
            "doc_vectors_sparse_terms": self.doc_vectors_sparse_terms,
            "doc_vectors_dense": self.doc_vectors_dense, 
            "code_docs": self.code_docs,
            "general_docs": self.general_docs,
            "bm25_k1": self.bm25.k1, 
            "bm25_b": self.bm25.b,
            "bm25_idf": self.bm25.idf,
            "bm25_doc_lengths": self.bm25.doc_lengths,
            "bm25_avg_doc_length": self.bm25.avg_doc_length,
            "bm25_document_count": self.bm25.document_count, 
            "vector_dim": self.vector_dim, 
            "dense_model_name": self.current_dense_model_name 
        }
        try:
            with open(path, 'wb') as f:
                pickle.dump(index_data, f)
            logger.info(f"Index successfully saved to {path}")
        except Exception as e:
            logger.error(f"Error saving index to {path}: {e}")

    def load_index(self, path: str):
        start_time = time.time()
        if not os.path.exists(path):
            logger.error(f"Index file '{path}' not found. Cannot load index.")
            return

        try:
            with open(path, 'rb') as f:
                index_data = pickle.load(f)
            
            self.documents = index_data.get("documents", {}) 
            self.doc_vectors_sparse_terms = index_data.get("doc_vectors_sparse_terms", {})
            self.doc_vectors_dense = index_data.get("doc_vectors_dense", {}) 
            self.code_docs = index_data.get("code_docs", set())
            self.general_docs = index_data.get("general_docs", set())
            
            self.bm25.k1 = index_data.get("bm25_k1", 1.2) 
            self.bm25.b = index_data.get("bm25_b", 0.75)   
            self.bm25.idf = index_data.get("bm25_idf", {})
            self.bm25.doc_lengths = index_data.get("bm25_doc_lengths", {})
            self.bm25.avg_doc_length = index_data.get("bm25_avg_doc_length", 0.0)
            self.bm25.document_count = index_data.get("bm25_document_count", 0)
            
            self.vector_dim = index_data.get("vector_dim", 0) 
            dense_model_name_from_index = index_data.get("dense_model_name", None)

            if dense_model_name_from_index:
                if not self.dense_model or self.current_dense_model_name != dense_model_name_from_index:
                    logger.info(f"Attempting to load dense model '{dense_model_name_from_index}' as specified in the index file.")
                    try:
                        self.dense_model = SentenceTransformer(dense_model_name_from_index)
                        loaded_dim = self.dense_model.get_sentence_embedding_dimension()
                        self.current_dense_model_name = dense_model_name_from_index
                        if self.vector_dim != 0 and self.vector_dim != loaded_dim:
                            logger.warning(f"Index vector dimension ({self.vector_dim}) differs from loaded model '{dense_model_name_from_index}' dimension ({loaded_dim}). Using model's dimension. Dense search results may be inconsistent if embeddings are not from this model.")
                        self.vector_dim = loaded_dim 
                        logger.info(f"Dense model '{dense_model_name_from_index}' loaded from index spec. Vector dim: {self.vector_dim}")
                    except Exception as e:
                        logger.error(f"Failed to load SentenceTransformer model '{dense_model_name_from_index}' specified in index: {e}. Dense search may be impaired or disabled.")
                        self.dense_model = None 
                        self.current_dense_model_name = None
                        self.doc_vectors_dense = {} 
                else:
                    logger.info(f"Current dense model '{self.current_dense_model_name}' matches index. No model reload needed.")
            elif self.dense_model: 
                 logger.warning(f"Index file does not specify a dense model, but engine was initialized with '{self.current_dense_model_name}'. Using this model. If index embeddings are from a different model, results will be inconsistent.")
            else: 
                 logger.info("No dense model specified in index and none initialized. Dense search remains disabled.")
                 self.dense_model = None
                 self.current_dense_model_name = None
                 self.doc_vectors_dense = {}

            self.initialized = True
            logger.info(f"Index loaded from '{path}' in {time.time() - start_time:.2f} seconds. Total {len(self.documents)} chunks/documents.")
        except Exception as e:
            logger.error(f"Critical error loading index from '{path}': {e}. Engine state might be inconsistent.")
            self.initialized = False 

    def detect_search_mode(self, query: str) -> SearchMode:
        query_lower = query.lower()
        
        code_indicators = {
            "keywords": ['python', 'javascript', 'java', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'golang', 'rust',
                         'function', 'class', 'method', 'module', 'import', 'export', 'def ', 'var ', 'let ', 'const ',
                         'error', 'exception', 'traceback', 'debug', 'bug', 'issue', 'fix', 'warning', 'fatal', 'segfault',
                         'api', 'sdk', 'library', 'framework', 'dependency', 'package', 'install', 'compile', 'build',
                         'algorithm', 'data structure', 'pointer', 'nullpointerexception', 'typeerror', 'valueerror',
                         'sql', 'select ', 'insert ', 'update ', 'delete ', 'database', 'query', 'server', 'client', 'http', 'tcp',
                         'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'cloud', 'lambda', 'serverless', 'github', 'gitlab', 'stackoverflow'],
            "patterns": [
                r'\w+\.\w+\(.*\)',  
                r'\w+\[.*?\]',      
                r'<\w+.*?>',        
                r'#\w+',            
                r'\.\w+',           
                r'[\{\}\(\)\[\];:]', 
                r'\S+Error:',       
                r'^\s*SELECT\s',    
                r'^\s*FROM\s',      
                r'\b(line \d+)\b',  
                r'[a-f0-9]{7,}',    
            ]
        }
        
        general_indicators = {
            "keywords": ['how to', 'what is', 'what are', 'why is', 'why does', 'explain', 'tutorial', 'guide', 'article', 'news', 'best', 'top', 'review',
                         'compare', 'difference', 'meaning', 'definition', 'example of', 'history', 'science', 'technology', 'business',
                         'benefits', 'disadvantages', 'impact of', 'future of', 'introduction to', 'overview of'],
            "patterns": [
                r'^\w+\s+\w+\?$', 
                r'^(who|what|where|when|why|how)\s', 
            ]
        }

        code_score = 0.0 
        general_score = 0.0

        for kw in code_indicators["keywords"]:
            if kw in query_lower: code_score += 1.0
            if len(kw) < 6 and re.search(r'\b' + re.escape(kw) + r'\b', query_lower): code_score += 0.5 
        for kw in general_indicators["keywords"]:
            if query_lower.startswith(kw) : general_score += 1.5 
            elif kw in query_lower: general_score += 1.0
            if len(kw) < 8 and re.search(r'\b' + re.escape(kw) + r'\b', query_lower): general_score += 0.5

        for pattern in code_indicators["patterns"]:
            if re.search(pattern, query, re.IGNORECASE): code_score += 1.5
        for pattern in general_indicators["patterns"]:
            if re.search(pattern, query_lower, re.IGNORECASE): general_score += 1.5
        
        if len(query_lower.split()) < 3:
            if code_score > 0 and general_score == 0 : code_score += 0.5 
            elif general_score > 0 and code_score == 0 : general_score += 0.5 
        
        if query.isupper() and len(query) > 4: 
            code_score += 1.2

        if '/' in query or '\\' in query or ('.' in query and query.count('.') >=2):
            code_score += 0.8

        logger.debug(f"Mode detection for '{query}': CodeScore={code_score:.1f}, GeneralScore={general_score:.1f}")
        
        if code_score == 0 and general_score == 0: return SearchMode.GENERAL 

        if code_score > general_score * 1.3 : 
            return SearchMode.CODE
        elif general_score > code_score * 1.3: 
            return SearchMode.GENERAL
        elif code_score > general_score: 
            if any(query_lower.startswith(gq) for gq in ['what is', 'how to', 'explain']) and code_score < general_score + 2.0:
                return SearchMode.GENERAL
            return SearchMode.CODE
        else: 
            return SearchMode.GENERAL

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores: return {}
        
        min_val = min(scores.values())
        max_val = max(scores.values())

        if max_val == min_val:
            return {doc_id: 0.5 if -0.001 < max_val < 0.001 else (1.0 if max_val > 0 else 0.0) for doc_id in scores} 

        normalized = {}
        effective_range = max_val - min_val
        if effective_range == 0: 
             return {doc_id: 0.5 for doc_id in scores}
        for doc_id, score in scores.items():
            norm_score = (score - min_val) / effective_range
            normalized[doc_id] = max(0, min(1, norm_score)) 
        return normalized

    def search(self, query: str, mode: SearchMode = SearchMode.AUTO,
               max_results: int = 10, min_score_threshold: float = 0.0001, 
               sparse_weight: float = 0.5, dense_weight: float = 0.5) -> List[SearchResult]:
        
        if not self.initialized:
            logger.error("Search engine index not initialized. Call build_index() or load_index() first.")
            return []
        if not self.documents: 
            logger.warning("No documents in the index to search.")
            return []
            
        start_time = time.time()

        actual_mode_used = mode
        if mode == SearchMode.AUTO:
            actual_mode_used = self.detect_search_mode(query)
        logger.info(f"Search query: '{query}', User Mode: {mode.name}, Actual Mode Used: {actual_mode_used.name}")

        if actual_mode_used == SearchMode.CODE: target_doc_ids = self.code_docs
        elif actual_mode_used == SearchMode.GENERAL: target_doc_ids = self.general_docs
        else: target_doc_ids = set(self.documents.keys()) 
        
        if not target_doc_ids:
            logger.warning(f"No documents available for the selected search mode '{actual_mode_used.name}'.")
            return []

        query_tokens_for_sparse = self.query_tokenizer.tokenize(query, remove_stopwords=True, min_length=2)
        
        sparse_scores: Dict[str, float] = {} 
        if query_tokens_for_sparse and sparse_weight > 0.001: 
            query_term_idfs = {term: self.bm25.idf.get(term, 0.0) for term in query_tokens_for_sparse if term in self.bm25.idf}

            if not query_term_idfs: 
                logger.info("All query terms for sparse search are Out-Of-Vocabulary for BM25.")
            else:
                for doc_id in target_doc_ids: 
                    doc_sparse_terms = self.doc_vectors_sparse_terms.get(doc_id)
                    if doc_sparse_terms is None: 
                        logger.debug(f"Missing sparse terms for doc_id: {doc_id}. Skipping for BM25.")
                        continue 
                    
                    bm25_score = self.bm25.score(query_term_idfs, doc_id, doc_sparse_terms)
                    if bm25_score > 1e-6: 
                        sparse_scores[doc_id] = bm25_score
        else:
            logger.info("Skipping sparse search (BM25) due to no query tokens or zero sparse_weight.")
        
        dense_scores: Dict[str, float] = {} 
        if self.dense_model and self.doc_vectors_dense and dense_weight > 0.001: 
            try:
                query_dense_vector = self.dense_model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].reshape(1, -1)
                dense_candidate_ids = [doc_id for doc_id in target_doc_ids if doc_id in self.doc_vectors_dense]
                
                if dense_candidate_ids:
                    doc_embeddings_matrix = np.array([self.doc_vectors_dense[doc_id] for doc_id in dense_candidate_ids])
                    
                    if doc_embeddings_matrix.ndim == 2 and doc_embeddings_matrix.shape[0] > 0:
                        similarities = cosine_similarity(query_dense_vector, doc_embeddings_matrix)[0] 
                        for i, doc_id in enumerate(dense_candidate_ids):
                            dense_scores[doc_id] = (similarities[i] + 1.0) / 2.0 
                    else: logger.debug("Dense embeddings matrix for candidates is not valid for similarity calculation.")
                else: logger.debug("No valid candidate document IDs (chunks) for dense search from the target set.")
            except Exception as e: logger.error(f"Error during dense search vectorization or similarity: {e}")
        else:
            logger.info("Skipping dense search due to no dense model/vectors or zero dense_weight.")

        rrf_k = 60 
        combined_scores_rrf: Dict[str, float] = Counter() 

        normalized_sparse = self._normalize_scores(sparse_scores)
        normalized_dense = self._normalize_scores(dense_scores)

        if normalized_sparse and sparse_weight > 0:
            sorted_sparse_ids = sorted(normalized_sparse.keys(), key=lambda id_key: normalized_sparse[id_key], reverse=True)
            for rank, doc_id in enumerate(sorted_sparse_ids):
                combined_scores_rrf[doc_id] += sparse_weight * (1.0 / (rrf_k + rank + 1)) 

        if normalized_dense and dense_weight > 0:
            sorted_dense_ids = sorted(normalized_dense.keys(), key=lambda id_key: normalized_dense[id_key], reverse=True)
            for rank, doc_id in enumerate(sorted_dense_ids):
                combined_scores_rrf[doc_id] += dense_weight * (1.0 / (rrf_k + rank + 1))
        
        if not combined_scores_rrf:
            logger.info(f"No combined scores generated for query '{query}'. Both sparse and dense searches might have yielded no results or had zero weight.")
            return []

        eligible_results_rrf = {
            doc_id: score for doc_id, score in combined_scores_rrf.items() if score >= min_score_threshold
        }
        
        if len(eligible_results_rrf) > max_results * 1.5 and max_results > 0: 
            top_n_items = heapq.nlargest(max_results, eligible_results_rrf.items(), key=lambda item: item[1])
        else: 
            sorted_ids_temp = sorted(eligible_results_rrf.keys(), key=lambda id_key: eligible_results_rrf[id_key], reverse=True)
            top_n_items = [(doc_id, eligible_results_rrf[doc_id]) for doc_id in sorted_ids_temp[:max_results]]
        
        final_results: List[SearchResult] = []
        query_tokens_for_highlighting = Tokenizer(mode=TokenizationMode.WORD).tokenize(query, remove_stopwords=False, min_length=1) 

        for doc_id, final_score in top_n_items:
            doc_obj = self.documents.get(doc_id) 
            if not doc_obj: 
                logger.warning(f"Document ID '{doc_id}' found in scores but not in main documents dictionary. Skipping.")
                continue 

            highlights = self._generate_highlights(doc_obj.content, query_tokens_for_highlighting)
            final_results.append(SearchResult(
                doc_id=doc_id, 
                score=final_score, 
                content=doc_obj.content, 
                metadata=doc_obj.metadata,
                doc_type=doc_obj.doc_type, 
                highlights=highlights,
                original_doc_id=doc_obj.original_doc_id or doc_obj.id 
            ))

        logger.info(f"Search for '{query}' (mode: {actual_mode_used.name}) found {len(final_results)} results in {time.time() - start_time:.4f} seconds.")
        return final_results


    def _generate_highlights(self, content: str, query_tokens: List[str],
                            context_size: int = 70, max_highlights: int = 3) -> List[str]: 
        if not query_tokens or not content: 
            return [content[:context_size*2] + "..." if len(content) > context_size*2 else content] if content else []

        content_lower = content.lower() 
        highlight_segments = [] 

        unique_query_tokens_for_match = sorted(
            list(set(t.lower() for t in query_tokens if t.strip() and (len(t) > 1 or not t.isalnum()))), 
            key=len, 
            reverse=True
        )
        if not unique_query_tokens_for_match: 
            return [content[:context_size*2] + "..." if len(content) > context_size*2 else content] if content else []

        match_positions: List[Tuple[int, int]] = [] 
        for token_lower in unique_query_tokens_for_match:
            start_offset = 0
            while True:
                pos = content_lower.find(token_lower, start_offset)
                if pos == -1: break
                match_positions.append((pos, pos + len(token_lower)))
                start_offset = pos + 1 
        
        if not match_positions:
             return [content[:context_size*2] + "..." if len(content) > context_size*2 else content] if content else []

        match_positions.sort() 

        if not match_positions : return [] 
        
        current_segment_start, current_segment_end = match_positions[0]
        for i in range(1, len(match_positions)):
            next_match_start, next_match_end = match_positions[i]
            if next_match_start < current_segment_end + context_size // 1.5 : 
                current_segment_end = max(current_segment_end, next_match_end)
            else: 
                highlight_segments.append((current_segment_start, current_segment_end))
                current_segment_start, current_segment_end = next_match_start, next_match_end
        highlight_segments.append((current_segment_start, current_segment_end)) 

        final_snippets: List[str] = []
        added_snippet_boundaries = set() 

        for seg_start, seg_end in highlight_segments:
            ctx_start = max(0, seg_start - context_size)
            ctx_end = min(len(content), seg_end + context_size)

            is_redundant = False
            current_snippet_len = ctx_end - ctx_start
            if current_snippet_len <=0: continue # Skip if segment becomes invalid

            for added_s, added_e in added_snippet_boundaries:
                overlap_start = max(ctx_start, added_s)
                overlap_end = min(ctx_end, added_e)
                overlap_len = overlap_end - overlap_start
                if overlap_len > 0:
                    # Check if segments are too similar (e.g. >70% overlap with the smaller of the two)
                    min_len_of_compared_segments = min(current_snippet_len, added_e - added_s)
                    if min_len_of_compared_segments > 0 and (overlap_len / min_len_of_compared_segments) > 0.7:
                        is_redundant = True
                        break
            if is_redundant and final_snippets: continue 

            added_snippet_boundaries.add((ctx_start, ctx_end))
            raw_snippet_text = content[ctx_start:ctx_end]
            
            temp_snippet_for_bolding = raw_snippet_text
            sorted_original_tokens_for_bolding = sorted(
                list(set(t for t in query_tokens if t.strip() and (len(t) > 1 or not t.isalnum()))),
                key=len,
                reverse=True
            )

            for token_to_bold in sorted_original_tokens_for_bolding: 
                try:
                    escaped_token = re.escape(token_to_bold)
                    pattern_for_bolding = r'\b(' + escaped_token + r')\b' if re.fullmatch(r'\w+', token_to_bold) else r'(' + escaped_token + r')'
                    
                    temp_snippet_for_bolding = re.sub(pattern_for_bolding, r'<b>\1</b>', temp_snippet_for_bolding, flags=re.IGNORECASE)
                except re.error: 
                    logger.debug(f"Regex error while trying to bold token: '{token_to_bold}'. Skipping.")
                    pass 

            prefix = "..." if ctx_start > 0 else ""
            suffix = "..." if ctx_end < len(content) else ""
            final_snippets.append(prefix + temp_snippet_for_bolding + suffix)
            
            if len(final_snippets) >= max_highlights: break 
        
        if not final_snippets:
            return [content[:context_size*2] + "..." if len(content) > context_size*2 else content] if content else []
            
        return list(dict.fromkeys(final_snippets)) 

class QueryProcessor: 
    def __init__(self, search_engine_ref: DualModeSearchEngine): 
        self.search_engine = search_engine_ref
        self.tokenizer = Tokenizer(mode=TokenizationMode.WORD) 

    def process_and_refine_query(self, query: str) -> Tuple[str, SearchMode]:
        normalized_query = ' '.join(query.lower().split()).strip()
        is_error_type_query = self._is_error_message_query(normalized_query)
        search_mode = self.search_engine.detect_search_mode(query) 
        
        refined_query = normalized_query
        if is_error_type_query and search_mode == SearchMode.CODE:
            pass 

        logger.debug(f"Processed query: '{refined_query}', Detected mode for this query: {search_mode.name}, IsErrorType: {is_error_type_query}")
        return refined_query, search_mode

    def _is_error_message_query(self, query_text: str) -> bool:
        error_keywords = [
            'error', 'exception', 'failed', 'undefined', 'null', 'nil', 'crash', 'panic',
            'traceback', 'stacktrace', 'syntax', 'fatal', 'warning', 'cannot', 'unable',
            'segmentation fault', 'core dumped', 'npe', 'issue #', 'bug id', 'errno', 'unhandled'
        ]
        if any(kw in query_text for kw in error_keywords):
            return True
        if re.search(r'\b([a-z_]+error|[a-z_]+exception):\s*', query_text, re.IGNORECASE): return True
        if re.search(r'\bat\s+([\w\$.<>/\\]+)\(([\w.-]+):(\d+)\)', query_text): return True 
        if re.search(r'File\s+".*?",\s+line\s+\d+', query_text): return True 
        return False

if __name__ == "__main__":
    logger.setLevel(logging.DEBUG) 
    logger.info("--- Running Search Engine Standalone Test Suite ---")
    
    engine_test = DualModeSearchEngine(dense_model_name=DEFAULT_DENSE_MODEL_NAME)

    doc1 = CodeDocument(id="py_avg", content="def average(nums):\n  return sum(nums)/len(nums) if nums else 0", language="python")
    doc2 = GeneralDocument(id="gen_py", content="Python is a popular language for data science and web development.")
    doc3 = CodeDocument(id="js_err", content="TypeError: Cannot read property 'value' of null in script.js at line 42", language="javascript") 
    doc4 = GeneralDocument(id="long_ethics", content="AI ethics involves considering moral implications of artificial intelligence. Bias in AI training data can lead to unfair outcomes. Job displacement is another concern.")
    doc5 = CodeDocument(id="html_sample", content="<body><h1>Hello World</h1><p>This is a sample HTML document.</p></body>") 
    
    engine_test.add_documents([doc1, doc2, doc3, doc4, doc5])
    logger.info(f"Added {len(engine_test.documents)} document chunks for testing.")
    
    engine_test.build_index()
    
    if not engine_test.initialized:
        logger.error("Engine initialization failed in test. Aborting further tests.")
    else:
        test_queries = {
            "python average function": SearchMode.AUTO, 
            "ai ethics and bias": SearchMode.AUTO,     
            "TypeError null javascript": SearchMode.CODE,
            "what is python": SearchMode.AUTO,
            "html body tag": SearchMode.AUTO
        }

        for test_query, test_mode in test_queries.items():
            logger.info(f"\n--- Testing Query: '{test_query}' (User Mode: {test_mode.name}) ---")
            results = engine_test.search(test_query, mode=test_mode, sparse_weight=0.5, dense_weight=0.5)
            if results:
                for res_idx, res in enumerate(results[:2]): 
                    logger.info(f"  Result {res_idx+1}: ID={res.doc_id}, Orig={res.original_doc_id}, Score={res.score:.4f}, Type={res.doc_type.name}, Lang={res.metadata.get('language')}")
                    logger.info(f"  Highlight: {res.highlights[0] if res.highlights else res.content[:80]}")
            else:
                logger.info("  No results found for this test query.")
        
        TEST_INDEX_FILE = "temp_standalone_test_index.pkl"
        engine_test.save_index(TEST_INDEX_FILE)
        
        engine_loaded = DualModeSearchEngine(dense_model_name=None) 
        logger.info(f"\n--- Testing Index Loading from '{TEST_INDEX_FILE}' ---")
        engine_loaded.load_index(TEST_INDEX_FILE)
        
        if engine_loaded.initialized and len(engine_loaded.documents) > 0:
            logger.info(f"Index loaded successfully. {len(engine_loaded.documents)} chunks.")
            results_after_load = engine_loaded.search("python average function", mode=SearchMode.CODE)
            assert len(results_after_load) > 0, "Search after loading index failed to return results."
            logger.info(f"  Search after load ('python average function') found: {results_after_load[0].doc_id if results_after_load else 'None'}")
        else:
            logger.error("Failed to load index or index was empty after load.")

        if os.path.exists(TEST_INDEX_FILE):
            try:
                os.remove(TEST_INDEX_FILE)
                logger.info(f"Cleaned up temporary index file: {TEST_INDEX_FILE}")
            except OSError as e_os:
                logger.error(f"Error removing temporary index file {TEST_INDEX_FILE}: {e_os}")

    logger.info("--- Search Engine Standalone Test Suite Finished ---")