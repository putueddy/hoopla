import pickle
import string
import math

from pathlib import Path
from collections import Counter

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stop_words,
)

from nltk.stem import PorterStemmer

# Initialize global variables after functions are defined
STOP_WORDS = load_stop_words()
STEMMER = PorterStemmer()

class InvertedIndex:
    """Simple inverted index for keyword-based retrieval."""

    def __init__(self) -> None:
        self.index: dict[str, set[int]] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequency: dict[int, Counter[str]] = {}
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"
        self.tf_path = CACHE_DIR / "term_frequency.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)

        if doc_id not in self.term_frequency:
            self.term_frequency[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequency[doc_id][token] += 1
    
    def get_documents(self, term: str) -> list[int]:
        """Return sorted document IDs for a given token."""
        term = term.lower()
        docs = self.index.get(term, set())
        return sorted(docs)
    
    def get_tf(self, doc_id: int, term: str) -> int:
        """Return term frequency for a given document and term."""
        term = tokenize_text(term)
        if len(term) != 1:
            raise ValueError("Term must be a single token and not a stopwords")
        
        token = term[0]
        if doc_id not in self.term_frequency:
            return 0

        return self.term_frequency[doc_id].get(token, 0)
    
    def build(self) -> None:
        """Build inverted index and docmap from all movies."""
        movies = load_movies()
        for movie in movies:
            doc_id = int(movie['id'])
            self.docmap[doc_id] = movie
            text = f"{movie.get('title', '')} {movie.get('description', '')}"
            self.__add_document(doc_id, text)
    
    def save(self) -> None:
        """Persist index and docmap to cache directory."""
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        with self.index_path.open("wb") as f:
            pickle.dump(self.index, f)
        
        with self.docmap_path.open("wb") as f:
            pickle.dump(self.docmap, f)

        with self.tf_path.open("wb") as f:
            pickle.dump(self.term_frequency, f)
        
        print(f"Inverted index + term frequencies saved to {CACHE_DIR}")
    
    def load(self) -> None:
        """Load index and docmap from disk, raising if missing."""
        if not ( self.index_path.exists() and
                 self.docmap_path.exists() and
                 self.tf_path.exists()):
            raise FileNotFoundError("Index cache not found. Please run 'build' first.")
        
        with self.index_path.open("rb") as f:
            self.index = pickle.load(f)
        
        with self.docmap_path.open("rb") as f:
            self.docmap = pickle.load(f)
        
        with self.tf_path.open("rb") as f:
            self.term_frequency = pickle.load(f)

def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

def tf_command(doc_id: int, term: str) -> int:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Index not found. Please run 'build' first.")
        return 0
    
    return inverted_index.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    inverted_index = InvertedIndex()
    
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Index not found. Please run 'build' first.")
        return 0.0
    
    token = tokenize_text(term)
    if len(token) != 1:
        raise ValueError("Term must be a single token and not a stopwords")
    
    token = token[0]
    
    total_docs = len(inverted_index.docmap)
    doc_freq = len(inverted_index.get_documents(token))

    idf = math.log((total_docs + 1) / (doc_freq + 1))
    
    return idf

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    inverted_index = InvertedIndex()
    
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Index not found. Please run 'build' first.")
        return []
    
    preprocess_query = tokenize_text(query)
    if not preprocess_query:
        print("No valid tokens to search for")
        return []

    seen, results = set(), []
    for token_query in preprocess_query:
        for id in inverted_index.get_documents(token_query):
            if id in seen:
                continue
            seen.add(id)
            results.append(inverted_index.docmap[id])
            if len(results) >= limit:
                return results
    results.sort(key=lambda x: x['id'])
    return results[:limit]

def preprocess_text(text: str) -> str:
    """Preprocess text by lowercasing and removing punctuation."""
    text = text.lower()
    text = text.translate(str.maketrans("","", string.punctuation))
    return text

def tokenize_text(text: str) -> list[str]:
    """Tokenize text, remove stopwords, and apply stemming."""
    text = preprocess_text(text)
    return [STEMMER.stem(word) for word in text.split() if word not in STOP_WORDS]