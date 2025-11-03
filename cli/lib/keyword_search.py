import pickle
import string
import math

from pathlib import Path
from collections import Counter

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
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
        self.doc_length: dict[int, int] = {}
        self.index_path = CACHE_DIR / "index.pkl"
        self.docmap_path = CACHE_DIR / "docmap.pkl"
        self.tf_path = CACHE_DIR / "term_frequency.pkl"
        self.doc_length_path = CACHE_DIR / "doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        self.doc_length[doc_id] = len(tokens)

        if doc_id not in self.term_frequency:
            self.term_frequency[doc_id] = Counter()

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequency[doc_id][token] += 1
    
    def __get_avg_doc_length(self) -> float:
        """Compute and return the average document length."""
        if not self.doc_length:
            return 0.0
        
        total_length = sum(self.doc_length.values())
        return total_length / len(self.doc_length)

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
    
    def get_idf(self, term: str) -> str:
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token and not a stopwords")
        
        token = tokens[0]

        total_docs = len(self.docmap)
        doc_freq = len(self.get_documents(token))

        idf = math.log((total_docs + 1) / (doc_freq + 1))
        
        return idf
    
    def get_tf_idf(self, doc_id: int, term: str) -> str:
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        """Return BM25 TF for a given document and term."""
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1)) / (tf + k1 * ((1 - b) + b * (self.doc_length.get(doc_id, 0) / self.__get_avg_doc_length())))
    
    def get_bm25_idf(self, term: str) -> float:
        """Return BM25 IDF for a given term."""
        tokens = tokenize_text(term)
        if len(tokens) != 1:
            raise ValueError("Term must be a single token and not a stopwords")
        
        token = tokens[0]

        total_docs = len(self.docmap)
        doc_freq = len(self.get_documents(token))

        # MB25 IDF formula with Laplace smoothing
        idf = math.log((total_docs - doc_freq + 0.5) / (doc_freq + 0.5) + 1)
        
        return idf
    
    def bm25(self, doc_id: int, term: str) -> float:
        """Return BM25 score for a given document and term."""
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf

    def bm25_search(self, query: str, limit: int = 5) -> list[tuple[int, float]]:
        """Perform BM25 search across all documents for the given query."""
        tokens = tokenize_text(query)
        if not tokens:
            print("No tokens found in query.")
            return []
        
        scores: dict[int, float] = {}

        for doc_id in self.docmap.keys():
            total_score = 0.0
            for token in tokens:
                total_score += self.bm25(doc_id, token)
            if total_score > 0:
                scores[doc_id] = total_score
        
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_scores[:limit]

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

        with self.doc_length_path.open("wb") as f:
            pickle.dump(self.doc_length, f)
        
        print(f"Inverted index + term frequencies saved to {CACHE_DIR}")
    
    def load(self) -> None:
        """Load index and docmap from disk, raising if missing."""
        if not ( self.index_path.exists() and
                 self.docmap_path.exists() and
                 self.tf_path.exists() and
                 self.doc_length_path.exists()):
            raise FileNotFoundError("Index cache not found. Please run 'build' first.")
        
        with self.index_path.open("rb") as f:
            self.index = pickle.load(f)
        
        with self.docmap_path.open("rb") as f:
            self.docmap = pickle.load(f)
        
        with self.tf_path.open("rb") as f:
            self.term_frequency = pickle.load(f)

        with self.doc_length_path.open("rb") as f:
            self.doc_length = pickle.load(f)

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
   
    return inverted_index.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Index not found. Please run 'build' first.")
        return 0.0
           
    return inverted_index.get_tf_idf(doc_id, term)

def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
    """CLI helper: load index and compute BM25 TF for the given document and term."""
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Index not found. Please run 'build' first.")
        return 0.0
    
    return inverted_index.get_bm25_tf(doc_id, term, k1, b)

def bm25_idf_command(term: str) -> float:
    """CLI helper: load index and compute BM25 IDF for the given term."""
    inverted_index = InvertedIndex()
    try:
        inverted_index.load()
    except FileNotFoundError:
        print("Index not found. Please run 'build' first.")
        return 0.0
    
    return inverted_index.get_bm25_idf(term)

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