import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from .search_utils import CACHE_DIR, load_movies

# ------------------- Add this outside the class -----------------------------
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)
# ----------------------------------------------------------------------------

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings = None
        self.documents = None
        self.document_map = {}
        self.embeddings_path = CACHE_DIR / "movie_embeddings.npy"

    def generate_embedding(self, text: str) -> np.ndarray:
        if len(text) == 0 or text.strip() == "":
            raise ValueError("Text must not be empty")

        return self.model.encode([text])[0]

    def build_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        movie_text = [
            f"{doc.get('title', '')}: {doc.get('description', '')}"
            for doc in documents
        ]

        self.embeddings = self.model.encode(movie_text, show_progress_bar=True)

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(self.embeddings_path, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        if self.embeddings_path.exists():
            self.embeddings = np.load(self.embeddings_path)

            if len(self.embeddings) == len(documents):
                return self.embeddings
            else:
                return self.build_embeddings(documents)
        else:
            return self.build_embeddings(documents)
    
    def search(self, query: str, limit: int = 5) -> list[dict]:
        if self.embeddings is None or self.documents is None:
            raise ValueError("No embeddings loaded. Call verify_embeddings command first.")

        query_embedding = self.generate_embedding(query)
        
        scores_and_docs = []
        for emb, doc in zip(self.embeddings, self.documents):
            score = cosine_similarity(emb, query_embedding)
            scores_and_docs.append((score, doc))

        scores_and_docs.sort(key=lambda x: x[0], reverse=True)
        top = scores_and_docs[:max(0, limit)]

        result = []
        for score, doc in top:
            result.append({
                "score": float(score),
                "title": doc.get("title", ""),
                "description": doc.get("description", ""),
            })
        
        return result

def verify_model() -> None:
    """Initialize the SemanticSearch model and print its configuration."""
    try:
        search_instance = SemanticSearch()
    except Exception as e:
        print(f"Failed to load model: {e}")
        exit(1)

    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")

def embed_text(text: str) -> None:
    try:
        search_instance = SemanticSearch()
        embedding = search_instance.generate_embedding(text)

        print(f"Text: {text}")
        print(f"First 3 dimensions: {embedding[:3]}")
        print(f"Dimensions: {embedding.shape[0]}")
    except Exception as e:
        print(f"Failed to generate embedding: {e}")

def verify_embeddings() -> None:
    try:
        search_instance = SemanticSearch()
        documents = load_movies()

        if not documents:
            return

        embeddings = search_instance.load_or_create_embeddings(documents)

        print(f"Number of documents: {len(documents)}")
        print(f"embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
    except Exception as e:
        print(f"Failed to load movies: {e}")

def embed_query_text(query: str) -> None:
    try:
        search_instance = SemanticSearch()
        embedding = search_instance.generate_embedding(query)

        print(f"Query: {query}")
        print(f"First 5 dimensions: {embedding[:5]}")
        print(f"Shape: {embedding.shape}")
    except Exception as e:
        print(f"Failed to embed query: {e}")
