import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    load_movies,
)

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
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
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
    
    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
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

def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> None:
    try:
        docs = load_movies()
        search_instance = SemanticSearch()
        search_instance.load_or_create_embeddings(docs)

        results = search_instance.search(query, limit)

        print(f"\nQuery: {query}\n")
        print(f"Top {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (score: {result['score']:.4f})")
            print(f"   {result['description'][:100]}...\n")

    except Exception as e:
        print(f"Failed to perform semantic search: {e}")

def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.strip().split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        chunk = words[start:start + chunk_size]
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
        start += chunk_size - overlap
    
    return chunks

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    try:
        chunks = fixed_size_chunking(text, chunk_size, overlap)
        print(f"Chunking {len(text)} characters")
        for i, chunk in enumerate(chunks, 1):
            print(f"{i}: {chunk}")
    except Exception as e:
        print(f"Failed to chunk text: {e}")

def semantic_chunking(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return []

    chunks = []
    start = 0
    while start < len(sentences):
        chunk = sentences[start:start + max_chunk_size]
        if chunks and len(chunk) <= overlap:
            break
        chunks.append(" ".join(chunk))
        start += max_chunk_size - overlap
    
    return chunks

def semantic_chunk_text(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    try:
        chunks = semantic_chunking(text, max_chunk_size, overlap)
        print(f"Semantically chunking {len(text)} characters")
        for i, chunk in enumerate(chunks, 1):
            print(f"{i}: {chunk}")
    except Exception as e:
        print(f"Failed to semantic chunk text: {e}")


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        # Populate documents and document_map
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        # Create lists to hold all chunks and their metadata
        all_chunks = []
        chunk_metadata = []

        # Process each document
        for movie_idx, doc in enumerate(documents):
            description = doc.get('description', '')
            
            # Skip if description is empty
            if not description or description.strip() == '':
                continue
            
            # Use semantic chunking with 4-sentence chunks and 1-sentence overlap
            chunks = semantic_chunking(description, max_chunk_size=4, overlap=1)
            
            # Add each chunk and its metadata
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": movie_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                })

        # Generate embeddings for all chunks
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        # Save to cache
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({
                "chunks": chunk_metadata,
                "total_chunks": len(all_chunks)
            }, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        # Populate documents and document_map
        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        # Check if cache files exist
        if CHUNK_EMBEDDINGS_PATH.exists() and CHUNK_METADATA_PATH.exists():
            # Load embeddings
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            
            # Load metadata
            with open(CHUNK_METADATA_PATH, "r") as f:
                metadata_json = json.load(f)
                self.chunk_metadata = metadata_json["chunks"]
            
            return self.chunk_embeddings
        else:
            # Build embeddings if cache doesn't exist
            return self.build_chunk_embeddings(documents)


def embed_chunks() -> None:
    """Load movie documents, initialize ChunkedSemanticSearch, and create chunk embeddings."""
    try:
        # Load movie documents
        documents = load_movies()
        
        if not documents:
            print("No documents found")
            return
        
        # Initialize ChunkedSemanticSearch instance
        chunked_search = ChunkedSemanticSearch()
        
        # Load or build chunk embeddings
        embeddings = chunked_search.load_or_create_chunk_embeddings(documents)
        
        # Print info about the embeddings
        print(f"Generated {len(embeddings)} chunked embeddings")
        
    except Exception as e:
        print(f"Failed to create chunk embeddings: {e}")

        