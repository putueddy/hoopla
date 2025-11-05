import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def generate_embedding(self, text: str) -> np.ndarray:
        if len(text) == 0 or text.strip() == "":
            raise ValueError("Text must not be empty")

        return self.model.encode([text])[0]

    
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