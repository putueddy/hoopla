from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import DEFAULT_SEARCH_LIMIT

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.document_map = {str(doc['id']): doc for doc in documents}
        
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.inverted_index = InvertedIndex()
        if not self.inverted_index.index_path.exists():
            self.inverted_index.build()
            self.inverted_index.save()
        else:
            self.inverted_index.load()

    def _bm25_search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        self.inverted_index.load()
        return self.inverted_index.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha: float = 0.5, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        # Get 500x the limit to ensure we have enough results
        search_limit = limit * 500
        
        # Get BM25 results - need to convert format from tuples to dicts
        bm25_tuples = self._bm25_search(query, search_limit)
        bm25_results = [{'doc_id': str(doc_id), 'score': score} for doc_id, score in bm25_tuples]
        
        # Get semantic results
        semantic_results = self.semantic_search.search_chunks(query, search_limit)
        
        # Extract scores for normalization
        bm25_scores = [result['score'] for result in bm25_results]
        semantic_scores = [result['score'] for result in semantic_results]
        
        # Normalize scores
        normalized_bm25_scores = normalize_scores(bm25_scores)
        normalized_semantic_scores = normalize_scores(semantic_scores)
        
        # Create mappings with normalized scores
        doc_id_to_bm25_score = {result['doc_id']: normalized_bm25_scores[i] for i, result in enumerate(bm25_results)}
        doc_id_to_semantic_score = {str(result['id']): normalized_semantic_scores[i] for i, result in enumerate(semantic_results)}
        
        # Get all unique document IDs from both searches
        all_doc_ids = set(doc_id_to_bm25_score.keys()) | set(doc_id_to_semantic_score.keys())
        
        # Create hybrid results
        hybrid_results = []
        for doc_id in all_doc_ids:
            bm25_score = doc_id_to_bm25_score.get(doc_id, 0.0)
            semantic_score = doc_id_to_semantic_score.get(doc_id, 0.0)
            
            # Calculate hybrid score using the function
            combined_score = hybrid_score(bm25_score, semantic_score, alpha)
            
            # Get document info
            doc = self.document_map.get(doc_id)
            if doc:
                hybrid_results.append({
                    'doc_id': doc_id,
                    'title': doc.get('title', ''),
                    'description': doc.get('description', ''),
                    'score': combined_score,
                    'bm25_score': bm25_score,
                    'semantic_score': semantic_score
                })
        
        # Sort by hybrid score in descending order
        hybrid_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top results
        return hybrid_results[:limit]
    
    def rrf_search(self, query: str, k: int = 1, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores: list[float]) -> list[float]:    
    if not scores:
        return []
    
    max_score = max(scores)
    min_score = min(scores)

    if min_score == max_score:
        return [1.0] * len(scores)
    
    return [(score - min_score) / (max_score - min_score) for score in scores]

def hybrid_score(bm25_score: float, semantic_score: float, alpha: float = 0.5) -> float:
    return alpha * bm25_score + (1 - alpha) * semantic_score
    
def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)
