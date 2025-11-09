#!/usr/bin/env python3

import argparse
from lib.hybrid_search import normalize_scores, HybridSearch
from lib.search_utils import load_movies

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Normalize Command ---
    normalize_parser = subparsers.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="List of scores to normalize")

    # --- Weighted Search Command ---
    hybrid_parser = subparsers.add_parser("weighted-search", help="Compute hybrid score")
    hybrid_parser.add_argument("query", type=str, help="Query to search")
    hybrid_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 score")
    hybrid_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized = normalize_scores(args.scores)
            for score in normalized:
                print(f"* {score:.4f}")
        case "weighted-search":
            documents = load_movies()
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            
            for i, result in enumerate(results, 1):
                title = result['title']
                hybrid_score = result['score']
                bm25_score = result['bm25_score']
                semantic_score = result['semantic_score']
                description = result['description']
                
                # Truncate description to ~100 characters
                if len(description) > 100:
                    description = description[:97] + "..."
                
                print(f"{i}. {title}")
                print(f"   Hybrid Score: {hybrid_score:.3f}")
                print(f"   BM25: {bm25_score:.3f}, Semantic: {semantic_score:.3f}")
                print(f"   {description}")
                if i < len(results):  # Don't add extra newline after last result
                    print()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()