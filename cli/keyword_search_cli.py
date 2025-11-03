#!/usr/bin/env python3

import argparse
from email.policy import default

from lib.keyword_search import (
    bm25_idf_command,
    bm25_tf_command,
    build_command, 
    search_command,
    tf_command,
    idf_command,
    tfidf_command,
    InvertedIndex,
)

from lib.search_utils import BM25_K1, BM25_B, DEFAULT_SEARCH_LIMIT

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Search Command ---
    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # --- Build Command --- 
    subparsers.add_parser("build", help="Build inverted index")

    # --- TF Command ---
    tf_parser = subparsers.add_parser("tf", help="Term frequency")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")

    # --- IDF Command ---
    idf_parser = subparsers.add_parser("idf", help="Inverse document frequency")
    idf_parser.add_argument("term", type=str, help="Term")

    # --- TFIDF Command ---
    tfidf_parser = subparsers.add_parser("tfidf", help="Term frequency inverse document frequency")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    # --- BM25 IDF Command ---
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    # --- BM25 TF Command ---
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs="?", default=BM25_K1, help="Tunable BM25 k1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")

    # --- BM25 Search Command ---
    bm25_search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25_search_parser.add_argument("query", type=str, help="Search query")
    bm25_search_parser.add_argument("--limit", type=int, default=DEFAULT_SEARCH_LIMIT, help="Number of top results to return")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()

        case "search":
            print("Searching for:", args.query)
            results = search_command(args.query)
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result['id']}] {result['title']}")
        
        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term '{args.term}' appears {tf} time(s) in document {args.doc_id}")

        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tf_idf = tfidf_command(args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25_idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        
        case "bm25tf":
            bm25_tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25_tf:.2f}")
        
        case "bm25search":
            inverted_index = InvertedIndex()

            try:
                inverted_index.load()
            except FileNotFoundError:
                print("Index not found. Please run 'build' first.")
                return []
            
            bm25_search = inverted_index.bm25_search(args.query, args.limit)

            if not bm25_search:
                print("No results found.")
                return
            for i, (doc_id, score) in enumerate(bm25_search, 1):
                movie = inverted_index.docmap.get(doc_id)
                title = movie.get("title", "")
                print(f"{i}. ({doc_id}) {title} - Score: {score:.2f}")
        
        case _:
            parser.exit(2, parser.format_help())

if __name__ == "__main__":
    main()
