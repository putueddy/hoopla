#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model, 
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
)

def _truncate(text: str, max_len: int = 120) -> str:
    return (text[:max_len] + "...") if len(text) > max_len else text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- Verify Command ---
    subparsers.add_parser("verify", help="Verify that the embedding model is loaded")

    # --- Embed Text Command ---
    single_embed_parser = subparsers.add_parser(
        "embed_text", help="Generate an embedding for a single text"
    )
    single_embed_parser.add_argument("text", type=str, help="Text to embed")

    # --- Verify Embeddings Command ---
    subparsers.add_parser(
        "verify_embeddings", help="Verify that the embeddings are loaded"
    )

    # --- Embed Query Text Command ---
    eq_parser = subparsers.add_parser("embedquery", help="Embed a user query")
    eq_parser.add_argument("query", type=str, help="Query to embed")

    # --- Search Command ---
    search_parser = subparsers.add_parser("search", help="Search for movies")
    search_parser.add_argument("query", type=str, help="Query to search for")
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            semantic_search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
