#!/usr/bin/env python3

import argparse

from lib.semantic_search import (
    verify_model, 
    embed_text,
    verify_embeddings,
    embed_query_text,
    semantic_search,
    chunk_text,
    semantic_chunk_text,
    embed_chunks
)
from lib.search_utils import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DEFAULT_SEMANTIC_CHUNK_SIZE

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

    # --- Chunk Command ---
    chunk_parser = subparsers.add_parser("chunk", help="Split text into fixed-size chunks")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Size of each chunk in words")
    chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Number of words to overlap between chunks")

    # --- Semantic Chunk Command ---
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text on sentence boundaries to preserve meaning")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=DEFAULT_SEMANTIC_CHUNK_SIZE, help="Maximum size of each chunk in sentences")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Number of sentences to overlap between chunks")

    # --- Embed Chunks Command ---
    subparsers.add_parser("embed_chunks", help="Generate embeddings for document chunks")

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
        case "chunk":
            chunk_text(args.text, args.chunk_size, args.overlap)
        case "semantic_chunk":
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
