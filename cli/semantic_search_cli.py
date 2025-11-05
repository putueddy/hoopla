#!/usr/bin/env python3

import argparse

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()
