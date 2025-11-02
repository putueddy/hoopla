#!/usr/bin/env python3

import argparse

from lib.keyword_search import build_command, search_command, tf_command, idf_command

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

        case _:
            parser.exit(2, parser.format_help())

if __name__ == "__main__":
    main()
