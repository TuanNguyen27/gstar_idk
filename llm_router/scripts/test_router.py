#!/usr/bin/env python3
"""
Script to test trained router with sample queries.
"""

import argparse
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deployment import RouterInference, EndToEndPipeline

def main():
    parser = argparse.ArgumentParser(description="Test trained router")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained router model"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-2b",
        help="Base model name"
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Single query to test"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default=None,
        help="File with test queries (one per line)"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key for end-to-end testing"
    )
    parser.add_argument(
        "--end-to-end",
        action="store_true",
        help="Run end-to-end pipeline (routing + answer generation)"
    )

    args = parser.parse_args()

    print(f"=== Router Testing ===")
    print(f"Model: {args.model_path}")
    print(f"Base: {args.base_model}\n")

    # Initialize router
    router = RouterInference(
        model_path=args.model_path,
        base_model_name=args.base_model,
    )

    # Test queries
    if args.query:
        queries = [args.query]
    elif args.test_file:
        with open(args.test_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]
    else:
        # Default test queries
        queries = [
            "What is 2 + 2?",
            "What is the wingspan of a Pteranodon?",
            "What is the meaning of life according to quantum physics?",
            "How many angels can dance on the head of a pin?",
        ]

    # End-to-end or routing only
    if args.end_to_end:
        gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            print("Error: Gemini API key required for end-to-end testing")
            sys.exit(1)

        pipeline = EndToEndPipeline(router=router, gemini_api_key=gemini_api_key)

        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            result = pipeline.answer_query(query, verbose=True)
            print(f"[Answer]\n{result['answer']}\n")

    else:
        # Routing only
        for query in queries:
            print(f"\n{'='*60}")
            print(f"Query: {query}")
            print('='*60)
            policy, model, reasoning = router.route_query(query)
            print(f"Policy: {policy}")
            print(f"Model: {model}")
            print(f"Reasoning: {reasoning}\n")

if __name__ == "__main__":
    main()
