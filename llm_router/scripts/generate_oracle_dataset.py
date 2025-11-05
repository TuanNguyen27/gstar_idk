#!/usr/bin/env python3
"""
Script to generate oracle dataset from benchmark data.
"""

import argparse
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import with absolute path
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.benchmark_loader import BenchmarkLoader
from data.oracle_generator import OracleDatasetGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate oracle dataset")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["simpleqa", "natural_questions", "custom"],
        help="Benchmark dataset name"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Path to benchmark JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/generated",
        help="Output directory for generated data"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to process"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for saving intermediate results"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Error: Gemini API key not provided")
        sys.exit(1)

    print(f"=== Oracle Dataset Generation ===")
    print(f"Benchmark: {args.benchmark}")
    print(f"Input: {args.input_file}")
    print(f"Output: {output_dir}")
    print(f"Limit: {args.limit or 'None'}\n")

    # Step 1: Load benchmark data
    print("Step 1: Loading benchmark data...")
    examples = BenchmarkLoader.load_benchmark(
        benchmark_name=args.benchmark,
        file_path=args.input_file,
        limit=args.limit,
    )
    print(f"Loaded {len(examples)} examples\n")

    # Step 2: Generate model answers
    print("Step 2: Generating model answers...")
    generator = OracleDatasetGenerator(gemini_api_key=gemini_api_key)

    model_answers_file = output_dir / "model_answers.jsonl"
    generator.generate_model_answers(
        benchmark_examples=examples,
        output_file=model_answers_file,
        batch_size=args.batch_size,
    )
    print(f"Model answers saved to {model_answers_file}\n")

    # Step 3: Judge and label
    print("Step 3: Judging answers and applying oracle matrix...")
    oracle_file = output_dir / "oracle_dataset.jsonl"
    oracle_examples = generator.judge_and_label(
        model_answers_file=model_answers_file,
        output_file=oracle_file,
        batch_size=args.batch_size,
    )
    print(f"Oracle dataset saved to {oracle_file}\n")

    print("=== Generation Complete ===")
    print(f"Total examples: {len(oracle_examples)}")
    print(f"Output: {oracle_file}")

if __name__ == "__main__":
    main()
