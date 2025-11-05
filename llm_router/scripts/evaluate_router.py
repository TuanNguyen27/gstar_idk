#!/usr/bin/env python3
"""
Script to evaluate trained router on SimpleQA benchmark.
"""

import argparse
import os
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deployment import RouterInference
from models import GeminiClient
from evaluation import SimpleQAEvaluator

def main():
    parser = argparse.ArgumentParser(description="Evaluate router on SimpleQA")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained router model"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to SimpleQA test JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./evaluation/results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-2b",
        help="Base model name"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of test examples"
    )
    parser.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Gemini API key"
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run baseline comparison"
    )
    parser.add_argument(
        "--baseline-model",
        type=str,
        default="gemini-2.5-flash",
        help="Model for baseline comparison"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gemini_api_key = args.gemini_api_key or os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        print("Error: Gemini API key required")
        sys.exit(1)

    print(f"=== Router Evaluation on SimpleQA ===")
    print(f"Router model: {args.model_path}")
    print(f"Test file: {args.test_file}")
    print(f"Limit: {args.limit or 'None'}\n")

    # Initialize components
    print("Loading router and Gemini client...")
    router = RouterInference(
        model_path=args.model_path,
        base_model_name=args.base_model,
    )
    gemini_client = GeminiClient(api_key=gemini_api_key)

    evaluator = SimpleQAEvaluator(
        router=router,
        gemini_client=gemini_client,
    )

    # Run evaluation
    print("\n=== Router Evaluation ===")
    results_file = output_dir / "router_results.jsonl"
    metrics = evaluator.evaluate(
        test_file=Path(args.test_file),
        output_file=results_file,
        limit=args.limit,
    )

    print("\n=== Router Metrics ===")
    print(json.dumps(metrics, indent=2))

    # Save metrics
    metrics_file = output_dir / "router_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_file}")

    # Baseline comparison
    if args.compare_baseline:
        print(f"\n=== Baseline Comparison ({args.baseline_model}) ===")
        baseline_metrics = evaluator.compare_with_baseline(
            test_file=Path(args.test_file),
            baseline_model=args.baseline_model,
            limit=args.limit,
        )

        print(json.dumps(baseline_metrics, indent=2))

        # Combined comparison
        print("\n=== Summary ===")
        print(f"Router Accuracy: {metrics['accuracy']}%")
        print(f"Baseline Accuracy: {baseline_metrics['baseline_accuracy']}%")
        print(f"Router Cost: ${metrics['cost']['total_estimated']:.4f}")
        print(f"Baseline Cost: ${baseline_metrics['baseline_cost']:.4f}")
        print(f"Cost Savings: {metrics['cost']['savings_percentage']}%")

        # Save comparison
        comparison = {
            "router": metrics,
            "baseline": baseline_metrics,
        }
        comparison_file = output_dir / "comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {comparison_file}")

if __name__ == "__main__":
    main()
