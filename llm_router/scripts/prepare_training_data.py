#!/usr/bin/env python3
"""
Script to prepare SFT + DPO training data from oracle dataset.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import DPODataPreparation

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT + DPO training data")
    parser.add_argument(
        "--oracle-file",
        type=str,
        required=True,
        help="Path to oracle dataset JSONL"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/training",
        help="Output directory for training data"
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    oracle_file = Path(args.oracle_file)

    print(f"=== Training Data Preparation ===")
    print(f"Oracle dataset: {oracle_file}")
    print(f"Output directory: {output_dir}\n")

    # Analyze distribution
    print("Analyzing dataset distribution...")
    stats = DPODataPreparation.analyze_dataset_distribution(oracle_file)
    print(f"\nDataset Statistics:")
    print(f"Total examples: {stats['total_examples']}")
    for policy, info in stats['policy_distribution'].items():
        print(f"  {policy}: {info['count']} ({info['percentage']}%)")
    print()

    # Create DPO dataset
    print("Creating DPO training dataset...")
    dpo_file = output_dir / "dpo_train.jsonl"
    DPODataPreparation.convert_oracle_to_dpo(
        oracle_file=oracle_file,
        output_file=dpo_file,
    )
    print(f"DPO dataset saved to {dpo_file}\n")

    # Create SFT dataset
    print("Creating SFT training dataset...")
    sft_file = output_dir / "sft_train.jsonl"
    DPODataPreparation.create_sft_dataset(
        dpo_file=dpo_file,
        output_file=sft_file,
    )
    print(f"SFT dataset saved to {sft_file}\n")

    print("=== Preparation Complete ===")
    print(f"SFT data: {sft_file}")
    print(f"DPO data: {dpo_file}")

if __name__ == "__main__":
    main()
