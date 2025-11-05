#!/usr/bin/env python3
"""
Script to split SimpleQA into train/test sets.
Ensures data generation only uses training set.
"""

import json
import random
import argparse
from pathlib import Path

def split_dataset(
    input_file: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
):
    """
    Split dataset into train and test sets.

    Args:
        input_file: Path to input JSONL file
        output_dir: Output directory for splits
        train_ratio: Ratio of training data (default 0.8 = 80/20 split)
        seed: Random seed for reproducibility
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Splitting dataset: {input_file}")
    print(f"Train ratio: {train_ratio}")
    print(f"Random seed: {seed}\n")

    # Load all examples
    examples = []
    with open(input_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line))

    total = len(examples)
    print(f"Total examples: {total}")

    # Shuffle with fixed seed
    random.seed(seed)
    random.shuffle(examples)

    # Split
    train_size = int(total * train_ratio)
    train_examples = examples[:train_size]
    test_examples = examples[train_size:]

    print(f"Train examples: {len(train_examples)}")
    print(f"Test examples: {len(test_examples)}\n")

    # Save train set
    train_file = output_path / "simpleqa_train.jsonl"
    with open(train_file, 'w') as f:
        for example in train_examples:
            f.write(json.dumps(example) + '\n')
    print(f"✓ Train set saved: {train_file}")

    # Save test set
    test_file = output_path / "simpleqa_test.jsonl"
    with open(test_file, 'w') as f:
        for example in test_examples:
            f.write(json.dumps(example) + '\n')
    print(f"✓ Test set saved: {test_file}")

    # Save split info
    split_info = {
        "total_examples": total,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "train_ratio": train_ratio,
        "seed": seed,
        "train_file": str(train_file),
        "test_file": str(test_file),
    }

    info_file = output_path / "split_info.json"
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"✓ Split info saved: {info_file}")

    return train_file, test_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train/test")
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/splits",
        help="Output directory"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training data ratio (default: 0.8)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    split_dataset(
        input_file=args.input_file,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
