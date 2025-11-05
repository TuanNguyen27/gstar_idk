#!/usr/bin/env python3
"""
Script to download SimpleQA benchmark from OpenAI.
SimpleQA: https://github.com/openai/simple-evals
"""

import json
import urllib.request
import ssl
from pathlib import Path

def download_simpleqa(output_dir: str = "./data/benchmarks"):
    """
    Download SimpleQA benchmark dataset.

    Args:
        output_dir: Directory to save the dataset
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # SimpleQA dataset URL (from OpenAI's simple-evals repo)
    url = "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test.jsonl"

    output_file = output_path / "simpleqa.jsonl"

    print(f"Downloading SimpleQA from: {url}")
    print(f"Saving to: {output_file}")

    try:
        # Handle SSL certificate issues
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        urllib.request.urlretrieve(url, output_file)
        print(f"✓ Download complete!")

        # Verify and show stats
        with open(output_file, 'r') as f:
            lines = f.readlines()
            count = len(lines)

        print(f"\nDataset statistics:")
        print(f"  Total examples: {count}")

        # Show first example
        if count > 0:
            first = json.loads(lines[0])
            print(f"\nFirst example:")
            print(f"  Problem: {first.get('problem', 'N/A')}")
            print(f"  Answer: {first.get('answer', 'N/A')}")

        return str(output_file)

    except Exception as e:
        print(f"Error downloading SimpleQA: {e}")
        print("\nAlternative: Download manually from:")
        print("https://github.com/openai/simple-evals")
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download SimpleQA benchmark")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data/benchmarks",
        help="Output directory"
    )

    args = parser.parse_args()
    download_simpleqa(args.output_dir)
