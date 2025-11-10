"""
Generate SFT training data with performance-based continuous confidence labels.

Uses k-NN on question embeddings to assign confidence based on historical
accuracy on similar questions. This approach:
- Uses ground truth correctness (no bootstrapping)
- Model-specific calibration
- Works with 1-shot factual questions (no CoT needed)
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse


def load_judged_data(filepath: Path) -> Tuple[List[str], List[str], List[bool]]:
    """Load questions, answers, and correctness from judged dataset."""
    questions = []
    answers = []
    correctness = []

    with open(filepath, 'r') as f:
        for line in f:
            ex = json.loads(line)
            questions.append(ex['problem'])
            answers.append(ex['model_answer'])
            correctness.append(ex['is_correct'])

    return questions, answers, correctness


def generate_knn_confidence_labels(
    questions: List[str],
    correctness: List[bool],
    k_neighbors: int = 20,
    batch_size: int = 256,
) -> List[float]:
    """
    Generate continuous confidence labels using k-NN on question embeddings.

    Args:
        questions: List of question strings
        correctness: List of ground truth correctness (True/False)
        k_neighbors: Number of nearest neighbors to consider
        batch_size: Batch size for embedding computation

    Returns:
        List of confidence labels (0.0-1.0)
    """
    print(f"\n🔍 Computing embeddings for {len(questions)} questions...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(questions, batch_size=batch_size, show_progress_bar=True)

    print(f"\n📊 Computing similarity matrix...")
    similarity_matrix = cosine_similarity(embeddings)

    # Exclude self-similarity
    np.fill_diagonal(similarity_matrix, -1)

    print(f"\n🎯 Finding {k_neighbors} nearest neighbors for each question...")
    top_k_indices = np.argsort(similarity_matrix, axis=1)[:, -k_neighbors:]

    print(f"\n💯 Computing confidence labels...")
    correctness_array = np.array(correctness, dtype=float)
    confidence_labels = []

    for i in tqdm(range(len(questions))):
        neighbor_indices = top_k_indices[i]
        neighbor_correctness = correctness_array[neighbor_indices]
        neighbor_accuracy = neighbor_correctness.mean()

        # Base confidence from neighbor accuracy
        # Map [0, 1] accuracy → [0.1, 0.9] confidence
        base_confidence = 0.1 + 0.8 * neighbor_accuracy

        # Adjust based on current example
        if correctness[i]:
            # Correct answer: boost confidence slightly
            target_confidence = min(0.95, base_confidence + 0.1)
        else:
            # Incorrect answer: reduce confidence more aggressively
            target_confidence = max(0.05, base_confidence - 0.15)

        confidence_labels.append(target_confidence)

    return confidence_labels


def generate_sft_training_data(
    questions: List[str],
    answers: List[str],
    confidence_labels: List[float],
    output_path: Path,
):
    """
    Generate SFT training data with continuous confidence labels.

    Format:
    {
        "input": "Question: {question}\n\nProvide your answer and confidence (0.0-1.0):",
        "output": "Answer: {answer}\nConfidence: {confidence:.2f}"
    }
    """
    print(f"\n📝 Generating SFT training data...")

    training_examples = []
    for question, answer, confidence in tqdm(zip(questions, answers, confidence_labels)):
        example = {
            "input": f"Question: {question}\n\nProvide your answer and confidence (0.0-1.0):",
            "output": f"Answer: {answer}\nConfidence: {confidence:.2f}"
        }
        training_examples.append(example)

    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for ex in training_examples:
            f.write(json.dumps(ex) + '\n')

    print(f"\n✅ Saved {len(training_examples)} training examples to {output_path}")

    # Print statistics
    confidences = np.array(confidence_labels)
    print(f"\n📊 Confidence Statistics:")
    print(f"  Mean: {confidences.mean():.3f}")
    print(f"  Std:  {confidences.std():.3f}")
    print(f"  Min:  {confidences.min():.3f}")
    print(f"  Max:  {confidences.max():.3f}")
    print(f"  Median: {np.median(confidences):.3f}")

    # Distribution by quartiles
    quartiles = np.percentile(confidences, [25, 50, 75])
    print(f"\n  Q1 (25%): {quartiles[0]:.3f}")
    print(f"  Q2 (50%): {quartiles[1]:.3f}")
    print(f"  Q3 (75%): {quartiles[2]:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate performance-based continuous confidence SFT data"
    )
    parser.add_argument(
        "--judged-data",
        type=str,
        required=True,
        help="Path to judged dataset (with is_correct labels)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save SFT training data",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=20,
        help="Number of nearest neighbors for k-NN (default: 20)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for embedding computation (default: 256)",
    )

    args = parser.parse_args()

    print(f"{'='*80}")
    print("PERFORMANCE-BASED CONTINUOUS CONFIDENCE SFT DATA GENERATION")
    print(f"{'='*80}")
    print(f"Input:  {args.judged_data}")
    print(f"Output: {args.output}")
    print(f"k-NN:   {args.k_neighbors} neighbors")
    print(f"{'='*80}")

    # Load data
    judged_path = Path(args.judged_data)
    questions, answers, correctness = load_judged_data(judged_path)

    print(f"\n📚 Loaded {len(questions)} examples")
    print(f"  Accuracy: {sum(correctness) / len(correctness):.1%}")

    # Generate confidence labels
    confidence_labels = generate_knn_confidence_labels(
        questions=questions,
        correctness=correctness,
        k_neighbors=args.k_neighbors,
        batch_size=args.batch_size,
    )

    # Generate SFT training data
    output_path = Path(args.output)
    generate_sft_training_data(
        questions=questions,
        answers=answers,
        confidence_labels=confidence_labels,
        output_path=output_path,
    )

    print(f"\n{'='*80}")
    print("GENERATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
