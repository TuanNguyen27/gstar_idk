"""
Confidence Score Calculation Methods for Router Training
"""

from typing import Dict, Tuple
import numpy as np


class ConfidenceCalculator:
    """Calculate confidence scores for router training."""

    @staticmethod
    def binary_confidence(medium_label: str, large_label: str, policy: str) -> float:
        """
        Method 1: Binary confidence (0 or 1).

        Simple but loses nuance about uncertainty.
        """
        if policy == "Standard_Query":
            return 1.0 if medium_label == "Correct" else 0.0
        elif policy == "Complex_Query":
            return 1.0 if large_label == "Correct" else 0.0
        elif policy == "Ambiguous_Query":
            return 0.1  # Low confidence for both-failed cases
        return 0.5

    @staticmethod
    def empirical_frequency(
        medium_label: str,
        large_label: str,
        policy: str,
        pattern_stats: Dict[str, Tuple[int, int]] = None
    ) -> float:
        """
        Method 2: Empirical frequency from training data.

        Uses observed success rates for each (policy, medium, large) pattern.

        Args:
            pattern_stats: Dict mapping pattern to (correct, total)
                e.g., {"Standard_Query|M:Correct|L:Correct": (70, 70)}
        """
        if pattern_stats is None:
            # Default to binary if no stats provided
            return ConfidenceCalculator.binary_confidence(medium_label, large_label, policy)

        pattern_key = f"{policy}|M:{medium_label}|L:{large_label}"

        if pattern_key in pattern_stats:
            correct, total = pattern_stats[pattern_key]
            if total > 0:
                # Laplace smoothing to avoid 0/1 extremes
                alpha = 1  # Smoothing parameter
                return (correct + alpha) / (total + 2 * alpha)

        # Fallback to binary
        return ConfidenceCalculator.binary_confidence(medium_label, large_label, policy)

    @staticmethod
    def agreement_based(
        medium_label: str,
        large_label: str,
        policy: str
    ) -> float:
        """
        Method 3: Model agreement as confidence proxy.

        High confidence when models agree, lower when they disagree.
        """
        # Base confidence from routing decision
        if policy == "Standard_Query":
            base_conf = 1.0 if medium_label == "Correct" else 0.0
        elif policy == "Complex_Query":
            base_conf = 1.0 if large_label == "Correct" else 0.0
        else:  # Ambiguous_Query
            base_conf = 0.1

        # Adjust based on model agreement
        agreement_boost = 0.0

        if medium_label == large_label:
            # Models agree - higher confidence
            agreement_boost = 0.1
        else:
            # Models disagree - lower confidence
            agreement_boost = -0.1

        # Clip to [0, 1]
        return np.clip(base_conf + agreement_boost, 0.0, 1.0)

    @staticmethod
    def softmax_temperature(
        medium_label: str,
        large_label: str,
        policy: str,
        temperature: float = 1.0
    ) -> float:
        """
        Method 4: Temperature-scaled confidence.

        Uses softmax with temperature to smooth confidence scores.
        Higher temperature = more uniform (less confident).
        """
        # Get base confidence
        base_conf = ConfidenceCalculator.binary_confidence(medium_label, large_label, policy)

        # Apply temperature scaling
        # Convert to logit space, scale, convert back
        if base_conf == 0.0:
            logit = -10  # Large negative
        elif base_conf == 1.0:
            logit = 10  # Large positive
        else:
            logit = np.log(base_conf / (1 - base_conf))

        scaled_logit = logit / temperature
        scaled_conf = 1 / (1 + np.exp(-scaled_logit))

        return float(scaled_conf)


def compute_pattern_statistics(oracle_examples: list) -> Dict[str, Tuple[int, int]]:
    """
    Compute empirical success rates for each oracle pattern.

    Args:
        oracle_examples: List of oracle examples with keys:
            - final_policy, medium_label, large_label

    Returns:
        Dict mapping pattern to (num_correct, total_count)
    """
    from collections import defaultdict

    stats = defaultdict(lambda: [0, 0])  # [correct, total]

    for ex in oracle_examples:
        policy = ex['final_policy']
        medium = ex['medium_label']
        large = ex['large_label']

        # Determine if routing would succeed
        if policy == 'Standard_Query':
            is_correct = (medium == 'Correct')
        elif policy == 'Complex_Query':
            is_correct = (large == 'Correct')
        else:  # Ambiguous_Query
            is_correct = False

        pattern_key = f"{policy}|M:{medium}|L:{large}"
        stats[pattern_key][1] += 1  # total
        if is_correct:
            stats[pattern_key][0] += 1  # correct

    # Convert to tuples
    return {k: tuple(v) for k, v in stats.items()}


if __name__ == "__main__":
    # Example usage
    calc = ConfidenceCalculator()

    print("="*80)
    print("CONFIDENCE CALCULATION METHODS COMPARISON")
    print("="*80)

    test_cases = [
        ("Standard_Query", "Correct", "Correct"),
        ("Standard_Query", "Incorrect", "Correct"),
        ("Complex_Query", "Incorrect", "Correct"),
        ("Ambiguous_Query", "Incorrect", "Incorrect"),
    ]

    for policy, medium, large in test_cases:
        print(f"\nPolicy: {policy}, Medium: {medium}, Large: {large}")
        print(f"  Binary:    {calc.binary_confidence(medium, large, policy):.2f}")
        print(f"  Agreement: {calc.agreement_based(medium, large, policy):.2f}")
        print(f"  Softmax(T=2.0): {calc.softmax_temperature(medium, large, policy, temperature=2.0):.2f}")
