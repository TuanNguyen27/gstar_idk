"""
SimpleQA Benchmark Evaluation
Evaluates router performance on OpenAI's SimpleQA benchmark.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Result for a single evaluation example."""
    query: str
    ground_truth: str
    routed_policy: str
    routed_model: str
    model_answer: str
    is_correct: bool
    routing_reasoning: str
    cost_estimate: float
    metadata: Dict = None

class SimpleQAEvaluator:
    """Evaluates router on SimpleQA benchmark."""

    # Cost per 1M tokens (input + output combined, rough estimate)
    MODEL_COSTS = {
        "gemini-2.5-flash": 0.10,  # ~$0.05 input + $0.05 output per 1M
        "gemini-2.5-pro": 2.00,    # ~$1.00 input + $1.00 output per 1M
    }

    def __init__(
        self,
        router,
        gemini_client,
        judge_model_id: str = "gemini-2.5-pro",
    ):
        """
        Initialize evaluator.

        Args:
            router: RouterInference instance
            gemini_client: GeminiClient instance
            judge_model_id: Model to use for judging correctness
        """
        self.router = router
        self.gemini_client = gemini_client
        self.judge_model_id = judge_model_id

    def evaluate(
        self,
        test_file: Path,
        output_file: Path,
        limit: int = None,
    ) -> Dict[str, Any]:
        """
        Evaluate router on SimpleQA benchmark.

        Args:
            test_file: Path to SimpleQA test JSONL
            output_file: Path to save results
            limit: Limit number of examples

        Returns:
            Dict with evaluation metrics
        """
        logger.info(f"Evaluating on SimpleQA: {test_file}")

        # Load test data
        test_examples = []
        with open(test_file, 'r') as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break
                data = json.loads(line)
                test_examples.append(data)

        logger.info(f"Loaded {len(test_examples)} test examples")

        results = []
        for example in tqdm(test_examples, desc="Evaluating"):
            try:
                result = self._evaluate_single(example)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating example: {e}")
                continue

        # Save results
        self._save_results(results, output_file)

        # Compute metrics
        metrics = self._compute_metrics(results)

        logger.info(f"Evaluation complete. Results saved to {output_file}")
        return metrics

    def _evaluate_single(self, example: Dict) -> EvaluationResult:
        """Evaluate a single example."""
        query = example.get('problem') or example.get('query')
        ground_truth = example.get('answer') or example.get('ground_truth')

        # Step 1: Route query
        policy, target_model, reasoning = self.router.route_query(query)

        # Step 2: Get answer from routed model
        model_answer = self.gemini_client.generate(
            model_id=target_model,
            prompt=query,
            temperature=0.7,
            max_tokens=512,
        )

        # Step 3: Judge correctness
        judgment = self.gemini_client.judge_answer(
            query=query,
            ground_truth=ground_truth,
            model_answer=model_answer,
        )
        is_correct = (judgment['label'] == 'Correct')

        # Step 4: Estimate cost (rough approximation)
        avg_tokens = len(query.split()) * 1.3 + len(model_answer.split()) * 1.3
        cost_estimate = (avg_tokens / 1_000_000) * self.MODEL_COSTS.get(target_model, 0.1)

        return EvaluationResult(
            query=query,
            ground_truth=ground_truth,
            routed_policy=policy,
            routed_model=target_model,
            model_answer=model_answer,
            is_correct=is_correct,
            routing_reasoning=reasoning,
            cost_estimate=cost_estimate,
            metadata=example.get('metadata', {}),
        )

    def _compute_metrics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute evaluation metrics."""
        total = len(results)
        if total == 0:
            return {}

        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total

        # Policy distribution
        policy_counts = {}
        policy_correct = {}
        for r in results:
            policy = r.routed_policy
            policy_counts[policy] = policy_counts.get(policy, 0) + 1
            if r.is_correct:
                policy_correct[policy] = policy_correct.get(policy, 0) + 1

        # Cost analysis
        total_cost = sum(r.cost_estimate for r in results)
        avg_cost = total_cost / total

        # Baseline cost (if all queries went to Pro)
        baseline_cost = total * self.MODEL_COSTS["gemini-2.5-pro"] * 200 / 1_000_000  # ~200 tokens avg

        metrics = {
            "total_examples": total,
            "correct": correct,
            "accuracy": round(accuracy * 100, 2),
            "policy_distribution": {
                policy: {
                    "count": count,
                    "percentage": round(count / total * 100, 2),
                    "accuracy": round(policy_correct.get(policy, 0) / count * 100, 2) if count > 0 else 0,
                }
                for policy, count in policy_counts.items()
            },
            "cost": {
                "total_estimated": round(total_cost, 4),
                "average_per_query": round(avg_cost, 6),
                "baseline_pro_only": round(baseline_cost, 4),
                "savings_percentage": round((1 - total_cost / baseline_cost) * 100, 2) if baseline_cost > 0 else 0,
            }
        }

        return metrics

    def _save_results(self, results: List[EvaluationResult], output_file: Path):
        """Save evaluation results."""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(asdict(result)) + '\n')

    def compare_with_baseline(
        self,
        test_file: Path,
        baseline_model: str = "gemini-2.5-flash",
        limit: int = None,
    ) -> Dict[str, Any]:
        """
        Compare router performance with baseline (single model).

        Args:
            test_file: Path to test JSONL
            baseline_model: Model to use as baseline
            limit: Limit number of examples

        Returns:
            Comparison metrics
        """
        logger.info(f"Running baseline comparison with {baseline_model}")

        # Load test data
        test_examples = []
        with open(test_file, 'r') as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break
                test_examples.append(json.loads(line))

        baseline_correct = 0
        baseline_cost = 0

        for example in tqdm(test_examples, desc="Baseline"):
            query = example.get('problem') or example.get('query')
            ground_truth = example.get('answer') or example.get('ground_truth')

            # Get answer
            answer = self.gemini_client.generate(
                model_id=baseline_model,
                prompt=query,
                temperature=0.7,
                max_tokens=512,
            )

            # Judge
            judgment = self.gemini_client.judge_answer(query, ground_truth, answer)
            if judgment['label'] == 'Correct':
                baseline_correct += 1

            # Cost
            avg_tokens = len(query.split()) * 1.3 + len(answer.split()) * 1.3
            baseline_cost += (avg_tokens / 1_000_000) * self.MODEL_COSTS.get(baseline_model, 0.1)

        baseline_accuracy = baseline_correct / len(test_examples) * 100

        return {
            "baseline_model": baseline_model,
            "baseline_accuracy": round(baseline_accuracy, 2),
            "baseline_cost": round(baseline_cost, 4),
        }
