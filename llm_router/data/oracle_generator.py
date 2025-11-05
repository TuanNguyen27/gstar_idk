"""
Oracle Dataset Generator
Generates labeled training data using the oracle logic.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

from models.gemini_client import GeminiClient
from config import get_policy_label, get_matrix_rationale, JudgmentLabel, PolicyType
from data.benchmark_loader import BenchmarkExample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OracleExample:
    """A labeled training example from the oracle."""
    query: str
    medium_label: JudgmentLabel
    medium_reasoning: str
    large_label: JudgmentLabel
    large_reasoning: str
    final_policy: PolicyType
    policy_rationale: str
    query_analysis: str
    source: str
    metadata: Dict = None

class OracleDatasetGenerator:
    """Generates oracle-labeled dataset for router training."""

    def __init__(self, gemini_api_key: str):
        """
        Initialize oracle generator.

        Args:
            gemini_api_key: API key for Gemini
        """
        self.client = GeminiClient(api_key=gemini_api_key)

    def generate_model_answers(
        self,
        benchmark_examples: List[BenchmarkExample],
        output_file: Path,
        batch_size: int = 10,
    ) -> None:
        """
        Step 1: Generate answers from medium and large models.

        Args:
            benchmark_examples: List of benchmark questions
            output_file: Path to save intermediate results
            batch_size: Number of examples to process before saving
        """
        logger.info(f"Generating model answers for {len(benchmark_examples)} examples")

        results = []
        for idx, example in enumerate(tqdm(benchmark_examples, desc="Generating answers")):
            try:
                # Get medium model answer
                medium_answer = self.client.generate(
                    model_id="gemini-2.5-flash",
                    prompt=example.query,
                    temperature=0.7,
                    max_tokens=512,
                )

                # Get large model answer
                large_answer = self.client.generate(
                    model_id="gemini-2.5-pro",
                    prompt=example.query,
                    temperature=0.7,
                    max_tokens=512,
                )

                results.append({
                    "query": example.query,
                    "ground_truth": example.ground_truth,
                    "medium_answer": medium_answer,
                    "large_answer": large_answer,
                    "source": example.source,
                    "metadata": example.metadata,
                })

                # Save periodically
                if (idx + 1) % batch_size == 0:
                    self._save_jsonl(results, output_file)
                    logger.info(f"Saved {len(results)} results to {output_file}")

            except Exception as e:
                logger.error(f"Error processing example {idx}: {e}")
                continue

        # Final save
        self._save_jsonl(results, output_file)
        logger.info(f"Completed: {len(results)} model answer pairs generated")

    def judge_and_label(
        self,
        model_answers_file: Path,
        output_file: Path,
        batch_size: int = 10,
    ) -> List[OracleExample]:
        """
        Step 2 & 3: Judge answers and apply oracle matrix to generate labels.

        Args:
            model_answers_file: Path to model answers from step 1
            output_file: Path to save oracle dataset
            batch_size: Number of examples to process before saving

        Returns:
            List of OracleExample objects
        """
        logger.info(f"Judging answers and applying oracle matrix")

        # Load model answers
        with open(model_answers_file, 'r') as f:
            model_answers = [json.loads(line) for line in f]

        oracle_examples = []

        for idx, data in enumerate(tqdm(model_answers, desc="Judging & labeling")):
            try:
                # Blind judging: Judge each answer separately
                medium_judgment = self.client.judge_answer(
                    query=data["query"],
                    ground_truth=data["ground_truth"],
                    model_answer=data["medium_answer"],
                )

                large_judgment = self.client.judge_answer(
                    query=data["query"],
                    ground_truth=data["ground_truth"],
                    model_answer=data["large_answer"],
                )

                # Apply oracle matrix
                medium_label = medium_judgment["label"]
                large_label = large_judgment["label"]
                final_policy = get_policy_label(medium_label, large_label)
                policy_rationale = get_matrix_rationale(medium_label, large_label)

                # Generate query analysis (no data leakage)
                query_analysis = self.client.analyze_query(data["query"])

                oracle_example = OracleExample(
                    query=data["query"],
                    medium_label=medium_label,
                    medium_reasoning=medium_judgment["reasoning"],
                    large_label=large_label,
                    large_reasoning=large_judgment["reasoning"],
                    final_policy=final_policy,
                    policy_rationale=policy_rationale,
                    query_analysis=query_analysis,
                    source=data["source"],
                    metadata=data.get("metadata"),
                )

                oracle_examples.append(oracle_example)

                # Save periodically
                if (idx + 1) % batch_size == 0:
                    self._save_oracle_examples(oracle_examples, output_file)
                    logger.info(f"Saved {len(oracle_examples)} oracle examples")

            except Exception as e:
                logger.error(f"Error processing example {idx}: {e}")
                continue

        # Final save
        self._save_oracle_examples(oracle_examples, output_file)
        logger.info(f"Completed: {len(oracle_examples)} oracle examples generated")

        return oracle_examples

    def _save_jsonl(self, data: List[Dict], file_path: Path) -> None:
        """Save data as JSONL file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')

    def _save_oracle_examples(self, examples: List[OracleExample], file_path: Path) -> None:
        """Save oracle examples as JSONL file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            for example in examples:
                f.write(json.dumps(asdict(example)) + '\n')
