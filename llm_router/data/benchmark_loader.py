"""
Benchmark Dataset Loader
Loads questions from standard benchmarks for oracle dataset generation.
"""

import json
from pathlib import Path
from typing import List, Dict, Iterator, Optional
from dataclasses import dataclass

@dataclass
class BenchmarkExample:
    """A single benchmark question-answer pair."""
    query: str
    ground_truth: str
    source: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BenchmarkLoader:
    """Loads benchmark datasets for training data generation."""

    @staticmethod
    def load_simpleqa(file_path: str, limit: Optional[int] = None) -> List[BenchmarkExample]:
        """
        Load SimpleQA benchmark.

        Expected format: JSONL with 'problem' and 'answer' fields.

        Args:
            file_path: Path to SimpleQA JSONL file
            limit: Maximum number of examples to load

        Returns:
            List of BenchmarkExample objects
        """
        examples = []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break

                data = json.loads(line)
                examples.append(BenchmarkExample(
                    query=data['problem'],
                    ground_truth=data['answer'],
                    source='simpleqa',
                    metadata={'id': data.get('id', idx)}
                ))

        return examples

    @staticmethod
    def load_natural_questions(file_path: str, limit: Optional[int] = None) -> List[BenchmarkExample]:
        """
        Load Natural Questions benchmark.

        Expected format: JSONL with 'question' and 'answer' fields.

        Args:
            file_path: Path to NQ JSONL file
            limit: Maximum number of examples to load

        Returns:
            List of BenchmarkExample objects
        """
        examples = []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break

                data = json.loads(line)
                # NQ can have multiple answers, take first one
                answer = data['answer']
                if isinstance(answer, list):
                    answer = answer[0]

                examples.append(BenchmarkExample(
                    query=data['question'],
                    ground_truth=answer,
                    source='natural_questions',
                    metadata={'id': data.get('id', idx)}
                ))

        return examples

    @staticmethod
    def load_custom_jsonl(file_path: str, limit: Optional[int] = None) -> List[BenchmarkExample]:
        """
        Load custom JSONL format.

        Expected format: JSONL with 'query' and 'ground_truth' fields.

        Args:
            file_path: Path to custom JSONL file
            limit: Maximum number of examples to load

        Returns:
            List of BenchmarkExample objects
        """
        examples = []
        with open(file_path, 'r') as f:
            for idx, line in enumerate(f):
                if limit and idx >= limit:
                    break

                data = json.loads(line)
                # Support both 'query'/'problem' and 'ground_truth'/'answer'
                query = data.get('query') or data.get('problem')
                ground_truth = data.get('ground_truth') or data.get('answer')

                examples.append(BenchmarkExample(
                    query=query,
                    ground_truth=ground_truth,
                    source='custom',
                    metadata=data.get('metadata', {})
                ))

        return examples

    @staticmethod
    def load_benchmark(
        benchmark_name: str,
        file_path: str,
        limit: Optional[int] = None
    ) -> List[BenchmarkExample]:
        """
        Load a benchmark by name.

        Args:
            benchmark_name: Name of benchmark ('simpleqa', 'natural_questions', 'custom')
            file_path: Path to benchmark file
            limit: Maximum number of examples to load

        Returns:
            List of BenchmarkExample objects
        """
        loaders = {
            'simpleqa': BenchmarkLoader.load_simpleqa,
            'natural_questions': BenchmarkLoader.load_natural_questions,
            'custom': BenchmarkLoader.load_custom_jsonl,
        }

        if benchmark_name not in loaders:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")

        return loaders[benchmark_name](file_path, limit)

