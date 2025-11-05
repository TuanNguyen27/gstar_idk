"""
DPO Training Data Preparation
Converts oracle dataset into SFT + DPO training format.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

from data.oracle_generator import OracleExample
from training.prompt_templates import create_router_prompt, create_reasoning_chain, create_rejected_reasoning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DPOTrainingExample:
    """A single DPO training example."""
    prompt: str
    chosen: str
    rejected: str

class DPODataPreparation:
    """Prepares oracle dataset for SFT + DPO training."""

    @staticmethod
    def convert_oracle_to_dpo(
        oracle_file: Path,
        output_file: Path,
    ) -> List[DPOTrainingExample]:
        """
        Convert oracle dataset to DPO training format.

        Args:
            oracle_file: Path to oracle dataset JSONL
            output_file: Path to save DPO training data JSONL

        Returns:
            List of DPOTrainingExample objects
        """
        logger.info(f"Converting oracle dataset to DPO format")

        # Load oracle examples
        oracle_examples = []
        with open(oracle_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                oracle_examples.append(data)

        dpo_examples = []

        for data in oracle_examples:
            query = data["query"]
            final_policy = data["final_policy"]
            query_analysis = data["query_analysis"]

            # Create full instruction prompt
            prompt = create_router_prompt(query)

            # Create chosen (correct) reasoning chain
            chosen = create_reasoning_chain(query_analysis, final_policy)

            # Create rejected (incorrect) reasoning chain
            rejected = create_rejected_reasoning(query_analysis, final_policy)

            dpo_example = DPOTrainingExample(
                prompt=prompt,
                chosen=chosen,
                rejected=rejected,
            )

            dpo_examples.append(dpo_example)

        # Save as JSONL
        DPODataPreparation._save_dpo_examples(dpo_examples, output_file)
        logger.info(f"Converted {len(dpo_examples)} examples to DPO format")

        return dpo_examples

    @staticmethod
    def create_sft_dataset(dpo_file: Path, output_file: Path) -> None:
        """
        Create SFT dataset from DPO data (prompt + chosen pairs).

        Args:
            dpo_file: Path to DPO training data JSONL
            output_file: Path to save SFT training data JSONL
        """
        logger.info(f"Creating SFT dataset from DPO data")

        sft_examples = []

        with open(dpo_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                sft_examples.append({
                    "prompt": data["prompt"],
                    "completion": data["chosen"],
                })

        # Save as JSONL
        with open(output_file, 'w') as f:
            for example in sft_examples:
                f.write(json.dumps(example) + '\n')

        logger.info(f"Created {len(sft_examples)} SFT examples")

    @staticmethod
    def _save_dpo_examples(examples: List[DPOTrainingExample], file_path: Path) -> None:
        """Save DPO examples as JSONL file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            for example in examples:
                f.write(json.dumps({
                    "prompt": example.prompt,
                    "chosen": example.chosen,
                    "rejected": example.rejected,
                }) + '\n')

    @staticmethod
    def analyze_dataset_distribution(oracle_file: Path) -> Dict[str, Any]:
        """
        Analyze the distribution of policies in the oracle dataset.

        Args:
            oracle_file: Path to oracle dataset JSONL

        Returns:
            Dict with distribution statistics
        """
        policy_counts = {
            "Standard_Query": 0,
            "Complex_Query": 0,
            "Ambiguous_Query": 0,
        }

        total = 0
        with open(oracle_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                policy = data["final_policy"]
                policy_counts[policy] += 1
                total += 1

        stats = {
            "total_examples": total,
            "policy_distribution": {
                policy: {
                    "count": count,
                    "percentage": round(count / total * 100, 2) if total > 0 else 0
                }
                for policy, count in policy_counts.items()
            }
        }

        logger.info(f"Dataset distribution: {stats}")
        return stats
