"""
Model Configuration for LLM Router Project
Defines all models used in the system.
"""

from dataclasses import dataclass
from typing import Literal

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: str
    model_id: str
    description: str
    cost_per_1m_tokens: float  # USD per 1M tokens

# Model Definitions
ROUTER_MODEL = ModelConfig(
    name="router",
    provider="huggingface",
    model_id="google/gemma-2b",  # Can swap to Qwen2.5-1.5B
    description="Small model to be trained as the router",
    cost_per_1m_tokens=0.0  # Local inference
)

MEDIUM_MODEL = ModelConfig(
    name="medium",
    provider="google",
    model_id="gemini-2.5-flash",
    description="Fast, cost-efficient model for standard tasks",
    cost_per_1m_tokens=0.075  # Flash pricing
)

LARGE_MODEL = ModelConfig(
    name="large",
    provider="google",
    model_id="gemini-2.5-pro",
    description="Advanced model for complex tasks",
    cost_per_1m_tokens=1.25  # Pro pricing
)

JUDGE_MODEL = ModelConfig(
    name="judge",
    provider="google",
    model_id="gemini-2.5-pro",
    description="Oracle model for labeling training data",
    cost_per_1m_tokens=1.25
)

# Policy Definitions
PolicyType = Literal["Standard_Query", "Complex_Query", "Ambiguous_Query"]

POLICIES = {
    "Standard_Query": {
        "description": "A simple, common, or high-confidence query that can be answered correctly and efficiently by a standard model.",
        "target_model": "gemini-2.5-flash"
    },
    "Complex_Query": {
        "description": "A complex, niche, or difficult factual query that requires an advanced model to answer correctly.",
        "target_model": "gemini-2.5-pro"
    },
    "Ambiguous_Query": {
        "description": "A query that is unanswerable, unsafe, or where both standard and advanced models are likely to fail or abstain (IDK).",
        "target_model": "gemini-2.5-flash"  # Default to cheaper model
    }
}

# Judgment Labels
JudgmentLabel = Literal["Correct", "Incorrect", "IDK"]
