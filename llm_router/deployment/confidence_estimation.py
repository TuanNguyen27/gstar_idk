"""
Confidence Estimation for Router Decisions
Uses entropy of token predictions to estimate routing confidence.

Lower entropy = Higher confidence (model is certain about next tokens)
Higher entropy = Lower confidence (model is uncertain)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """
    Estimates routing confidence using entropy of token predictions.

    Only works for SLM routers where we have access to logits.
    Cannot be applied to API-based models (Gemini, GPT-4, etc.).
    """

    def __init__(
        self,
        method: str = "mean_entropy",
        high_confidence_threshold: float = 1.0,
        low_confidence_threshold: float = 3.0,
    ):
        """
        Initialize confidence estimator.

        Args:
            method: Confidence calculation method
                - "mean_entropy": Average entropy across all tokens
                - "min_entropy": Minimum entropy (most confident token)
                - "max_entropy": Maximum entropy (least confident token)
                - "decision_token_entropy": Entropy of the decision token only
            high_confidence_threshold: Entropy threshold for high confidence
            low_confidence_threshold: Entropy threshold for low confidence
        """
        self.method = method
        self.high_conf_threshold = high_confidence_threshold
        self.low_conf_threshold = low_confidence_threshold

    def calculate_entropy(self, logits: torch.Tensor) -> float:
        """
        Calculate entropy for a single token's logits.

        Entropy = -Σ(p * log(p)) where p is probability distribution

        Args:
            logits: Logits for a single token [vocab_size]

        Returns:
            Entropy value (lower = more confident)
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # Calculate entropy: -Σ(p * log(p))
        log_probs = torch.log(probs + 1e-10)  # Add epsilon to avoid log(0)
        entropy = -torch.sum(probs * log_probs)

        return entropy.item()

    def calculate_sequence_entropy(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Calculate various entropy metrics for a sequence.

        Args:
            logits: Model output logits [batch_size, seq_len, vocab_size]
            attention_mask: Mask for valid tokens [batch_size, seq_len]

        Returns:
            Dict with entropy metrics
        """
        # Remove batch dimension if present
        if logits.dim() == 3:
            logits = logits[0]  # [seq_len, vocab_size]

        if attention_mask is not None and attention_mask.dim() == 2:
            attention_mask = attention_mask[0]  # [seq_len]

        # Calculate entropy for each token
        entropies = []
        for i in range(logits.shape[0]):
            if attention_mask is None or attention_mask[i] == 1:
                entropy = self.calculate_entropy(logits[i])
                entropies.append(entropy)

        if not entropies:
            return {
                "mean_entropy": float('inf'),
                "min_entropy": float('inf'),
                "max_entropy": float('inf'),
                "std_entropy": 0.0,
            }

        return {
            "mean_entropy": np.mean(entropies),
            "min_entropy": np.min(entropies),
            "max_entropy": np.max(entropies),
            "std_entropy": np.std(entropies),
            "entropies": entropies,  # Per-token entropies
        }

    def estimate_confidence(
        self,
        logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decision_token_idx: Optional[int] = None,
    ) -> Tuple[float, str]:
        """
        Estimate confidence level for routing decision.

        Args:
            logits: Model output logits
            attention_mask: Mask for valid tokens
            decision_token_idx: Index of the decision token (e.g., token after [DECISION])

        Returns:
            Tuple of (confidence_score, confidence_level)
            - confidence_score: 0.0-1.0 (1.0 = highest confidence)
            - confidence_level: "high", "medium", or "low"
        """
        entropy_metrics = self.calculate_sequence_entropy(logits, attention_mask)

        # Select entropy based on method
        if self.method == "mean_entropy":
            entropy = entropy_metrics["mean_entropy"]
        elif self.method == "min_entropy":
            entropy = entropy_metrics["min_entropy"]
        elif self.method == "max_entropy":
            entropy = entropy_metrics["max_entropy"]
        elif self.method == "decision_token_entropy":
            if decision_token_idx is not None:
                entropy = entropy_metrics["entropies"][decision_token_idx]
            else:
                # Fallback to last token (usually the decision)
                entropy = entropy_metrics["entropies"][-1]
        else:
            entropy = entropy_metrics["mean_entropy"]

        # Convert entropy to confidence score (0-1 scale)
        # Lower entropy = higher confidence
        if entropy <= self.high_conf_threshold:
            confidence_score = 1.0
            confidence_level = "high"
        elif entropy >= self.low_conf_threshold:
            confidence_score = 0.0
            confidence_level = "low"
        else:
            # Linear interpolation between thresholds
            confidence_score = 1.0 - (
                (entropy - self.high_conf_threshold) /
                (self.low_conf_threshold - self.high_conf_threshold)
            )
            confidence_level = "medium"

        return confidence_score, confidence_level

    def get_calibrated_thresholds(
        self,
        validation_data: list,
        model,
        tokenizer,
        percentiles: Tuple[float, float] = (25, 75),
    ) -> Tuple[float, float]:
        """
        Calibrate entropy thresholds on validation data.

        Args:
            validation_data: List of validation examples
            model: Router model
            tokenizer: Tokenizer
            percentiles: (low, high) percentiles for thresholds

        Returns:
            Tuple of (high_conf_threshold, low_conf_threshold)
        """
        entropies = []

        model.eval()
        with torch.no_grad():
            for example in validation_data:
                inputs = tokenizer(
                    example["prompt"],
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                )

                # Calculate mean entropy
                logits = torch.stack(outputs.scores)  # [seq_len, batch, vocab]
                logits = logits.squeeze(1)  # [seq_len, vocab]
                metrics = self.calculate_sequence_entropy(logits)
                entropies.append(metrics["mean_entropy"])

        # Calculate percentile-based thresholds
        high_threshold = np.percentile(entropies, percentiles[0])
        low_threshold = np.percentile(entropies, percentiles[1])

        logger.info(f"Calibrated thresholds: high={high_threshold:.2f}, low={low_threshold:.2f}")

        return high_threshold, low_threshold


class ConfidenceAwareRouter:
    """
    Router that uses confidence scores to make decisions.

    High confidence: Use router decision
    Low confidence: Escalate to large model or use fallback strategy
    """

    def __init__(
        self,
        router_inference,
        confidence_estimator: ConfidenceEstimator,
        low_confidence_strategy: str = "escalate",
        escalation_threshold: float = 0.3,
    ):
        """
        Initialize confidence-aware router.

        Args:
            router_inference: RouterInference instance
            confidence_estimator: ConfidenceEstimator instance
            low_confidence_strategy: What to do on low confidence
                - "escalate": Always route to large model
                - "standard": Use Standard_Query mapping
                - "complex": Use Complex_Query mapping
            escalation_threshold: Confidence threshold for escalation
        """
        self.router = router_inference
        self.confidence_estimator = confidence_estimator
        self.low_confidence_strategy = low_confidence_strategy
        self.escalation_threshold = escalation_threshold

    def route_with_confidence(
        self,
        query: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> Dict:
        """
        Route query with confidence estimation.

        Args:
            query: User query
            temperature: Sampling temperature
            max_tokens: Maximum tokens

        Returns:
            Dict with:
            - policy: Routed policy
            - target_model: Target model name
            - reasoning: Routing reasoning
            - confidence_score: 0-1 confidence score
            - confidence_level: "high", "medium", or "low"
            - escalated: Whether decision was escalated due to low confidence
        """
        from ..training.prompt_templates import create_router_prompt

        # Create prompt
        prompt = create_router_prompt(query)

        # Tokenize
        inputs = self.router.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.router.model.device) for k, v in inputs.items()}

        # Generate with output scores
        with torch.no_grad():
            outputs = self.router.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.router.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

        # Decode response
        response = self.router.tokenizer.decode(
            outputs.sequences[0],
            skip_special_tokens=True
        )
        response = response.split("[/INST]")[-1].strip()

        # Parse policy and reasoning
        policy, reasoning = self.router._parse_response(response)

        # Calculate confidence
        logits = torch.stack(outputs.scores)  # [seq_len, batch, vocab]
        logits = logits.squeeze(1)  # [seq_len, vocab]

        confidence_score, confidence_level = self.confidence_estimator.estimate_confidence(
            logits=logits
        )

        # Determine if we should escalate
        escalated = False
        original_policy = policy

        if confidence_score < self.escalation_threshold:
            escalated = True
            if self.low_confidence_strategy == "escalate":
                policy = "Complex_Query"
            elif self.low_confidence_strategy == "standard":
                policy = "Standard_Query"
            elif self.low_confidence_strategy == "complex":
                policy = "Complex_Query"

        # Map to target model
        target_model = self.router.policy_map.get(policy, "gemini-2.5-flash")

        return {
            "query": query,
            "policy": policy,
            "original_policy": original_policy,
            "target_model": target_model,
            "reasoning": reasoning,
            "confidence_score": round(confidence_score, 3),
            "confidence_level": confidence_level,
            "escalated": escalated,
            "escalation_reason": f"Low confidence ({confidence_score:.2f} < {self.escalation_threshold})" if escalated else None,
        }


# Example usage and calibration utilities

def calibrate_on_validation_set(
    router_inference,
    validation_file: str,
    method: str = "mean_entropy",
) -> ConfidenceEstimator:
    """
    Calibrate confidence estimator on validation data.

    Args:
        router_inference: RouterInference instance
        validation_file: Path to validation JSONL
        method: Entropy calculation method

    Returns:
        Calibrated ConfidenceEstimator
    """
    import json

    # Load validation data
    validation_data = []
    with open(validation_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            validation_data.append({
                "prompt": create_router_prompt(data.get("query") or data.get("problem")),
                "ground_truth": data.get("answer") or data.get("ground_truth"),
            })

    # Create estimator
    estimator = ConfidenceEstimator(method=method)

    # Calibrate thresholds
    high_thresh, low_thresh = estimator.get_calibrated_thresholds(
        validation_data=validation_data[:100],  # Use subset for speed
        model=router_inference.model,
        tokenizer=router_inference.tokenizer,
    )

    # Update thresholds
    estimator.high_conf_threshold = high_thresh
    estimator.low_conf_threshold = low_thresh

    return estimator


from ..training.prompt_templates import create_router_prompt
