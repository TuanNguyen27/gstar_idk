"""
Router Inference Pipeline
Loads trained router and routes queries to appropriate models.
"""

import json
import re
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RouterInference:
    """Handles inference with the trained router model."""

    def __init__(
        self,
        model_path: str,
        base_model_name: str = "google/gemma-2b",
        policy_map_path: str = "./deployment/policy_map.json",
        device: str = "auto",
    ):
        """
        Initialize router inference.

        Args:
            model_path: Path to trained router model
            base_model_name: Base model identifier
            policy_map_path: Path to policy mapping JSON
            device: Device to run on ('auto', 'cuda', 'cpu')
        """
        self.model_path = model_path
        self.base_model_name = base_model_name
        self.device = device

        # Load policy map
        with open(policy_map_path, 'r') as f:
            self.policy_map = json.load(f)

        # Load model and tokenizer
        logger.info(f"Loading router model from {model_path}")
        self._load_model()

    def _load_model(self):
        """Load the trained router model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map=self.device,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Load PEFT adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()

        logger.info("Router model loaded successfully")

    def route_query(
        self,
        query: str,
        temperature: float = 0.1,
        max_tokens: int = 256,
    ) -> Tuple[str, str, str]:
        """
        Route a query to the appropriate model.

        Args:
            query: User query to route
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Tuple of (policy_name, target_model, reasoning)
        """
        # Create prompt
        from ..training.prompt_templates import create_router_prompt
        prompt = create_router_prompt(query)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract response (after [/INST])
        response = response.split("[/INST]")[-1].strip()

        # Parse decision
        policy_name, reasoning = self._parse_response(response)

        # Map to target model
        target_model = self.policy_map.get(policy_name, "gemini-2.5-flash")

        logger.info(f"Query routed to {policy_name} → {target_model}")

        return policy_name, target_model, reasoning

    def _parse_response(self, response: str) -> Tuple[str, str]:
        """
        Parse router response to extract policy and reasoning.

        Args:
            response: Raw model response

        Returns:
            Tuple of (policy_name, reasoning)
        """
        # Look for [DECISION] marker
        decision_match = re.search(r'\[DECISION\]\s*(\w+)', response)
        if decision_match:
            policy = decision_match.group(1)
        else:
            # Fallback: look for any policy name
            for policy_name in self.policy_map.keys():
                if policy_name in response:
                    policy = policy_name
                    break
            else:
                policy = "Standard_Query"  # Safe default

        # Extract reasoning (everything before [DECISION])
        reasoning_match = re.search(r'\[REASONING\](.*?)\[DECISION\]', response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            reasoning = response

        return policy, reasoning

class EndToEndPipeline:
    """Complete pipeline: Router + Model Execution."""

    def __init__(
        self,
        router: RouterInference,
        gemini_api_key: str,
    ):
        """
        Initialize end-to-end pipeline.

        Args:
            router: RouterInference instance
            gemini_api_key: API key for Gemini models
        """
        self.router = router

        # Initialize Gemini client
        from ..models import GeminiClient
        self.gemini_client = GeminiClient(api_key=gemini_api_key)

    def answer_query(
        self,
        query: str,
        verbose: bool = True,
    ) -> Dict:
        """
        Answer a query using the full pipeline.

        Args:
            query: User query
            verbose: If True, print routing decision

        Returns:
            Dict with routing info and final answer
        """
        # Step 1: Route query
        policy, target_model, reasoning = self.router.route_query(query)

        if verbose:
            print(f"[Router] Policy: {policy}")
            print(f"[Router] Model: {target_model}")
            print(f"[Router] Reasoning: {reasoning}\n")

        # Step 2: Get answer from target model
        answer = self.gemini_client.generate(
            model_id=target_model,
            prompt=query,
            temperature=0.7,
            max_tokens=1024,
        )

        return {
            "query": query,
            "policy": policy,
            "target_model": target_model,
            "reasoning": reasoning,
            "answer": answer,
        }
