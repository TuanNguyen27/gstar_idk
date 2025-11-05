"""
Gemini API Client
Handles all interactions with Google's Gemini models.
"""

import os
import json
import time
from typing import Optional, Dict, Any
import google.generativeai as genai

class GeminiClient:
    """Client for interacting with Gemini models."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Gemini client.

        Args:
            api_key: Google AI API key. If None, reads from GEMINI_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided")

        genai.configure(api_key=self.api_key)

    def generate(
        self,
        model_id: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
        retry_count: int = 3,
        retry_delay: float = 1.0,
    ) -> str:
        """
        Generate text using a Gemini model.

        Args:
            model_id: Gemini model identifier (e.g., "gemini-2.5-flash")
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: If True, request JSON output
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds

        Returns:
            Generated text response
        """
        model = genai.GenerativeModel(model_id)

        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if json_mode:
            generation_config["response_mime_type"] = "application/json"

        for attempt in range(retry_count):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                )
                return response.text

            except Exception as e:
                if attempt == retry_count - 1:
                    raise RuntimeError(f"Failed after {retry_count} attempts: {e}")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

        raise RuntimeError("Should not reach here")

    def judge_answer(
        self,
        query: str,
        ground_truth: str,
        model_answer: str,
    ) -> Dict[str, Any]:
        """
        Judge a model's answer using Gemini as oracle (blind judging).

        Args:
            query: The original query
            ground_truth: The correct answer
            model_answer: The model's answer to judge

        Returns:
            Dict with 'label' (Correct/Incorrect/IDK) and 'reasoning'
        """
        judge_prompt = f"""Analyze this query, ground truth answer, and model's answer.

Query: {query}

Ground Truth: {ground_truth}

Model's Answer: {model_answer}

Evaluate the model's answer and respond in JSON format with:
- "label": Must be exactly one of: "Correct", "Incorrect", or "IDK"
- "reasoning": Brief explanation (1-2 sentences) for your judgment

Examples:
- "The answer is factually correct and matches the ground truth"
- "The answer is factually wrong"
- "The model abstained or said it doesn't know"

Respond ONLY with valid JSON."""

        response_text = self.generate(
            model_id="gemini-2.5-pro",
            prompt=judge_prompt,
            temperature=0.0,  # Deterministic judging
            max_tokens=256,
            json_mode=True,
        )

        try:
            result = json.loads(response_text)
            # Validate label
            if result["label"] not in ["Correct", "Incorrect", "IDK"]:
                raise ValueError(f"Invalid label: {result['label']}")
            return result
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            raise ValueError(f"Invalid judge response: {response_text}") from e

    def analyze_query(self, query: str) -> str:
        """
        Analyze a query's characteristics (for preventing data leakage).

        Args:
            query: The query to analyze

        Returns:
            One-sentence description of query characteristics
        """
        analyzer_prompt = f"""Analyze the following query and describe its characteristics in one sentence.

Query: {query}

Examples:
- "This is a simple arithmetic question"
- "This is a niche, complex question about theoretical physics"
- "This is a straightforward factual question about history"
- "This is an ambiguous or unanswerable question"

Respond with ONLY the one-sentence analysis, no additional text."""

        return self.generate(
            model_id="gemini-2.5-pro",
            prompt=analyzer_prompt,
            temperature=0.3,
            max_tokens=100,
        ).strip()
