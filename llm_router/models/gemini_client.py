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

        # Adjusted safety settings to reduce blocking
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

        for attempt in range(retry_count):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )

                # Check for blocked content BEFORE accessing .text
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        reason_name = str(finish_reason)

                        # Only block on actual error conditions
                        if 'SAFETY' in reason_name:
                            return "[BLOCKED: Safety filter]"
                        elif 'RECITATION' in reason_name:
                            return "[BLOCKED: Recitation]"
                        elif 'OTHER' in reason_name:
                            return "[BLOCKED: Other]"
                        # If STOP or MAX_TOKENS, proceed to get text

                # Now safely try to access text
                try:
                    if response.text:
                        return response.text
                except:
                    pass

                return "[ERROR: No valid response]"

            except Exception as e:
                # Check if it's a safety/blocking error
                error_msg = str(e)
                if "finish_reason" in error_msg and "is 2" in error_msg:
                    return "[BLOCKED: Safety filter]"
                if "finish_reason" in error_msg and "is 3" in error_msg:
                    return "[BLOCKED: Recitation]"

                if attempt == retry_count - 1:
                    return f"[ERROR: {error_msg}]"
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
            model_id="gemini-2.0-flash",
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
        analyzer_prompt = f"""Analyze the following query and describe its characteristics in one sentence. Focus on what makes it simple/complex, common/niche, verifiable/ambiguous.

Query: {query}

Good examples:
- "Simple arithmetic requiring basic calculation"
- "Obscure historical fact about a lesser-known figure requiring specialized knowledge"
- "Common knowledge question about a widely-known celebrity"
- "Highly specific question about recent pop culture requiring up-to-date information"
- "Ambiguous question missing context or asking about non-existent information"
- "Niche technical question requiring domain expertise"
- "Well-known historical event that is widely documented"

Respond with ONLY the one-sentence analysis focusing on complexity and domain, no additional text."""

        return self.generate(
            model_id="gemini-2.0-flash",
            prompt=analyzer_prompt,
            temperature=0.5,
            max_tokens=150,
        ).strip()
