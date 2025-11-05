"""
Modal-based Oracle Dataset Generation
Parallel processing for 10x speedup over local generation.
"""

import modal
import json
from pathlib import Path

# Create Modal app
app = modal.App("llm-router-oracle-generation")

# Image with dependencies
oracle_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "google-generativeai>=0.3.0",
        "tqdm>=4.65.0",
    )
)

# Volume for storing results
volume = modal.Volume.from_name("llm-router-data", create_if_missing=True)

@app.function(
    image=oracle_image,
    secrets=[modal.Secret.from_name("gemini-secret")],
    timeout=600,
    retries=2,
)
def generate_model_answers(example: dict, gemini_api_key: str) -> dict:
    """
    Generate answers from both medium and large models.

    Args:
        example: {query, ground_truth, source, metadata}
        gemini_api_key: Gemini API key

    Returns:
        Dict with medium_answer and large_answer
    """
    import google.generativeai as genai
    import time

    genai.configure(api_key=gemini_api_key)

    query = example["query"]
    ground_truth = example["ground_truth"]

    def generate_with_retry(model_id: str, prompt: str, max_retries: int = 3) -> str:
        """Generate with exponential backoff retry."""
        model = genai.GenerativeModel(model_id)

        # Adjusted safety settings for factual QA
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]

        for attempt in range(max_retries):
            try:
                response = model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.7, "max_output_tokens": 512},
                    safety_settings=safety_settings,
                )

                # Check if response has text
                if hasattr(response, 'text') and response.text:
                    return response.text

                # Handle blocked content
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        if finish_reason == 2:  # SAFETY
                            return "[BLOCKED: Safety filter triggered]"
                        elif finish_reason == 3:  # RECITATION
                            return "[BLOCKED: Recitation detected]"

                return "[ERROR: No valid response]"

            except Exception as e:
                if attempt == max_retries - 1:
                    return f"[ERROR: {str(e)}]"
                time.sleep(2 ** attempt)  # Exponential backoff

        return "[ERROR: Max retries exceeded]"

    # Generate answers
    medium_answer = generate_with_retry("gemini-2.0-flash-exp", query)
    large_answer = generate_with_retry("gemini-exp-1206", query)

    return {
        "query": query,
        "ground_truth": ground_truth,
        "medium_answer": medium_answer,
        "large_answer": large_answer,
        "source": example.get("source", "unknown"),
        "metadata": example.get("metadata", {}),
    }

@app.function(
    image=oracle_image,
    secrets=[modal.Secret.from_name("gemini-secret")],
    timeout=300,
    retries=2,
)
def judge_and_label(example: dict, gemini_api_key: str) -> dict:
    """
    Judge answers and apply oracle matrix to generate final labels.

    Args:
        example: {query, ground_truth, medium_answer, large_answer, ...}
        gemini_api_key: Gemini API key

    Returns:
        OracleExample dict
    """
    import google.generativeai as genai
    import time

    genai.configure(api_key=gemini_api_key)

    # Oracle matrix
    ORACLE_MATRIX = {
        ("Correct", "Correct"): "Standard_Query",
        ("Correct", "Incorrect"): "Standard_Query",
        ("Correct", "IDK"): "Standard_Query",
        ("Incorrect", "Correct"): "Complex_Query",
        ("Incorrect", "Incorrect"): "Ambiguous_Query",
        ("Incorrect", "IDK"): "Complex_Query",
        ("IDK", "Correct"): "Complex_Query",
        ("IDK", "Incorrect"): "Ambiguous_Query",
        ("IDK", "IDK"): "Ambiguous_Query",
    }

    def judge_answer(query: str, ground_truth: str, model_answer: str) -> dict:
        """Judge a single answer (blind judging)."""
        judge_prompt = f"""Analyze this query, ground truth answer, and model's answer.

Query: {query}

Ground Truth: {ground_truth}

Model's Answer: {model_answer}

Evaluate the model's answer and respond in JSON format with:
- "label": Must be exactly one of: "Correct", "Incorrect", or "IDK"
- "reasoning": Brief explanation (1-2 sentences)

Examples:
- "The answer is factually correct and matches the ground truth"
- "The answer is factually wrong"
- "The model abstained or said it doesn't know"

Respond ONLY with valid JSON."""

        model = genai.GenerativeModel("gemini-exp-1206")

        try:
            response = model.generate_content(
                judge_prompt,
                generation_config={
                    "temperature": 0.0,
                    "max_output_tokens": 256,
                    "response_mime_type": "application/json",
                },
            )

            if hasattr(response, 'text') and response.text:
                result = json.loads(response.text)
                if result["label"] in ["Correct", "Incorrect", "IDK"]:
                    return result

            return {"label": "IDK", "reasoning": "Failed to judge"}

        except Exception as e:
            return {"label": "IDK", "reasoning": f"Error: {str(e)}"}

    def analyze_query(query: str) -> str:
        """Analyze query characteristics (no data leakage)."""
        analyzer_prompt = f"""Analyze the following query and describe its characteristics in one sentence.

Query: {query}

Examples:
- "This is a simple arithmetic question"
- "This is a niche, complex question about theoretical physics"
- "This is a straightforward factual question about history"

Respond with ONLY the one-sentence analysis, no additional text."""

        model = genai.GenerativeModel("gemini-exp-1206")

        try:
            response = model.generate_content(
                analyzer_prompt,
                generation_config={"temperature": 0.3, "max_output_tokens": 100},
            )

            if hasattr(response, 'text') and response.text:
                return response.text.strip()

            return "This is a standard factual question."

        except Exception as e:
            return "This is a standard factual question."

    # Skip if answers are errors/blocked
    if example["medium_answer"].startswith("[") or example["large_answer"].startswith("["):
        return None

    # Judge both answers (blind judging)
    medium_judgment = judge_answer(
        example["query"],
        example["ground_truth"],
        example["medium_answer"],
    )

    large_judgment = judge_answer(
        example["query"],
        example["ground_truth"],
        example["large_answer"],
    )

    # Apply oracle matrix
    medium_label = medium_judgment["label"]
    large_label = large_judgment["label"]
    final_policy = ORACLE_MATRIX.get((medium_label, large_label), "Ambiguous_Query")

    # Rationales
    rationales = {
        ("Correct", "Correct"): "Both correct. Optimize for cost.",
        ("Correct", "Incorrect"): "Medium is correct. Optimize for cost/correctness.",
        ("Correct", "IDK"): "Medium is correct, which is the best outcome.",
        ("Incorrect", "Correct"): "Classic escalation. Large model is required.",
        ("Incorrect", "Incorrect"): "Both failed. Don't escalate; save cost.",
        ("Incorrect", "IDK"): "Escalate. IDK is safer than Incorrect.",
        ("IDK", "Correct"): "Classic escalation. Large model knows the answer.",
        ("IDK", "Incorrect"): "Medium was safer (IDK). Don't escalate.",
        ("IDK", "IDK"): "Both failed. Don't escalate; save cost.",
    }
    policy_rationale = rationales.get((medium_label, large_label), "Unknown case")

    # Analyze query
    query_analysis = analyze_query(example["query"])

    return {
        "query": example["query"],
        "medium_label": medium_label,
        "medium_reasoning": medium_judgment["reasoning"],
        "large_label": large_label,
        "large_reasoning": large_judgment["reasoning"],
        "final_policy": final_policy,
        "policy_rationale": policy_rationale,
        "query_analysis": query_analysis,
        "source": example["source"],
        "metadata": example.get("metadata", {}),
    }

@app.local_entrypoint()
def main(
    input_file: str = "./data/splits/simpleqa_train.jsonl",
    output_dir: str = "./data/generated",
    limit: int = None,
    concurrency: int = 10,
):
    """
    Run parallel oracle dataset generation on Modal.

    Args:
        input_file: Path to input JSONL file
        output_dir: Output directory
        limit: Limit number of examples
        concurrency: Number of parallel workers
    """
    import os

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    print(f"=== Modal Oracle Dataset Generation ===")
    print(f"Input: {input_file}")
    print(f"Output: {output_dir}")
    print(f"Limit: {limit or 'None'}")
    print(f"Concurrency: {concurrency}\n")

    # Load input data
    examples = []
    with open(input_file, 'r') as f:
        for idx, line in enumerate(f):
            if limit and idx >= limit:
                break
            data = json.loads(line)
            examples.append({
                "query": data.get("query") or data.get("problem"),
                "ground_truth": data.get("ground_truth") or data.get("answer"),
                "source": "simpleqa",
                "metadata": data.get("metadata", {}),
            })

    print(f"Loaded {len(examples)} examples\n")

    # Step 1: Generate model answers (parallel)
    print("Step 1: Generating model answers in parallel...")
    model_answers = []
    for result in generate_model_answers.map(
        examples,
        kwargs={"gemini_api_key": gemini_api_key},
        order_outputs=False,
    ):
        model_answers.append(result)
        print(f"  Progress: {len(model_answers)}/{len(examples)}")

    print(f"✓ Generated {len(model_answers)} model answer pairs\n")

    # Step 2: Judge and label (parallel)
    print("Step 2: Judging answers and applying oracle matrix...")
    oracle_examples = []
    for result in judge_and_label.map(
        model_answers,
        kwargs={"gemini_api_key": gemini_api_key},
        order_outputs=False,
    ):
        if result is not None:
            oracle_examples.append(result)
        print(f"  Progress: {len(oracle_examples)} valid examples")

    print(f"✓ Generated {len(oracle_examples)} oracle examples\n")

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / "oracle_dataset.jsonl"
    with open(output_file, 'w') as f:
        for example in oracle_examples:
            f.write(json.dumps(example) + '\n')

    print(f"=== Generation Complete ===")
    print(f"Total examples: {len(oracle_examples)}")
    print(f"Output: {output_file}")

    # Show distribution
    policy_counts = {}
    for ex in oracle_examples:
        policy = ex["final_policy"]
        policy_counts[policy] = policy_counts.get(policy, 0) + 1

    print(f"\nPolicy Distribution:")
    for policy, count in policy_counts.items():
        pct = count / len(oracle_examples) * 100
        print(f"  {policy}: {count} ({pct:.1f}%)")
