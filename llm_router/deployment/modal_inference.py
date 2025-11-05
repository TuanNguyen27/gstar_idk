"""
Modal-based Router Inference
Serverless deployment for router inference.
"""

import modal
import json

app = modal.App("llm-router-inference")

# Inference image with trained model
inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "peft>=0.7.0",
        "accelerate>=0.24.0",
    )
)

# Volume with trained model
volume = modal.Volume.from_name("llm-router-models")

@app.cls(
    image=inference_image,
    gpu="T4",  # Smaller GPU for inference
    volumes={"/vol": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class RouterModel:
    """Modal class for router inference."""

    def __init__(
        self,
        model_path: str = "/vol/router_final",
        base_model_name: str = "google/gemma-2b",
    ):
        self.model_path = model_path
        self.base_model_name = base_model_name

    @modal.enter()
    def load_model(self):
        """Load model when container starts."""
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print(f"Loading router model from {self.model_path}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        # Load adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.model.eval()

        # Load policy map
        policy_map_data = '''
        {
          "Standard_Query": "gemini-2.5-flash",
          "Complex_Query": "gemini-2.5-pro",
          "Ambiguous_Query": "gemini-2.5-flash"
        }
        '''
        self.policy_map = json.loads(policy_map_data)

        print("Router model loaded and ready!")

    @modal.method()
    def route(self, query: str, temperature: float = 0.1) -> dict:
        """
        Route a query to appropriate model.

        Args:
            query: User query
            temperature: Sampling temperature

        Returns:
            Dict with policy, target_model, and reasoning
        """
        import torch
        import re

        # Create prompt
        routes_xml = """<routes>
  <route>
    <name>Standard_Query</name>
    <description>A simple, common, or high-confidence query that can be answered correctly and efficiently by a standard model.</description>
  </route>
  <route>
    <name>Complex_Query</name>
    <description>A complex, niche, or difficult factual query that requires an advanced model to answer correctly.</description>
  </route>
  <route>
    <name>Ambiguous_Query</name>
    <description>A query that is unanswerable, unsafe, or where both standard and advanced models are likely to fail or abstain (IDK).</description>
  </route>
</routes>"""

        prompt = f"""[INST]
You are an expert query routing assistant. Your task is to analyze the user's query and select the most appropriate policy from the list below.

{routes_xml}

<conversation>
{query}
</conversation>

Analyze the query and respond *only* with your reasoning and final decision in the specified JSON format.
[/INST]"""

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("[/INST]")[-1].strip()

        # Parse decision
        decision_match = re.search(r'\[DECISION\]\s*(\w+)', response)
        if decision_match:
            policy = decision_match.group(1)
        else:
            policy = "Standard_Query"

        # Extract reasoning
        reasoning_match = re.search(r'\[REASONING\](.*?)\[DECISION\]', response, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else response

        # Map to target model
        target_model = self.policy_map.get(policy, "gemini-2.5-flash")

        return {
            "policy": policy,
            "target_model": target_model,
            "reasoning": reasoning,
            "raw_response": response,
        }

@app.local_entrypoint()
def test_router(query: str = "What is the wingspan of a Pteranodon?"):
    """Test the router with a sample query."""
    print(f"Testing router with query: {query}\n")

    router = RouterModel()
    result = router.route.remote(query)

    print("=== Routing Result ===")
    print(f"Policy: {result['policy']}")
    print(f"Target Model: {result['target_model']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"\nRaw Response:\n{result['raw_response']}")
