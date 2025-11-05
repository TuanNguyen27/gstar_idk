# LLM Router: Performance-Aligned Model Selection

A flexible, efficient router that intelligently directs queries to the optimal LLM (e.g., Gemini Flash vs Pro) based on query complexity, reducing costs while maintaining accuracy.

## 🎯 Key Features

- **Decoupled Architecture**: Router learns policies, not hard-coded models
- **In-Context Routes**: Adaptable to new models without retraining
- **Oracle-Based Training**: Ground truth from comparative model performance
- **Bias Mitigation**: Blind judging prevents self-preference
- **Two-Stage Training**: SFT + DPO for optimal routing decisions
- **Modal Integration**: Serverless training and inference

## 🏗️ Architecture

```
Query → Router (Gemma-2B) → Policy → Model Map → Target LLM → Answer
                ↓
    Standard_Query  → gemini-2.5-flash
    Complex_Query   → gemini-2.5-pro
    Ambiguous_Query → gemini-2.5-flash
```

## 📦 Installation

```bash
# Clone repository
cd llm_router

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your-key"
export MODAL_TOKEN_ID="your-modal-token"
export MODAL_TOKEN_SECRET="your-modal-secret"
```

## 🚀 Quick Start

### Step 1: Generate Oracle Dataset

```bash
python scripts/generate_oracle_dataset.py \
  --benchmark simpleqa \
  --input-file ./data/simpleqa.jsonl \
  --output-dir ./data/generated \
  --limit 1000 \
  --gemini-api-key $GEMINI_API_KEY
```

### Step 2: Prepare Training Data

```bash
python scripts/prepare_training_data.py \
  --oracle-file ./data/generated/oracle_dataset.jsonl \
  --output-dir ./data/training
```

### Step 3: Train Router (Modal)

```bash
cd training
modal run modal_train_router.py \
  --model-name google/gemma-2b \
  --sft-data-path ../data/training/sft_train.jsonl \
  --dpo-data-path ../data/training/dpo_train.jsonl \
  --sft-epochs 3 \
  --dpo-epochs 1
```

### Step 4: Test Router

```bash
# Routing only
python scripts/test_router.py \
  --model-path ./router_checkpoints/router_final \
  --query "What is the wingspan of a Pteranodon?"

# End-to-end (routing + answer)
python scripts/test_router.py \
  --model-path ./router_checkpoints/router_final \
  --query "What is the wingspan of a Pteranodon?" \
  --end-to-end \
  --gemini-api-key $GEMINI_API_KEY
```

### Step 5: Deploy (Modal)

```bash
cd deployment
modal deploy modal_inference.py

# Test deployed router
modal run modal_inference.py --query "What is 2+2?"
```

## 📊 Oracle Logic (9-Cell Matrix)

| Medium Model | Large Model | Policy           | Rationale                     |
|--------------|-------------|------------------|-------------------------------|
| Correct      | Correct     | Standard_Query   | Both correct → optimize cost  |
| Correct      | Incorrect   | Standard_Query   | Medium correct → use medium   |
| Correct      | IDK         | Standard_Query   | Medium correct is best        |
| Incorrect    | Correct     | Complex_Query    | Classic escalation            |
| Incorrect    | Incorrect   | Ambiguous_Query  | Both failed → don't escalate  |
| Incorrect    | IDK         | Complex_Query    | IDK safer than incorrect      |
| IDK          | Correct     | Complex_Query    | Large knows answer            |
| IDK          | Incorrect   | Ambiguous_Query  | IDK was safer                 |
| IDK          | IDK         | Ambiguous_Query  | Both failed → don't escalate  |

## 🔧 Configuration

### Models (config/models.py)

- **Router**: `google/gemma-2b` (or `Qwen/Qwen2.5-1.5B`)
- **Medium**: `gemini-2.5-flash`
- **Large**: `gemini-2.5-pro`
- **Judge**: `gemini-2.5-pro`

### Policies (deployment/policy_map.json)

```json
{
  "Standard_Query": "gemini-2.5-flash",
  "Complex_Query": "gemini-2.5-pro",
  "Ambiguous_Query": "gemini-2.5-flash"
}
```

Update this file to change model assignments without retraining!

## 📁 Project Structure

```
llm_router/
├── config/              # Model and policy configurations
│   ├── models.py
│   └── oracle_matrix.py
├── data/                # Dataset loaders and generators
│   ├── benchmark_loader.py
│   └── oracle_generator.py
├── training/            # Training pipelines
│   ├── prompt_templates.py
│   ├── dpo_data_prep.py
│   ├── train_router.py
│   └── modal_train_router.py
├── deployment/          # Inference and deployment
│   ├── router_inference.py
│   ├── modal_inference.py
│   └── policy_map.json
├── scripts/             # Helper scripts
│   ├── generate_oracle_dataset.py
│   ├── prepare_training_data.py
│   └── test_router.py
└── README.md
```

## 🔬 Research Details

### Phase 1: Foundation
- Define models, policies, and oracle logic
- Implement 9-cell matrix for ground truth labeling

### Phase 2: Oracle Dataset Generation
1. Load benchmark data (SimpleQA, Natural Questions, custom)
2. Generate answers from Medium and Large models
3. Blind judging to prevent bias
4. Query analysis to prevent data leakage
5. Apply oracle matrix to label policies

### Phase 3: Training Data Preparation
1. Convert oracle dataset to DPO format
2. Generate reasoning chains (CoT)
3. Create chosen/rejected pairs for preference learning

### Phase 4: Model Training
1. **Stage 1 (SFT)**: Teach task and format (3 epochs)
2. **Stage 2 (DPO)**: Refine preferences (1 epoch)
3. QLoRA (4-bit) for efficiency

### Phase 5: Deployment
1. Load trained router
2. Parse routing decisions
3. Map policies to models
4. Execute on target LLM

## 🎓 Key Innovations

1. **Decoupled Logic**: Policies separate from models
2. **Blind Judging**: Prevents judge self-preference
3. **Query Analysis**: Prevents data leakage in reasoning
4. **In-Context Routes**: XML-based route descriptions
5. **DPO Training**: Preference optimization for better decisions

## 📈 Expected Benefits

- **Cost Reduction**: 40-60% by routing simple queries to Flash
- **Quality Maintenance**: Complex queries escalated to Pro
- **Flexibility**: Change models via JSON, no retraining
- **Efficiency**: Small router (2B params) with fast inference

## 🤝 Contributing

This is a research implementation. Key areas for improvement:

- Additional benchmarks (MMLU, TruthfulQA)
- Alternative router models (Phi-2, Mistral-7B)
- Multi-model routing (3+ models)
- Cost-accuracy tradeoff analysis
- Production optimization

## 📄 License

MIT License - See LICENSE file

## 🙏 Acknowledgments

Inspired by:
- Arch-Router architecture
- Google's Gemini model family
- TRL library for RLHF/DPO
- Modal for serverless compute

## 📧 Contact

For questions or collaborations, open an issue on GitHub.
