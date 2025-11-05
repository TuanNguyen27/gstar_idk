# LLM Router - System Architecture

## High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ROUTER MODEL (Gemma-2B)                        │
│  • Analyzes query complexity                                     │
│  • Outputs policy + reasoning                                    │
│  • In-context learning with policy descriptions                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   POLICY DECISION    │
              │  (with reasoning)    │
              └──────────┬───────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    Standard_Query  Complex_Query  Ambiguous_Query
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│               FLEXIBLE POLICY MAPPER                             │
│  • Cost budget rule                                              │
│  • Keyword override rule                                         │
│  • Query length heuristic                                        │
│  • Math detection rule                                           │
│  • Custom business logic                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   gemini-2.5-flash  gemini-2.5-pro  [future models]
          │              │              │
          └──────────────┼──────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FINAL ANSWER                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Training Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA COLLECTION                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │      SimpleQA Benchmark           │
        │    (OpenAI's factual QA set)      │
        └─────────────────┬─────────────────┘
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │      Train/Test Split (80/20)       │
        │  • Train: Oracle generation          │
        │  • Test: Evaluation only             │
        └─────────────────┬─────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────────┐
│              PHASE 2: ORACLE DATASET GENERATION                  │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │   For each query in TRAIN set:    │
        └─────────────────┬─────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
    ┌──────────┐                   ┌──────────┐
    │  Medium  │                   │  Large   │
    │  Model   │                   │  Model   │
    │ (Flash)  │                   │  (Pro)   │
    └────┬─────┘                   └────┬─────┘
         │                              │
         │  Answer                      │  Answer
         │                              │
         └───────────┬──────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   BLIND JUDGING       │
         │  • Judge each answer   │
         │    separately          │
         │  • Prevent self-       │
         │    preference bias     │
         └───────────┬───────────┘
                     │
              ┌──────┴──────┐
              │             │
              ▼             ▼
        Medium_Label    Large_Label
      (Correct/Wrong)  (Correct/Wrong)
              │             │
              └──────┬──────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   9-CELL MATRIX       │
         │  Apply oracle logic    │
         │  to assign policy      │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  QUERY ANALYSIS       │
         │  • Analyze query       │
         │    characteristics     │
         │  • No model answers!   │
         │  • Prevent leakage     │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   ORACLE DATASET      │
         │  (query, policy,      │
         │   analysis)           │
         └───────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│              PHASE 3: TRAINING DATA PREPARATION                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │  DPO FORMAT           │
         │  • Prompt (query)     │
         │  • Chosen (correct)   │
         │  • Rejected (wrong)   │
         └───────────┬───────────┘
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    ┌──────────┐          ┌──────────┐
    │   SFT    │          │   DPO    │
    │  Data    │          │  Data    │
    │ (prompt+ │          │ (triplet)│
    │  chosen) │          │          │
    └────┬─────┘          └────┬─────┘
         │                     │
┌────────┴─────────────────────┴─────────────────────────────────┐
│                 PHASE 4: MODEL TRAINING (Modal)                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │  STAGE 1: SFT         │
         │  • Learn task format   │
         │  • 3 epochs            │
         │  • QLoRA (4-bit)       │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  router_base.model    │
         └───────────┬───────────┘
                     │
         ┌───────────┴───────────┐
         │  STAGE 2: DPO         │
         │  • Prefer correct     │
         │  • 1 epoch            │
         │  • Reference model    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  router_final.model   │
         └───────────────────────┘
                     │
┌────────────────────┴────────────────────────────────────────────┐
│                 PHASE 5: EVALUATION (Test Set)                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │  Run on TEST set      │
         │  • No data leakage!    │
         │  • Compare baseline    │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │   METRICS             │
         │  • Accuracy           │
         │  • Cost savings       │
         │  • Distribution       │
         └───────────────────────┘
```

## 9-Cell Oracle Matrix (Decision Logic)

```
                    LARGE MODEL PERFORMANCE
                 │ Correct │ Incorrect │   IDK   │
    ─────────────┼─────────┼───────────┼─────────┤
    M  Correct   │ STANDARD│  STANDARD │STANDARD │
    E            │  (Both) │  (Medium) │ (Medium)│
    D  ──────────┼─────────┼───────────┼─────────┤
    I  Incorrect │ COMPLEX │ AMBIGUOUS │ COMPLEX │
    U            │(Escalate)│(Both fail)│(Escalate)│
    M  ──────────┼─────────┼───────────┼─────────┤
       IDK       │ COMPLEX │ AMBIGUOUS │AMBIGUOUS│
                 │(Escalate)│(IDK safer)│(Both fail)│
    ─────────────┴─────────┴───────────┴──────────
```

## Component Architecture

### 1. Configuration Layer (`config/`)

```
config/
├── models.py
│   ├── ModelConfig
│   ├── ROUTER_MODEL (Gemma-2B)
│   ├── MEDIUM_MODEL (Flash)
│   ├── LARGE_MODEL (Pro)
│   ├── JUDGE_MODEL (Pro)
│   └── POLICIES (3 policies)
│
└── oracle_matrix.py
    ├── ORACLE_MATRIX (9-cell dict)
    ├── get_policy_label()
    └── get_matrix_rationale()
```

### 2. Data Layer (`data/`)

```
data/
├── benchmark_loader.py
│   ├── BenchmarkExample
│   └── BenchmarkLoader
│       ├── load_simpleqa()
│       ├── load_natural_questions()
│       └── load_custom_jsonl()
│
└── oracle_generator.py
    ├── OracleExample
    └── OracleDatasetGenerator
        ├── generate_model_answers()
        └── judge_and_label()
```

### 3. Training Layer (`training/`)

```
training/
├── prompt_templates.py
│   ├── get_routes_xml()
│   ├── create_router_prompt()
│   ├── create_reasoning_chain()
│   └── create_rejected_reasoning()
│
├── dpo_data_prep.py
│   ├── DPOTrainingExample
│   └── DPODataPreparation
│       ├── convert_oracle_to_dpo()
│       └── create_sft_dataset()
│
├── train_router.py (local)
│   └── RouterTrainer
│       ├── train_sft()
│       └── train_dpo()
│
└── modal_train_router.py (Modal)
    ├── train_sft_stage()
    └── train_dpo_stage()
```

### 4. Deployment Layer (`deployment/`)

```
deployment/
├── router_inference.py
│   ├── RouterInference
│   │   ├── _load_model()
│   │   ├── route_query()
│   │   └── _parse_response()
│   └── EndToEndPipeline
│       └── answer_query()
│
├── modal_inference.py
│   └── RouterModel (Modal class)
│       └── route()
│
├── flexible_policy_map.py
│   └── FlexiblePolicyMapper
│       ├── get_model()
│       ├── create_cost_aware_mapper()
│       ├── create_keyword_based_mapper()
│       └── create_hybrid_mapper()
│
└── policy_map.json (static fallback)
```

### 5. Evaluation Layer (`evaluation/`)

```
evaluation/
└── simpleqa_evaluator.py
    ├── EvaluationResult
    └── SimpleQAEvaluator
        ├── evaluate()
        ├── _evaluate_single()
        ├── _compute_metrics()
        └── compare_with_baseline()
```

### 6. Model Layer (`models/`)

```
models/
└── gemini_client.py
    └── GeminiClient
        ├── generate()
        ├── judge_answer()
        └── analyze_query()
```

## Data Flow

### Training Data Flow

```
SimpleQA.jsonl
    │
    ▼ (split)
┌────────────────┬────────────────┐
│  train.jsonl   │  test.jsonl    │
│  (80%)         │  (20%)         │
└────┬───────────┴────────────────┘
     │
     ▼ (generate answers)
model_answers.jsonl
  {query, ground_truth, medium_answer, large_answer}
     │
     ▼ (judge + label)
oracle_dataset.jsonl
  {query, medium_label, large_label, final_policy, query_analysis}
     │
     ▼ (prepare training)
┌─────────────────┬─────────────────┐
│ sft_train.jsonl │ dpo_train.jsonl │
│ {prompt,        │ {prompt,        │
│  completion}    │  chosen,        │
│                 │  rejected}      │
└─────┬───────────┴─────┬───────────┘
      │                 │
      ▼                 ▼
   SFT Stage        DPO Stage
      │                 │
      └────────┬────────┘
               ▼
        router_final/
```

### Inference Data Flow

```
User Query
    │
    ▼ (tokenize)
Router Input Tokens
    │
    ▼ (generate)
Router Output
  "[REASONING] ... [DECISION] Complex_Query"
    │
    ▼ (parse)
Policy + Reasoning
    │
    ▼ (map)
FlexiblePolicyMapper
  • Check budget
  • Check keywords
  • Apply heuristics
    │
    ▼
Target Model Name
  "gemini-2.5-pro"
    │
    ▼ (call API)
Final Answer
```

## Key Design Decisions

### 1. Why Decoupled Policies?
- **Flexibility**: Change model assignments without retraining
- **Scalability**: Add new models easily
- **Interpretability**: Clear reasoning for each decision

### 2. Why Blind Judging?
- **Bias Prevention**: Judge doesn't favor its own answers (Pro model)
- **Fairness**: Each answer evaluated independently
- **Quality**: More accurate ground truth labels

### 3. Why Query Analysis?
- **No Leakage**: Reasoning based on query, not model answers
- **Generalization**: Router learns to analyze queries, not memorize answers
- **Transferability**: Works on new queries

### 4. Why Two-Stage Training?
- **SFT**: Learn task format and basic routing
- **DPO**: Refine to prefer correct over incorrect reasoning
- **Better than single-stage**: More stable, higher quality

### 5. Why Flexible Mapping?
- **Adaptability**: Handle changing requirements (budget, SLAs)
- **Context-aware**: Consider real-time factors (cost, latency)
- **Business logic**: Encode domain knowledge

## System Requirements

### Development
- Python 3.11+
- 16GB RAM (for local testing)
- GPU optional (for local inference)

### Training (Modal)
- A100 GPU (40GB)
- ~40GB storage
- 3-4 hours compute time

### Production
- T4 GPU (for inference, optional)
- Modal or similar serverless platform
- Gemini API access

## Performance Characteristics

### Router Inference
- Latency: ~100-300ms (local GPU)
- Latency: ~200-500ms (Modal T4)
- Memory: ~6GB (4-bit quantized)
- Throughput: ~10-20 qps (single GPU)

### End-to-End Pipeline
- Latency: Router + Target Model (~1-3s total)
- Cost: $0.001-0.01 per query (depending on routing)
- Accuracy: Target 95%+ of Pro-only baseline

## Monitoring & Observability

### Metrics to Track
1. **Routing Distribution**: % Standard/Complex/Ambiguous
2. **Model Usage**: % Flash vs Pro
3. **Cost**: Running total vs budget
4. **Accuracy**: Per-policy and overall
5. **Latency**: p50, p95, p99

### Logging Strategy
```python
{
  "timestamp": "2025-01-10T12:00:00Z",
  "query": "...",
  "policy": "Complex_Query",
  "model": "gemini-2.5-pro",
  "reasoning": "...",
  "latency_ms": 250,
  "cost_usd": 0.005,
  "answer": "..."
}
```

## Failure Modes & Mitigations

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Router misroutes | Quality drop | Monitor accuracy, retrain |
| API rate limit | Service down | Implement retry with backoff |
| Budget exceeded | Cost overrun | Hard budget cap in mapper |
| Model unavailable | No answer | Fallback to default model |
| Query too long | OOM | Truncate or reject gracefully |

## Extension Points

### Adding New Models
1. Update `policy_map.json` or mapper
2. Add model config to `config/models.py`
3. Implement client in `models/`
4. No router retraining needed!

### Adding New Policies
1. Define policy in `config/models.py`
2. Update oracle matrix logic
3. Regenerate training data
4. Retrain router

### Custom Business Rules
1. Implement custom mapper in `flexible_policy_map.py`
2. Add rule functions
3. Use in production

## Security Considerations

- API keys stored in environment variables
- No PII in training data
- Rate limiting on API calls
- Input validation for queries
- Output sanitization for safety

## Conclusion

This architecture provides:
- ✅ Modular, extensible design
- ✅ Rigorous training methodology
- ✅ Production-ready deployment
- ✅ Cost-effective operation
- ✅ Easy maintenance and updates

The decoupled design allows independent updates to router logic and model assignments, making it suitable for both research and production use.
