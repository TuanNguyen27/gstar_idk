#!/bin/bash
# Run SLM baseline inference for all 3 models in parallel

echo "=========================================="
echo "Running SLM Baseline Inference (Parallel)"
echo "=========================================="
echo ""
echo "Models:"
echo "  1. google/gemma-2-9b-it (9B params)"
echo "  2. google/gemma-2-2b-it (2B params)"
echo "  3. Qwen/Qwen2.5-1.5B-Instruct (1.5B params)"
echo ""
echo "Dataset: SimpleQA-Verified (1000 examples)"
echo ""

# Launch all 3 jobs in parallel using modal run
echo "Launching job 1/3: Gemma-2-9B..."
modal run scripts/modal_slm_baseline.py --model google/gemma-2-9b-it &
PID1=$!

echo "Launching job 2/3: Gemma-2-2B..."
modal run scripts/modal_slm_baseline.py --model google/gemma-2-2b-it &
PID2=$!

echo "Launching job 3/3: Qwen-1.5B..."
modal run scripts/modal_slm_baseline.py --model Qwen/Qwen2.5-1.5B-Instruct &
PID3=$!

echo ""
echo "All jobs launched!"
echo "  Job 1 PID: $PID1 (Gemma-9B)"
echo "  Job 2 PID: $PID2 (Gemma-2B)"
echo "  Job 3 PID: $PID3 (Qwen-1.5B)"
echo ""
echo "Waiting for all jobs to complete..."
echo ""

# Wait for all jobs
wait $PID1
STATUS1=$?
echo "✓ Job 1 (Gemma-9B) completed with status: $STATUS1"

wait $PID2
STATUS2=$?
echo "✓ Job 2 (Gemma-2B) completed with status: $STATUS2"

wait $PID3
STATUS3=$?
echo "✓ Job 3 (Qwen-1.5B) completed with status: $STATUS3"

echo ""
echo "=========================================="
echo "All jobs complete!"
echo "=========================================="
echo ""
echo "Results saved to Modal volume: slm-baseline-results"
echo ""
echo "To download results:"
echo "  modal volume get slm-baseline-results /vol/google_gemma-2-9b-it_results.jsonl"
echo "  modal volume get slm-baseline-results /vol/google_gemma-2-2b-it_results.jsonl"
echo "  modal volume get slm-baseline-results /vol/Qwen_Qwen2.5-1.5B-Instruct_results.jsonl"
