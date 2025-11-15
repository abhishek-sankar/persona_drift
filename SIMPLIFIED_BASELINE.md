# Summary: Simplified Baseline Implementation

## What Was Done

I've simplified the persona drift repository to focus on baseline reproduction as described in your proposal. Here's what was created:

### New Files Created

1. **`selected_personas.py`**: Curated subset of 20 personas from the benchmark with original judge functions.
2. **`baseline_run.py`**: Simplified experiment runner with:
   - Multiple decoding strategies (greedy, nucleus, best-of-n with persona/context re-ranking)
   - Extended turn support (20+ turns) and resume capability
   - Replicate-only inference (no OpenAI/local backends required)
3. **`metrics.py`**: Comprehensive metrics implementation:
   - Persona consistency (embedding similarity using Sentence-BERT)
   - Contradiction rate (NLI-based using RoBERTa-MNLI)
   - Drift index (temporal divergence from early turns)
   - Conversation quality (BERTScore)
4. **`evaluate.py`**: Script to compute metrics from conversation logs
5. **`requirements_baseline.txt`**: Dependencies for baseline experiments
6. **`ANALYSIS.md`**: Detailed analysis of what was simplified

### Key Simplifications

1. **20-persona subset**: Diverse selection spanning affect, style, and memory behaviours.
2. **Removed complex probe mechanism**: Direct metric computation on generated responses.
3. **Decoding strategy suite**: Greedy, nucleus sampling, and best-of-n with persona/context-aware re-ranking.
4. **Telemetry logging**: Per-turn latency and token estimates, plus detailed best-of-n candidate scores.
5. **Replicate-only inference**: Keeps setup lightweight and avoids local GPU requirements.

## How to Use

### 1. Install Dependencies
```bash
pip install -r requirements_baseline.txt
```

### 2. Run Baseline Experiments

Ensure your Replicate token is exported before running any experiments:
```bash
export REPLICATE_API_TOKEN=your_token_here
```

**Greedy decoding:**
```bash
python baseline_run.py \
  --model_name meta/llama-4-scout-instruct \
  --agent 0 --user 5 \
  --turns 20 \
  --decoding greedy \
  --seed 42
```

**Nucleus sampling:**
```bash
python baseline_run.py \
  --model_name meta/llama-4-scout-instruct \
  --agent 0 --user 5 \
  --turns 20 \
  --decoding nucleus \
  --top_p 0.9 --temperature 0.7 \
  --seed 42
```

**Best-of-n (persona/context re-ranking):**
```bash
python baseline_run.py \
  --model_name meta/llama-4-scout-instruct \
  --agent 0 --user 5 \
  --turns 20 \
  --decoding best_of_n \
  --best_of_n 5 \
  --alpha 1.0 --beta 0.5 --gamma 0.0 \
  --seed 42
```

### 3. Evaluate Results
```bash
python evaluate.py \
  --input selfchat \
  --output results.csv \
  --early_turns 3 \
  --use_gpu       # optional: speeds up embedding/NLI/BERTScore metrics
```
The script now emits:
- Per-run metric means/std for persona consistency, contradiction rate, drift index, and conversation quality.
- Total/average latency and token estimates.
- Best-of-n selection statistics (persona/context similarity, length penalty, candidate counts).

## Recommended Next Steps

1. **Run ablations** across decoding strategies and persona pairs; aggregate with `evaluate.py`.
2. **Plot temporal trends** (e.g., drift index vs. turn) using the per-turn arrays saved in the JSON output.
3. **Leverage telemetry** to study compute trade-offs (latency vs. adherence).
4. **Layer additional mitigations** (e.g., Approach B in the proposal) on top of this baseline pipeline.

## Questions?

- The original `run.py` is still available if you need the full 100-persona benchmark
- The simplified baseline focuses on reproducibility and clear metrics
- All conversation logs are saved as pickle files in `selfchat/` directory

