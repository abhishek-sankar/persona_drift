# Persona Drift Repo Analysis & Simplification Plan

## Current State

### What the repo does:
- Reproduces Li et al. (2024) persona drift experiments
- Runs self-chat between two personas (agent and user)
- Uses 100+ complex personas with custom judge functions
- Measures persona adherence via probe questions at each turn
- Supports Replicate API and Modal cloud execution

### What's overdone for your needs:
1. **100+ personas**: You only need 3-5 simple personas for baseline reproduction
2. **Complex judge functions**: Many personas have intricate evaluation logic (e.g., checking for specific word patterns, sentiment analysis, language detection)
3. **Modal integration**: Cloud execution is nice but not needed for baseline reproduction
4. **Probe mechanism**: The current probe system is complex - you can simplify this
5. **Limited decoding support**: Only supports basic sampling, no beam search or best-of-n

### What you need (from your proposal):
1. **Baseline reproduction** from Li et al. 2024
2. **Extended horizons**: 20+ turns (current default is 8-16)
3. **Multiple decoding strategies**:
   - Greedy (deterministic)
   - Nucleus sampling (p=0.9, T∈{0.7,0.9})
   - Beam search (beams∈{3,5})
   - Best-of-n with persona-aware re-ranking (n∈{3,5})
4. **Proper metrics**:
   - Persona consistency (embedding similarity)
   - Contradiction rate (NLI-based)
   - Drift Index (temporal divergence)
   - Conversation quality (BERTScore)

## Simplification Plan

### Step 1: Create simplified personas
- Use 3-5 simple personas from `minimal_prompts.py`
- Focus on clear, measurable personas (e.g., "You are a friendly teacher", "You are a doctor")

### Step 2: Add decoding strategy support
- Extend `run.py` to support different decoding methods
- Add parameters for temperature, top_p, beam size, etc.

### Step 3: Implement proper metrics
- Create `metrics.py` with:
  - Persona consistency using sentence embeddings (Sentence-BERT)
  - Contradiction detection using NLI models (RoBERTa-MNLI)
  - Drift index calculation
  - BERTScore for conversation quality

### Step 4: Support multiple model backends
- Local models (via transformers library)
- OpenAI API (properly)
- Keep Replicate as optional

### Step 5: Simplify evaluation
- Remove complex probe mechanism
- Use direct metrics on conversation turns
- Create evaluation script that computes all metrics

## Files to Create/Modify

1. **`baseline_run.py`**: Simplified version of `run.py` with:
   - Few personas
   - Multiple decoding strategies
   - Extended turn support (20+)
   - Proper model backend support

2. **`metrics.py`**: New file with metric implementations

3. **`evaluate.py`**: Script to compute metrics from conversation logs

4. **`simple_personas.py`**: 3-5 simple personas for baseline

5. **Update `README.md`**: Add instructions for baseline reproduction

