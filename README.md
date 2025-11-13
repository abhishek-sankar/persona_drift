# Model Drift

Code accompanying [Measuring and Controlling Persona Drift in Language Model Dialogs](https://arxiv.org/html/2402.10962v1). It reproduces the self-chat experiments, stores conversation logs, and includes utilities for visualizing persona drift.

## Quick start

1. **Create a Python environment (3.9+).** Either let Conda resolve packages from `environment.yml` (removing the Linux-specific pins if you are on macOS/Windows) or create a fresh virtual environment and install the dependencies you need:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install datasets langdetect nltk  # optional; omit if you only need the bundled subset
   ```

2. **Export your Replicate token.** The Replicate-hosted `meta/llama-4-scout-instruct` model is the default backend, so the token must be present in your environment:
   ```bash
   export REPLICATE_API_TOKEN=...  # required for any Replicate invocation
   ```

3. **Run the self-chat driver.**
   ```bash
   python run.py \
     --model_name meta/llama-4-scout-instruct \
     --agent -1 --user -1 \
     --turns 8 --runs 2 --seed 1
   ```
   The script samples personas, launches a self-chat of eight turns (four rounds), and saves artifacts under `selfchat/` by default. Set `SELFCHAT_DIR=/path/to/output` to override the destination.

Optional dependencies (`datasets`, `langdetect`, `nltk`) enable the full 100-persona benchmark. Without them the script transparently falls back to the bundled subset in `minimal_prompts.py`, which is sufficient for smoke tests or running on air-gapped systems.

## Running targeted experiments

- **Limit to a slice of personas.** The persona indices are zero-based. To approximate a 5 % sample of the full list, run indices `0–4`:
  ```bash
  for idx in $(seq 0 4); do
    python run.py --model_name meta/llama-4-scout-instruct \
      --agent $idx --user $idx --turns 8 --runs 1 --seed 1
  done
  ```
- **Use other chat backends.** Any provider that follows the OpenAI Chat Completions schema can be integrated by swapping `--model_name`. Aliases defined in `utils.py` (for example, `llama2_chat_7B`) transparently resolve to the Replicate model.
- **Resume or inspect existing runs.** Conversation logs are JSON files in `selfchat/`. Re-running with the same arguments appends new data without overwriting previous logs.

You can also skip local generation by downloading precomputed self-chats from [Google Drive](https://drive.google.com/drive/folders/1Iho3KfDbpxrMzEBum_VriKaUuaMji7zu?usp=sharing) and dropping them into `selfchat/`.

## Running on Modal


`modal_app.py` mirrors the CLI workflow inside Modal’s serverless environment.

1. **Create the Replicate secret** (replace the token with your own):
   ```bash
   modal secret create replicate-api-token REPLICATE_API_TOKEN=YOUR_TOKEN
   ```
2. **Provision persistent storage** (one-time setup):
   ```bash
   modal volume create persona-drift-selfchat
   ```
3. **Launch a run** with your desired arguments:
   ```bash
   modal run modal_app.py::run_selfchat --turns 8 --runs 2 --agent -1 --user -1 --seed 1
   ```
   
   Note: If you encounter argument parsing errors, try without the `--` separator, as newer versions of Modal may handle arguments differently.

Outputs are written to `/modal-selfchat` inside the Modal volume, which you can retrieve via `modal volume get persona-drift-selfchat ./local-selfchat`.

## Simplified Baseline for Reproducibility

For baseline reproduction experiments (as described in the project proposal), use the simplified scripts:

### Quick Start (Simplified Baseline)

1. **Install dependencies:**
   ```bash
   pip install -r requirements_baseline.txt
   ```

2. **Authenticate with Replicate:**
   ```bash
   export REPLICATE_API_TOKEN=your_token_here
   # optional sanity check
   curl https://api.replicate.com/v1/account \
     -H "Authorization: Bearer $REPLICATE_API_TOKEN"
   ```

3. **Run baseline experiments with different decoding strategies (Replicate-only backend):**
   ```bash
   # Greedy decoding
   python baseline_run.py \
     --model_name meta/llama-4-scout-instruct \
     --agent 0 --user 5 \
     --turns 20 \
     --decoding greedy \
     --seed 42
   
   # Nucleus sampling
   python baseline_run.py \
     --model_name meta/llama-4-scout-instruct \
     --agent 0 --user 5 \
     --turns 20 \
     --decoding nucleus \
     --top_p 0.9 --temperature 0.7 \
     --seed 42
   
   # Best-of-n with persona/context re-ranking
   python baseline_run.py \
     --model_name meta/llama-4-scout-instruct \
      --agent 0 --user 5 \
     --turns 20 \
     --decoding best_of_n \
     --best_of_n 5 \
     --alpha 1.0 --beta 0.5 --gamma 0.0 \
     --seed 42
   ```

   By default, conversations are saved to `selfchat/` as pickled dictionaries containing the full history, per-turn latency/token telemetry, and (when applicable) best-of-n scoring metadata.

4. **Evaluate metrics and telemetry:**
   ```bash
   python evaluate.py \
     --input selfchat \
     --output results.csv \
     --early_turns 3 \
     --use_gpu         # optional: speeds up embedding/NLI/BERTScore computation
   ```
   This writes a CSV (plus a detailed JSON) summarising persona consistency, contradiction rate, drift index, conversation quality, total/average token counts, per-turn latency, and best-of-n selection statistics.

### Features of Simplified Baseline

- **20-Persona Subset**: Curated slice of the original benchmark (`selected_personas.py`) spanning affect, style, and memory behaviours.
- **Replicate-Only Backend**: All decoding strategies call the Replicate REST API; no OpenAI or local inference required.
- **Decoding Strategies**: Greedy, nucleus, and best-of-n sampling with persona/context/length re-ranking.
- **Extended Horizons**: Supports long conversations (20+ turns) with automatic resume support.
- **Comprehensive Metrics**:
  - Persona consistency (Sentence-BERT similarity)
  - Contradiction rate (RoBERTa-MNLI)
  - Drift index (temporal divergence vs. early turns)
  - Conversation quality (BERTScore)
- **Telemetry Logging**: Per-turn latency and token estimates, aggregated summaries, and detailed best-of-n candidate scores for ablations.
- **Evaluation Pipeline**: `evaluate.py` produces CSV/JSON tables ready for plotting or statistical comparison across decoding settings.

## Plotting persona drift

Use `plot_convergence.ipynb` to reproduce the figures from the paper. Point the notebook at the conversation logs generated in `selfchat/` (or the Modal volume download).

For simplified baseline results, use `evaluate.py` to generate CSV/JSON outputs that can be easily plotted.

## Citation

```
@article{li2024measuring,
  title={Measuring and Controlling Persona Drift in Language Model Dialogs},
  author={Li, Kenneth and Liu, Tianle and Bashkansky, Naomi and Bau, David and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  journal={arXiv preprint arXiv:2402.10962},
  year={2024}
}
```
