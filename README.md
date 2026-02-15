# *AmongUs*: A Sandbox for Measuring and Detecting Agentic Deception

> **Fork of [7vik/AmongUs](https://github.com/7vik/AmongUs)** — extended with structured prompt engineering, LLM-based behavioral evaluation, cross-dataset linear probe pipelines, and expanded model support.

This project simulates the multiplayer game "Among Us" with LLM agents and studies how models learn to express lying and deception, while evaluating the effectiveness of AI safety techniques to detect and control out-of-distribution deception.

---

## What's New in This Fork

### Structured Prompt System
We redesigned the agent prompt architecture with three interchangeable prompt modules:

| Module | Description |
|--------|-------------|
| **`neutral_prompts.py`** (active) | Structured output (`[World State Ledger]`, `[Thinking Process]`, `[Action]`), anti-hallucination rules (hard memory vs social memory), line-of-sight constraints, 3-stage meeting phases (Testimony → Accusation & Defense → Final Arguments), discussion roles (Prosecutor, Detective, Defender, Bystander, Counter-Attacker), ghost task system, spatial adjacency checks, and public alibi generation after kills. |
| **`personality_prompts.py`** | Strategy-driven prompts with personality archetypes (Strategist, Manipulator, Lone Wolf, etc.). |
| **`original_prompts.py`** | Minimal prompt baseline closest to the upstream version. |

### LLM-Based Behavioral Evaluation
- New `evals_prompts.py` defines structured evaluation prompts that score each agent action on four dimensions: **Awareness**, **Lying**, **Deception**, and **Planning** (1–10 scale).
- `eval.py` runs async evaluation with rate limiting, exponential backoff, and configurable evaluator models (default: `llama-3.3-70b-instruct` via OpenRouter).
- `eval_qwen3_4b.ipynb` provides an end-to-end evaluation notebook for Qwen3-4B game logs.

### All-Layers Linear Probe Pipeline
- New `all_layers_cache_train_eval.py` provides a single end-to-end script that caches activations → trains linear probes → evaluates cross-dataset generalization and generates ROC curves.
- Added Phi-4 and Llama-3.3-70B model configs with layer-specific activation caching.
- Cross-dataset evaluation: probes trained on one dataset (e.g., TruthfulQA) are tested against AmongUs, RepEng, DishonestQA, and others.

### Expanded Model Support
`main.py` now supports a wide range of models via OpenRouter:
- `microsoft/phi-4`, `qwen/qwen3-235b-a22b`, `qwen/qwen-2.5-7b-instruct`
- `google/gemma-3-4b-it`, `deepseek/deepseek-r1`, `deepseek/deepseek-r1-distill-llama-70b`
- `anthropic/claude-3.7-sonnet:thinking`, `openai/o3`, `openai/o4-mini-high`
- And more — see `main.py` for the full list.

### Experiment Automation
- `run_games.sh` defines batch experiments (100 games each) for model matchups: Phi-4 vs Llama-3.3-70B, Phi-4 vs Phi-4, etc.
- `main.py` adds `--tournament_style` (random / 1on1), `--crewmate_llm`, and `--impostor_llm` flags for fine-grained control.
- Experiment logs auto-indexed as `YYYY-MM-DD_exp_N`.

### New Notebooks
Exploratory notebooks for activation caching, data deduplication, GPU utilization, HuggingFace dataset loading, local model speed testing, and OpenRouter API usage.

---

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/karan-13-hub/among-us.git
   cd among-us
   ```

2. Set up the environment:
   ```bash
   conda create -n amongus python=3.10
   conda activate amongus
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Run Games

To run games of various LLMs playing against each other:

```bash
python main.py --crewmate_llm microsoft/phi-4 --impostor_llm meta-llama/llama-3.3-70b-instruct
```

You will need a `.env` file with an [OpenRouter](https://openrouter.ai/) API key.

To run batch experiments:

```bash
bash run_games.sh
```

This runs predefined 100-game matchups across model pairings (Phi-4, Llama-3.3-70B combinations).

Alternatively, you can download 400 full-game logs (for `Phi-4-15b` and `Llama-3.3-70b-instruct`) and 810 game summaries from the [HuggingFace](https://huggingface.co/) dataset.

## LLM-Based Evaluation

To evaluate agent actions on Awareness, Lying, Deception, and Planning:

```bash
bash evaluations/run_evals.sh
```

Or run the evaluation notebook for a specific model:

```
evaluations/eval_qwen3_4b.ipynb
```

This scores each game step using an LLM evaluator and outputs structured JSONL results.

## Caching Activations

To cache LLM activations for linear probe analysis:

```bash
python linear-probes/cache_activations.py --dataset <dataset_name>
```

This is computationally expensive — GPU recommended. Use `configs.py` to configure the model (Phi-4, Llama-3, GPT-2) and target layers.

## Training & Evaluating Linear Probes

**Option A — Individual steps:**

```bash
python linear-probes/train_probes.py
python linear-probes/evaluate_probes.py
```

**Option B — End-to-end pipeline (recommended):**

```bash
python linear-probes/all_layers_cache_train_eval.py
```

This caches activations, trains probes, and evaluates cross-dataset generalization (AmongUs, TruthfulQA, DishonestQA, RepEng) in a single run, generating ROC curves and metrics.

Results are stored in `linear-probes/results/`.

## Project Structure

```
.
├── among-agents/                # Core game engine and agent code
│   ├── amongagents/
│   │   ├── agent/               # LLM agent implementation
│   │   │   ├── agent.py         # Main agent class
│   │   │   ├── neutral_prompts.py    # Structured prompt system (active)
│   │   │   ├── personality_prompts.py # Personality-based prompts
│   │   │   └── original_prompts.py   # Baseline prompts
│   │   ├── envs/                # Game environment, map, player, tasks
│   │   ├── evaluation/          # Controlled and end-to-end evaluation
│   │   └── UI/                  # Map visualization (Streamlit & web)
│   └── notebooks/               # Game run notebooks and sample logs
├── evaluations/                 # LLM-based behavioral evaluation
│   ├── eval.py                  # Async evaluation pipeline
│   ├── evals_prompts.py         # Evaluation prompt definitions
│   ├── eval_qwen3_4b.ipynb      # Qwen3-4B evaluation notebook
│   └── run_evals.sh             # Batch evaluation script
├── human_trials/                # Web app for human-vs-AI trials
├── linear-probes/               # Activation probing for deception detection
│   ├── all_layers_cache_train_eval.py  # End-to-end probe pipeline
│   ├── cache_activations.py     # Activation caching
│   ├── configs.py               # Model configs (Phi-4, Llama-3, GPT-2)
│   ├── train_probes.py          # Probe training
│   ├── evaluate_probes.py       # Probe evaluation
│   ├── probe_datasets.py        # Dataset definitions
│   └── plots.py                 # ROC and metric visualization
├── notebooks/                   # Exploratory notebooks
├── tests/                       # Unit tests
├── main.py                      # Game runner entry point
├── utils.py                     # Experiment setup and log utilities
├── run_games.sh                 # Batch experiment script
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker environment setup
└── .gitignore                   # Excludes expt-logs, results, checkpoints
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to contribute to this project.

## License

This project is licensed under CC0 1.0 Universal — see [LICENSE](LICENSE).

## Acknowledgments

- Forked from [7vik/AmongUs](https://github.com/7vik/AmongUs) by Satvik Golechha.
- Game logic adapted from [AmongAgents](https://github.com/AtaishNehra/Among_Agents).
