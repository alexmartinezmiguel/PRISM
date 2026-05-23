
This repository contains the code for the paper **PRISM: Disentangling Position Bias in Generative Listwise Recommendation via Plackett-Luce Modeling**. 

---

## Overview

```
code/
├── DockerFile                             # GPU container (TF 2.14 + PyTorch cu118)
├── requirements.txt                       # Classical ML / TensorFlow dependencies
├── llm_requirements.txt                   # LLM / Transformers dependencies
│
├── llm_recommender_permutations.py        # [Step 1] Generate multi-permutation rankings
├── rise.py                                # [Baseline] RISE iterative selection
├── stella_debiasing.py                    # [Baseline] STELLA Bayesian debiasing
├── two_stage_pl_interaction_debiasing.py  # [Ours] Two-stage PL interaction model
├── majority_voting_debiasing.py           # [Baseline] Borda / greedy voting
│
└── utils/
    ├── data_utils.py      # Permutation helpers and prompt builder
    ├── training_utils.py  # Ranking parsers and quality checks
    └── misc_utils.py      # Logging setup
```

## Environment

### Option A — Docker (recommended)

```bash
docker build -t llm-rec-debiasing -f DockerFile .
docker run --gpus all -v /path/to/your/data:/workspace/data \
           -v /path/to/checkpoints:/workspace/mind_output \
           -e HF_TOKEN=$HF_TOKEN \
           -it llm-rec-debiasing
```

### Option B — Conda / pip

```bash
# Classical ML stack (TF, scikit-learn, …)
pip install -r requirements.txt

# LLM stack (transformers, trl, peft, …)
pip install -r llm_requirements.txt

# Microsoft Recommenders (no extra deps)
pip install --no-deps recommenders[gpu]

# PyTorch with CUDA 11.8
pip3 install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
```

> **Note:** Set your Hugging Face token before running any script:
> ```bash
> export HF_TOKEN=<your_token>
> ```

---

## Workflow

### Step 1 — Generate multi-permutation rankings

Run the finetuned LLM over the validation/test set under multiple candidate
permutations to collect the raw data used by all debiasing methods.

```bash
# Random permutations
python llm_recommender_permutations.py \
    --model-path   <path/to/checkpoint> \
    --train-file   <path/to/train.csv> \
    --output-dir   <path/to/output/> \
    --num-permutations 100 \
    --type random \
    --batch-size 4
```

Output: `data/<dataset>/recommendation_lists.csv`

---

### Step 2a — Two-Stage PRISM Debiasing (proposed method)

Estimates shared additive (β) and multiplicative (γ) position-bias parameters
from offline data, then debiases each evaluation impression.

```bash
python two_stage_pl_interaction_debiasing.py \
    --offline_data_path  <offline-data> \
    --eval_data_path     <inference-data> \
    --n_offline  1000 \
    --n_eval     -1 \
    --R          10 \
    --n_samples  2 3 5 7 10 \
    --n_repeats  10 \
    --n_iters_bcd 30 \
    --sigma_q    5.0 \
    --sigma_beta 5.0 \
    --sigma_gamma 5.0 \
    --tol        1e-5 \
    --seed       42 \
    --output_dir results/pl_interaction/
```

---

### Step 2b — STELLA Baseline

```bash
python stella_debiasing.py \
    --probing_data_path  <offline-data> \
    --eval_data_path     <inference-data> \
    --output_dir         results/stella/ \
    --n_offline 1000 \
    --n_samples 2 3 5 7 10 \
    --n_repeats 10
```

---

### Step 2c — Majority Voting Baselines

```bash
# Borda count
python majority_voting_debiasing.py \
    --eval_data_path  <inference-data> \
    --indices_path    results/pl_interaction/sample_indices_*.npy \
    --method borda \
    --n_samples 2 3 5 7 10 \
    --n_repeats 20 \
    --output_dir results/majority_voting/
```

---

### Step 2d — RISE Baseline

```bash
python rise.py \
    --model-path <path/to/checkpoint> \
    --test-file  <path/to/test.csv> \
    --output-dir <path/to/output/> \
    --N 1 \
    --num-permutations 5
```

## Repository Structure Notes

The `utils/` package contains only the functions actively used by the evaluation
pipeline:

| File | Kept functions |
|------|----------------|
| `utils/misc_utils.py` | `setup_logging` |
| `utils/training_utils.py` | `_coerce_to_text`, `analyze_ranking_quality`, `contains_invalid_h_ids` |
| `utils/data_utils.py` | `permute_candidates`, `rebuild_prompt_candidates`, `map_labels_to_permuted`, `build_chat_template_dataset_PL` |

The active prompt format in `build_chat_template_dataset_PL` is JSON-only
output `{"ranking": ["C#", ...]}` in a single user turn (compatible with Gemma,
Llama, and Qwen chat templates).
