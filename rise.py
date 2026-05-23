#!/usr/bin/env python3
"""
rise.py

RISE (Ranking by Iterative Selection) inference for LLM-based news recommendation.

At each step the LLM selects the single best item from the remaining candidate set,
building a full ranking from top to bottom. This iterative approach sidesteps the
need to produce a complete ordered list in one shot.

Pipeline:
  1. Load the test CSV and generate `num_permutations` random shuffles per
     impression (saved/reloaded inside --output-dir for reproducibility).
  2. Run RISE@N iterative inference: at each step, prompt the LLM to pick N items
     from the remaining candidates; repeat until all candidates are ranked.
  3. Map rankings back to canonical (original) C-space using the stored permutation.
  4. Compute NDCG@5, NDCG@10, AUC, MRR; write per-checkpoint CSV.

Usage
-----
    python rise.py \\
        --model-path   <path/to/checkpoint or HF model ID> \\
        --test-file    <path/to/test.csv> \\
        --output-dir   <path/to/output/> \\
        --N 1 \\
        --num-permutations 5

Environment
-----------
    HF_TOKEN  — Hugging Face access token (required if loading gated models).
                Set with: export HF_TOKEN=<your_token>
"""
# Standard library
import os
import re
import gc
import json
import random
import time
import argparse

# Third-party
import numpy as np #type: ignore
import pandas as pd #type: ignore
import torch #type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

# Workspace root is expected to be /workspace (Docker) or the repo root locally.
os.chdir('/workspace')

# ---------------------------------------------------------------------------
# GPU and cache configuration
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# HF_TOKEN must be set in the environment before running this script.
# Example: export HF_TOKEN=<your_token>
if not os.environ.get('HF_TOKEN'):
    raise EnvironmentError("HF_TOKEN environment variable is not set.")
os.environ["TORCHDYNAMO_DISABLE"] = "1"  # required for Gemma models
os.environ['HF_HOME'] = 'hf_cache'
os.environ['TRANSFORMERS_CACHE'] = 'hf_cache/hub'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'hf_cache/hub'
cache_dir = 'hf_cache/hub'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'

os.makedirs(cache_dir, exist_ok=True)

from utils.misc_utils import setup_logging
from utils.training_utils import analyze_ranking_quality, contains_invalid_h_ids
from utils.data_utils import permute_candidates, rebuild_prompt_candidates, map_labels_to_permuted
from recommenders.models.deeprec.deeprec_utils import cal_metric # type: ignore

logger = setup_logging()

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', type=str, required=True,
                        help='Full path to the model checkpoint directory (or a Hugging Face model ID)')
    parser.add_argument('--test-file', type=str, required=True,
                        help='Path to the test CSV (must have columns: impression_id, history, candidate, label)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory where RISE results and the permuted test set are written')
    parser.add_argument('--N', type=int, default=1,
                        help='Number of items to select per iteration (RISE@N)')
    parser.add_argument('--num-permutations', type=int, default=5,
                        help='Number of random candidate permutations per impression')
    return parser.parse_args()

def generate_RISE_prompt(history: list, candidates: list, N: int = 1) -> str:
    """
    Generates the prompt for a single iteration of the RISE algorithm.
    """

    system_prompt = (
        "You serve as a personalized news recommendation system. Understand the User's History News, and then select exactly 1 news article from the Candidate News that best match the user's interests (i.e., similarity/continuity with the history)."
        "Output Format: Output your answer strictly in valid JSON with this exact structure and field order:\n\n"
        "{\n"
        '  "ranking": ["C#"]\n'
        "}\n\n"
        "Important:\n"
        "- The JSON must be syntactically valid (no markdown, no extra text, no commentary).\n"
        "- The ranking array must contain exactly 1 item.\n"
        "- Do not include anything outside the JSON block."
    )
    
    candidates_str = '\n'.join(candidates)
    user_prompt = (
            f"User's History News: {history}\n"
            f"Candidate News: {candidates_str}\n"
            f"Important: Imperative to only include 1 id from the Candidate News set.\n"
            )
    # Llama and Qwen models: system role, everything in the user turn
    # messages = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": user_prompt}
    # ]

    # Gemma-style: no system role, everything in the user turn
    messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]

    return messages

def parse_selected_items(response_text: str, current_candidates: list, N: int) -> list:
    """
    Extracts the chosen items from the LLM's raw text response.
    Expects the LLM to output identifiers like 'C1', 'C2', etc.
    Matches these identifiers to the full candidate strings in current_candidates.
    """
    selected = []
    
    # 1. Extract all occurrences of "C" followed by digits (e.g., C1, C2, C10)
    # Using re.IGNORECASE to handle "c1" or "c10" just in case the model lowercases it.
    matches = re.findall(r'C\d+', response_text, re.IGNORECASE)
    
    # 2. Normalize matches to uppercase and remove duplicates while preserving order
    seen = set()
    unique_matches = []
    for match in matches:
        m = match.upper()
        if m not in seen:
            seen.add(m)
            unique_matches.append(m)
            
    # 3. Find the corresponding full candidate strings
    for match in unique_matches:
        for candidate in current_candidates:
            # Match the prefix strictly (e.g., "C1:" or "C1 ") so "C1" doesn't accidentally trigger "C10"
            if candidate.startswith(f"{match}:") or candidate.startswith(f"{match} "):
                if candidate not in selected:
                    selected.append(candidate)
                break # Found the candidate for this match, move to the next match
                
        # Stop early if we've reached the required N items
        if len(selected) >= N:
            break
            
    # 4. Fallback to prevent infinite loops if the LLM hallucinates or returns no valid ID
    if not selected:
        selected = random.sample(current_candidates, min(N, len(current_candidates)))
        
    return selected[:N]

def build_rise_test_set(test_df: pd.DataFrame, save_path: str, num_permutations: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    For each impression in test_df, generate `num_permutations` random shuffles
    of the candidate list. Candidates are relabelled C1..C10 after each shuffle
    and the ground-truth label is remapped accordingly.

    Saves the result to `save_path` (CSV) so it can be reused across runs.
    If the file already exists, loads and returns it directly.
    """
    if os.path.exists(save_path):
        logger.info(f"Loading existing permuted test set from {save_path}...")
        return pd.read_csv(save_path)

    logger.info(f"Generating permuted test set ({num_permutations} permutations per impression)...")

    rows = []
    for _, row in test_df.iterrows():
        original_candidates = row['candidate'].split('\n')
        original_labels     = row['label'].split(',')

        if len(original_candidates) != 10:
            continue

        for i in range(num_permutations):
            rng = np.random.default_rng(random_state + i)
            permuted, perm  = permute_candidates(original_candidates, rng)
            relabelled       = rebuild_prompt_candidates(permuted)
            new_labels       = map_labels_to_permuted(original_labels, perm)

            rows.append({
                'impression_id':  row['impression_id'],
                'history':        row['history'],
                'candidate':      '\n'.join(relabelled),
                'label':          ','.join(new_labels),
                'permutation_id': i,
                'permutation':    json.dumps(perm.tolist()),
            })

    permuted_df = pd.DataFrame(rows)
    permuted_df.to_csv(save_path, index=False)
    logger.info(f"Permuted test set saved to {save_path} ({len(permuted_df)} rows).")
    return permuted_df

def run_rise_inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, test_df: pd.DataFrame, 
                       N: int, generation_kwargs: dict):
    """
    Iterates over test_df, running the recursive RISE algorithm for each impression.
    Returns ranked lists as plain Python lists (in the permuted C-space of each row).
    """
    recommendations_list = []
    ids_list = []
    times_list = []
    
    for index, row in test_df.iterrows():
        impression_id = row['impression_id']
        history       = row['history']
        candidates    = row['candidate']
        
        current_candidates = candidates.split('\n')
        ranked_list = []

        start_time = time.time()
        
        # Iterative selection loop [cite: 147, 148]
        while len(current_candidates) > 0:
            num_to_pick = min(N, len(current_candidates))
            
            # 1. Generate prompt for this step
            message = generate_RISE_prompt(history, current_candidates, num_to_pick)
            
            # 2. Format with chat template
            formatted_message = tokenizer.apply_chat_template(
                message, 
                padding=True, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 3. Tokenize
            inputs = tokenizer(
                formatted_message, 
                truncation=False, 
                return_tensors="pt",
                padding=True,
            )
            
            # 4. Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs['input_ids'].to('cuda'),
                    attention_mask=inputs['attention_mask'].to('cuda'),
                    **generation_kwargs
                )

            # clean memory 
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
            torch.cuda.empty_cache()
            gc.collect()
            
            # 5. Decode only the newly generated tokens
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            decoded_text = tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=True
            )
            
            # 6. Parse selected item(s) and update lists [cite: 147]
            selected_items = parse_selected_items(decoded_text, current_candidates, num_to_pick)
            
            for item in selected_items:
                # Extract just the ID (e.g., "C1" from "C1: Headline")
                item_id = item.split(':')[0].strip()
                
                if item_id not in ranked_list:
                    ranked_list.append(item_id)
                    
                # We still use the full 'item' string to remove it from the remaining candidates
                if item in current_candidates:
                    current_candidates.remove(item)

            logger.info(f"Impression ID: {impression_id}, Ranking: {ranked_list}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        torch.cuda.empty_cache()
        gc.collect()
        
        # Store as plain list (in permuted C-space)
        recommendations_list.append(ranked_list)
        ids_list.append(impression_id)
        times_list.append(elapsed_time)
        
        logger.info(f"Impression ID: {impression_id}, RISE Final Ranking: {ranked_list}, Time: {elapsed_time} seconds")
        
    return ids_list, recommendations_list, times_list

def parse_ranking_from_text(recommendations: str):
    # (Kept exactly as in your original script)
    recommendation_list = None
    try:
        recommendation_list = json.loads(recommendations)
    except json.JSONDecodeError:
        pass
    
    if recommendation_list is None or 'ranking' not in recommendation_list:
        recommendation_list = {'ranking': [f"C{i}" for i in range(1, 11)]}
        random.shuffle(recommendation_list['ranking'])

    return recommendation_list

def compute_recommendation_metrics(generated_text: pd.DataFrame):
    """
    Compute NDCG@5, NDCG@10, AUC and MRR for a DataFrame of recommendations.

    Parameters
    ----------
    generated_text : DataFrame with columns ['impression_id', 'recommendations', 'label']
                     where 'label' is the ground-truth relevant item in the same
                     permuted C-space as 'recommendations'.

    Returns
    -------
    auc, mrr, ndcg_5, ndcg_10 : lists of per-impression scores
    """
    ndcg_5, ndcg_10, auc, mrr = [], [], [], []

    for _, row in generated_text.iterrows():
        impression_id   = row['impression_id']
        recommendations = row['recommendations']
        relevant_label  = row['label']

        quality = analyze_ranking_quality(recommendations)

        if contains_invalid_h_ids(recommendations):
            group_preds  = np.array([float(10 - i) for i in range(10)], dtype=np.float32)
            group_labels = np.zeros(10, dtype=np.int32)
            group_labels[random.randint(0, 9)] = 1
            logger.info(f"Invalid H IDs for impression_id: {impression_id}")
        else:
            if quality['count'] == 10 and not quality['has_duplicates'] and recommendations:
                group_preds = np.array(
                    [float(len(recommendations) - i) for i in range(len(recommendations))],
                    dtype=np.float32
                )
                try:
                    idx          = recommendations.index(relevant_label)
                    group_labels = [0 if i != idx else 1 for i in range(len(recommendations))]
                except ValueError:
                    group_preds  = np.array([float(10 - i) for i in range(10)], dtype=np.float32)
                    group_labels = np.zeros(10, dtype=np.int32)
                    group_labels[random.randint(0, 9)] = 1
                    logger.info(f"No label found for impression_id: {impression_id}")
            else:
                group_preds  = np.array([float(10 - i) for i in range(10)], dtype=np.float32)
                group_labels = np.zeros(10, dtype=np.int32)
                group_labels[random.randint(0, 9)] = 1
                logger.info(f"No recommendations found for impression_id: {impression_id}")

        res = cal_metric([group_labels], [group_preds], ['group_auc', 'mean_mrr', 'ndcg@5;10'])
        ndcg_5.append(res['ndcg@5'])
        ndcg_10.append(res['ndcg@10'])
        auc.append(res['group_auc'])
        mrr.append(res['mean_mrr'])

    return auc, mrr, ndcg_5, ndcg_10

def main():
    args = parse_arguments()

    logger.info(f"Using model {args.model_path} for RISE analysis...")

    # Load the test dataset from the path provided on the command line
    original_test_df = pd.read_csv(args.test_file, sep=',', header=0)

    # Build (or load) the permuted test set — stored alongside results for reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    permuted_test_file = os.path.join(
        args.output_dir,
        f"MIND_test_permuted_{args.num_permutations}.csv"
    )
    test_df = build_rise_test_set(original_test_df, save_path=permuted_test_file, num_permutations=args.num_permutations)

    generation_kwargs = {
        "max_new_tokens": 800,
        "temperature": 1.0,
        "do_sample": True,
        "top_k": 50,
    }

    logger.info(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side='left', cache_dir=cache_dir)
    llm_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        cache_dir=cache_dir,
        attn_implementation='eager',  # required for some model architectures (e.g. Gemma)
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Running RISE Inference (N={args.N}) over {len(test_df)} rows ({args.num_permutations} permutations × {len(original_test_df)} impressions)...")

    ids, recommendations, generation_times = run_rise_inference(llm_model, tokenizer, test_df, args.N, generation_kwargs)

    # Build results DataFrame — carry label and permutation info for metric computation
    results_df = pd.DataFrame({
        'impression_id':   ids,
        'recommendations': recommendations,
        'label':           test_df['label'].values,
        'permutation_id':  test_df['permutation_id'].values,
        'permutation':     test_df['permutation'].values,
    })

    # Compute per-row recommendation metrics (label is already in permuted C-space)
    auc, mrr, ndcg_5, ndcg_10 = compute_recommendation_metrics(results_df)

    # Ranked lists in permuted C-space (as produced by RISE)
    current_run_ranked_lists = []
    for _, row in results_df.iterrows():
        ranking = parse_ranking_from_text(json.dumps({"ranking": row['recommendations']}))
        current_run_ranked_lists.append(ranking['ranking'])

    # Ranked lists mapped back to canonical (original) C-space
    canonical_ranked_lists = []
    for (_, row), ranked in zip(results_df.iterrows(), current_run_ranked_lists):
        perm = np.array(json.loads(row['permutation']))
        # perm[new_pos] = old_pos  =>  canonical id = old_pos + 1
        canonical = [f"C{perm[int(cid[1:]) - 1] + 1}" for cid in ranked]
        canonical_ranked_lists.append(canonical)

    results = pd.DataFrame({
        'model':                          args.model_path,
        'ids':                            ids,
        'permutation_id':                 test_df['permutation_id'].values,
        'auc':                            auc,
        'mrr':                            mrr,
        'ndcg@5':                         ndcg_5,
        'ndcg@10':                        ndcg_10,
        'recommendation_lists':           current_run_ranked_lists,
        'canonical_recommendation_lists': canonical_ranked_lists,
        'generation_times':               generation_times,
    })
    results.to_csv(os.path.join(args.output_dir, 'RISE_results.csv'), index=False)

    logger.info("RISE evaluation complete.")

if __name__ == "__main__":
    main()
