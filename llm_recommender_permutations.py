#!/usr/bin/env python3
"""
llm_recommender_self_consistency.py

Self-consistency data collection for LLM-based news recommendation.

Generates multiple LLM rankings per impression under different candidate
permutations, producing the offline dataset used as input to the debiasing
methods (STELLA, PL-interaction, majority voting).

Pipeline:
  1. Load the training CSV and select a fixed validation slice.
  2. For the selected checkpoint, build prompts with either random or
     controlled candidate permutations via `build_chat_template_dataset_PL`.
  3. Run batched LLM inference; parse JSON rankings; remap to original C-space.
  4. Save `recommendation_lists.csv` to the specified --output-dir.

Usage
-----
    python llm_recommender_permutations.py \\
        --model-path   <path/to/checkpoint or HF model ID> \\
        --train-file   <path/to/train.csv> \\
        --output-dir   <path/to/output/> \\
        --num-permutations 100 \\
        --type random \\
        --batch-size 4

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
import argparse

# Third-party
import numpy as np # type: ignore
import torch # type: ignore
import pandas as pd # type: ignore
import datasets # type: ignore
import tensorflow as tf # type: ignore
from datasets import Dataset # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM # type: ignore

# Workspace root is expected to be /workspace (Docker) or the repo root locally.
os.chdir('/workspace')

# Internal utilities
from utils.data_utils import build_chat_template_dataset_PL
from utils.misc_utils import setup_logging
from utils.training_utils import _coerce_to_text, analyze_ranking_quality, contains_invalid_h_ids
from recommenders.models.deeprec.deeprec_utils import cal_metric # type: ignore

logger = setup_logging()
datasets.disable_progress_bar()

# ---------------------------------------------------------------------------
# GPU and cache configuration
# ---------------------------------------------------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model-path', type=str, required=True,
                        help='Full path to the model checkpoint directory (or a Hugging Face model ID)')
    parser.add_argument('--train-file', type=str, required=True,
                        help='Path to the training CSV; a fixed slice is used as the validation set')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory where recommendation_lists.csv is written')
    parser.add_argument('--num-permutations', type=int, default=100,
                        help='Number of permutations to generate per impression')
    parser.add_argument('--type', type=str, default='random', choices=['random', 'controlled'],
                        help='Permutation strategy: random or controlled')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for inference')
    return parser.parse_args()

def run_inference(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dataset, ids: list, 
                  batch_size: int, generation_kwargs: dict):
    """
    Run batched LLM inference over a prompt dataset.

    Parameters
    ----------
    model           : loaded causal LM
    tokenizer       : associated tokenizer (must have pad_token set)
    dataset         : Dataset of chat-template messages (list of message dicts per row)
    ids             : impression IDs aligned with `dataset` rows
    batch_size      : number of prompts per forward pass
    generation_kwargs : dict passed directly to model.generate()

    Returns
    -------
    list of str — raw decoded text for each prompt (generated tokens only)
    """
    recommendations_list = []
    for i in range(0, len(dataset), batch_size):
        batch_end = min(i + batch_size, len(dataset))
        batch_ids = ids[i:batch_end]
        batch_messages = dataset[i:batch_end]
        
        # convert messages chat template
        batch_formatted_messages = tokenizer.apply_chat_template(
            batch_messages, 
            padding=True, 
            tokenize=False, 
            add_generation_prompt=True)

        # tokenize batch
        tokenized_batch = tokenizer(
            batch_formatted_messages, 
            truncation=False, 
            return_tensors="pt",
            padding=True,
        )

        # Generate responses
        with torch.no_grad():
            batch_outputs = model.generate(
                tokenized_batch['input_ids'].to('cuda'),
                attention_mask=tokenized_batch['attention_mask'].to('cuda'),
                **generation_kwargs
                )
        # clean memory 
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:50'
        torch.cuda.empty_cache()
        gc.collect()

        # Decode the generated outputs
        # Get the length of input prompts to remove them from outputs (since generate includes input)
        input_lengths = tokenized_batch['input_ids'].shape[1]

        for i, output in enumerate(batch_outputs):
            # Remove the input tokens from the output (keep only newly generated tokens)
            generated_tokens = output[input_lengths:]
            
            # Decode the generated tokens to text
            decoded_text = tokenizer.decode(
                generated_tokens, 
                skip_special_tokens=True,  # Remove special tokens like <eos>, <pad>, etc.
                clean_up_tokenization_spaces=True  # Clean up extra spaces
            )
            recommendations_list.append(decoded_text)
            logger.info(f"Impression ID: {batch_ids[i]}, Generation: {decoded_text}")
        
    return recommendations_list

def parse_ranking_from_text(recommendations: str):
    """
    Parse a JSON ranking string produced by the LLM into a Python dict.

    Attempts strict JSON parsing first; falls back to bracket-balancing and
    then a regex search for the 'ranking' array. Returns a random fallback
    ranking if all parsing attempts fail.

    Returns
    -------
    dict with key 'ranking' mapping to a list of C-space identifiers.
    """
    try:
        recommendation_list = json.loads(_coerce_to_text(recommendations))
    except json.JSONDecodeError:
        text = _coerce_to_text(recommendations)
        fixed_text = text.rstrip()
        open_braces = fixed_text.count('{') - fixed_text.count('}')
        open_brackets = fixed_text.count('[') - fixed_text.count(']')
        fixed_text += ']' * open_brackets + '}' * open_braces
        try:
            recommendation_list = json.loads(fixed_text)
        except json.JSONDecodeError:
            match = re.search(r'"ranking"\s*:\s*\[(.*?)\]', text, re.DOTALL)
            if match:
                try:
                    ranking_str = '[' + match.group(1) + ']'
                    ranking = json.loads(ranking_str)
                    recommendation_list = {'ranking': ranking}
                except:
                    pass

    if recommendation_list is None or 'ranking' not in recommendation_list:
        recommendation_list = {'ranking': [f"C{i}" for i in range(1, 11)]}
        random.shuffle(recommendation_list['ranking'])

    return recommendation_list

def _remap_ranking_to_original(ranked_list, perm):
    """Map C-ids from permuted space back to original C-space using the perm array."""
    try:
        return [f'C{perm[int(c[1:]) - 1] + 1}' for c in ranked_list]
    except (ValueError, IndexError):
        return None  # signals a bad row

def main():
    """Main inference function."""
    args = parse_arguments()
    
    # Check GPU availability
    gpu_devices = tf.config.list_physical_devices('GPU')
    if not gpu_devices or 'GPU' not in str(gpu_devices[0]):
        logger.error('GPU not detected')
    logger.info(f"Using model {args.model_path}...")


    # Load training data and use a fixed validation slice (rows 20 000–21 000)
    logger.info(f"Loading dataset from {args.train_file}...")
    global test_df
    train_df = pd.read_csv(args.train_file, sep=',', header=0)
    test_df = train_df.iloc[20000:21000]  # held-out validation slice
    os.makedirs(args.output_dir, exist_ok=True)

    parsed_recommendations = []

    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "temperature": 1.0,
        "do_sample": True,
        "max_new_tokens": 800,
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

    if args.type == 'random':
        # Build all num_permutations in a single call; results are row-major
        logger.info(f"Building {args.num_permutations} random permutations for the full test set...")
        ids, dataset, permuted_labels, permutations, permuted_prompts = build_chat_template_dataset_PL(
            test_df, permutation_type='random', num_permutations=args.num_permutations, random_state=42
        )
        # Permutation seed: items are row-major (row0_perm0, row0_perm1, …)
        perm_seeds = [(i % args.num_permutations) + 1 for i in range(len(ids))]

        recommendations_list = run_inference(llm_model, tokenizer, dataset, ids, args.batch_size, generation_kwargs)

        for generation in recommendations_list:
            ranking = parse_ranking_from_text(generation)
            parsed_recommendations.append(ranking['ranking'])

        generated_text = pd.DataFrame({
            'model':            [args.model_path] * len(ids),
            'impression_id':    ids,
            'permutation_seed': perm_seeds,
            'recommendations':  parsed_recommendations,
            'labels':           permuted_labels,
            'permutations':     permutations,
            'prompts_original': permuted_prompts,
        })

        generated_text['recommendations_original'] = generated_text.apply(
            lambda row: _remap_ranking_to_original(row['recommendations'], row['permutations']),
            axis=1
        )
        generated_text = generated_text.dropna(subset=['recommendations_original']).reset_index(drop=True)

    elif args.type == 'controlled':
        iteration_frames = []

        for p in range(args.num_permutations):
            logger.info(f"Running controlled inference for position {p + 1}/{args.num_permutations}...")
            ids, dataset, permuted_labels, permutations, permuted_prompts = build_chat_template_dataset_PL(
                test_df, permutation_type='controlled', num_permutations=1, random_state=p + 1
            )
            recommendations_list = run_inference(llm_model, tokenizer, dataset, ids, args.batch_size, generation_kwargs)

            parsed_recommendations = [
                parse_ranking_from_text(generation)['ranking']
                for generation in recommendations_list
            ]

            iteration_df = pd.DataFrame({
                'model':            [args.model_path] * len(ids),
                'impression_id':    ids,
                'permutation_seed': [p + 1] * len(ids),
                'recommendations':  parsed_recommendations,
                'labels':           permuted_labels,
                'permutations':     permutations,
                'prompts_original': permuted_prompts,
            })

            iteration_df['recommendations_original'] = iteration_df.apply(
                lambda row: _remap_ranking_to_original(row['recommendations'], row['permutations']),
                axis=1
            )
            iteration_df = iteration_df.dropna(subset=['recommendations_original']).reset_index(drop=True)

            iteration_frames.append(iteration_df)

        generated_text = pd.concat(iteration_frames, ignore_index=True)

    generated_text.to_csv(os.path.join(args.output_dir, 'recommendation_lists.csv'), index=False)
    logger.info(f"Saved {len(generated_text)} rows to {args.output_dir}/recommendation_lists.csv")


if __name__ == "__main__":
    main()