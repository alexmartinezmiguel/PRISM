#!/usr/bin/env python3
"""
majority_voting_debiasing.py

Majority voting (Borda count / Greedy) debiasing for LLM-based news recommendations.

For each evaluation impression and each (n_sample, repeat_idx) combination,
aggregates n_sample LLM rankings using majority voting and computes recommendation
quality metrics. Uses the same sample_indices file as two_stage_pl_debiasing.py
to ensure a fair comparison.

Usage example
-------------
python majority_voting_debiasing.py \
    --eval_csv     <path/to/eval/recommendation_lists.csv> \
    --test_csv     <path/to/test.csv> \
    --indices_path results/pl_interaction/sample_indices_*.npy \
    --method       borda \
    --n_eval       -1 \
    --n_samples    2 3 5 7 10 15 20 \
    --n_repeats    20 \
    --n_workers    8 \
    --output_dir   results/majority_voting/
"""

import os
import ast
import time
import logging
import argparse
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np #type: ignore
import pandas as pd #type: ignore
import random

from utils.training_utils import analyze_ranking_quality, contains_invalid_h_ids
from recommenders.models.deeprec.deeprec_utils import cal_metric  # type: ignore
from utils.misc_utils import setup_logging

# ---------------------------------------------------------------------------
# Module-level globals for workers
# Set by main() before ProcessPoolExecutor is created so that forked child
# processes can access them without pickling (copy-on-write via fork).
# ---------------------------------------------------------------------------
_EVAL_DF        = None   # evaluation recommendations DataFrame
_SAMPLE_INDICES = None   # pre-generated random subset indices (shared with PL script)
_TEST_DF        = None   # ground-truth labels (MIND_test.csv)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Majority voting debiasing for LLM rankings.')

    # Data
    p.add_argument('--eval_csv', required=True,
                   help='Full path to the evaluation recommendation_lists.csv')
    p.add_argument('--test_csv', required=True,
                   help='Full path to the ground-truth test CSV (columns: impression_id, label)')
    p.add_argument('--indices_path', required=True,
                   help='Path to sample_indices .npy file (shared with PL script for fair comparison)')
    p.add_argument('--output_dir', default='results/majority_voting/',
                   help='Directory where result CSVs are saved')

    # Method
    p.add_argument('--method', type=str, default='borda', choices=['borda', 'greedy'],
                   help='Aggregation method: borda (average position) or greedy (positional majority vote)')

    # Evaluation config — should match the values used to generate sample_indices
    p.add_argument('--n_eval',    type=int, default=-1,
                   help='Number of evaluation impressions (-1 = all)')
    p.add_argument('--n_samples', type=int, nargs='+', default=[2, 3, 5, 7, 10, 15, 20],
                   help='List of sample sizes (must match those in sample_indices file)')
    p.add_argument('--n_repeats', type=int, default=20,
                   help='Number of random-subset repeats per (impression, n_sample)')

    # Compute
    p.add_argument('--n_workers', type=int, default=None,
                   help='Number of parallel workers (default: all CPU cores)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_path: str) -> pd.DataFrame:
    """Load and parse recommendation_lists.csv into a DataFrame."""
    df = pd.read_csv(data_path)
    df['recommendations']          = df['recommendations'].apply(ast.literal_eval)
    df['prompts_original']         = df['prompts_original'].apply(ast.literal_eval)
    df['recommendations_original'] = df['recommendations_original'].apply(ast.literal_eval)
    df['permutations'] = df['permutations'].apply(
        lambda x: np.fromstring(x.strip('[]'), sep=' ', dtype=int).tolist()
        if isinstance(x, str) else x
    )
    return df


# ---------------------------------------------------------------------------
# Majority voting aggregation
# ---------------------------------------------------------------------------

def aggregate_rankings(rankings, method='borda'):
    """
    Aggregate a list of rankings into a single consensus ranking.

    Parameters
    ----------
    rankings : list of lists
        Each inner list is a ranked ordering of items (best → worst).
    method : str
        'borda'  — Borda count: rank by average output position (lower = better)
        'greedy' — greedy positional majority vote

    Returns
    -------
    list : aggregated ranking, best item first.
    """
    items = list({item for ranking in rankings for item in ranking})
    n     = len(rankings[0])

    # Count how many times each item appeared at each position
    position_counts = {item: Counter() for item in items}
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            position_counts[item][pos] += 1

    if method == 'borda':
        # Lower average position = better
        borda_scores = {
            item: sum(pos * count for pos, count in position_counts[item].items()) / len(rankings)
            for item in items
        }
        return sorted(items, key=lambda x: borda_scores[x])

    elif method == 'greedy':
        # At each slot, assign the item with the most votes at that position
        final_ranking = []
        assigned      = set()
        for pos in range(n):
            best_item = max(
                (item for item in items if item not in assigned),
                key=lambda item: position_counts[item][pos]
            )
            final_ranking.append(best_item)
            assigned.add(best_item)
        return final_ranking

    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'borda' or 'greedy'.")


# ---------------------------------------------------------------------------
# Recommendation metrics
# ---------------------------------------------------------------------------

def compute_recommendation_metrics(generated_text: pd.DataFrame, test_df: pd.DataFrame):
    """
    Compute NDCG@5, NDCG@10, AUC and MRR for a DataFrame of recommendations.

    Parameters
    ----------
    generated_text : DataFrame with columns ['impression_id', 'recommendations']
    test_df        : ground-truth DataFrame with columns ['impression_id', 'label']

    Returns
    -------
    auc, mrr, ndcg_5, ndcg_10 : lists of per-impression scores
    """
    ndcg_5, ndcg_10, auc, mrr = [], [], [], []

    for _, row in generated_text.iterrows():
        impression_id   = row['impression_id']
        recommendations = row['recommendations']
        relevant_label  = test_df.loc[
            test_df['impression_id'] == impression_id, 'label'
        ].values[0]

        quality = analyze_ranking_quality(recommendations)

        if contains_invalid_h_ids(recommendations):
            group_preds  = np.array([float(10 - i) for i in range(10)], dtype=np.float32)
            group_labels = np.zeros(10, dtype=np.int32)
            group_labels[random.randint(0, 9)] = 1
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
            else:
                group_preds  = np.array([float(10 - i) for i in range(10)], dtype=np.float32)
                group_labels = np.zeros(10, dtype=np.int32)
                group_labels[random.randint(0, 9)] = 1

        res = cal_metric([group_labels], [group_preds], ['group_auc', 'mean_mrr', 'ndcg@5;10'])
        ndcg_5.append(res['ndcg@5'])
        ndcg_10.append(res['ndcg@10'])
        auc.append(res['group_auc'])
        mrr.append(res['mean_mrr'])

    return auc, mrr, ndcg_5, ndcg_10


# ---------------------------------------------------------------------------
# Worker (one impression_id, all n_samples and repeats)
# ---------------------------------------------------------------------------

def _worker_mv(args):
    """
    Process one impression_id for all (n_sample, repeat_idx) combinations.

    Uses module-level globals _EVAL_DF, _SAMPLE_INDICES, _TEST_DF which are
    set by main() before the ProcessPoolExecutor is created. With fork, child
    processes inherit these read-only objects via copy-on-write — no pickling
    and no race conditions.

    Returns a list of result dicts, one per (n_sample, repeat_idx) triple.
    """
    id_, n_samples, n_repeats, method = args

    imp_df = (
        _EVAL_DF[_EVAL_DF['impression_id'] == id_]
        .sort_values('permutation_seed')
        .reset_index(drop=True)
    )

    rows = []
    for n_sample in n_samples:
        if n_sample > len(imp_df):
            continue
        for repeat_idx in range(n_repeats):
            # Use the pre-generated shared indices — same as PL script for fair comparison
            idx      = _SAMPLE_INDICES[(id_, n_sample, repeat_idx)]
            rankings = imp_df.iloc[idx]['recommendations_original'].tolist()

            t_i = time.time()
            aggregated_ranking = aggregate_rankings(rankings, method=method)
            t_f = time.time()

            res_df = pd.DataFrame({
                'impression_id':   [id_],
                'recommendations': [aggregated_ranking]
            })
            auc, mrr, ndcg_5, ndcg_10 = compute_recommendation_metrics(res_df, _TEST_DF)

            rows.append({
                'impression_id':      id_,
                'n_samples':          n_sample,
                'repeat':             repeat_idx,
                'time_s':             t_f - t_i,
                'ndcg_10':            ndcg_10[0],
                'ndcg_5':             ndcg_5[0],
                'auc':                auc[0],
                'mrr':                mrr[0],
                'aggregated_ranking': aggregated_ranking,
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    logger = setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Majority Voting Debiasing")
    logger.info("=" * 60)
    logger.info(f"  eval_csv       : {args.eval_csv}")
    logger.info(f"  test_csv       : {args.test_csv}")
    logger.info(f"  indices_path   : {args.indices_path}")
    logger.info(f"  method         : {args.method}")
    logger.info(f"  n_eval         : {args.n_eval}")
    logger.info(f"  n_samples      : {args.n_samples}")
    logger.info(f"  n_repeats      : {args.n_repeats}")
    logger.info(f"  n_workers      : {args.n_workers or mp.cpu_count()}")
    logger.info("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("[1/3] Loading data ...")
    eval_df = load_data(args.eval_csv)
    test_df = pd.read_csv(args.test_csv, sep=',', header=0)

    eval_ids = eval_df['impression_id'].unique()
    eval_ids = eval_ids if args.n_eval == -1 else eval_ids[:args.n_eval]

    logger.info(f"  Eval impressions available : {eval_df['impression_id'].nunique()}")
    logger.info(f"  Eval impressions used      : {len(eval_ids)}")

    # ── Load shared sample_indices ────────────────────────────────────────────
    # Loading the same indices file used by the PL script guarantees that both
    # methods operate on identical permutation subsets for every
    # (impression_id, n_sample, repeat_idx) — the only fair basis for comparison.
    logger.info(f"[2/3] Loading sample_indices from {args.indices_path} ...")
    if not os.path.exists(args.indices_path):
        raise FileNotFoundError(
            f"sample_indices file not found at {args.indices_path}.\n"
            f"Run two_stage_pl_debiasing.py first to generate it."
        )
    sample_indices = np.load(args.indices_path, allow_pickle=True).item()
    logger.info(f"  Loaded {len(sample_indices)} index entries.")

    # ── Expose shared data as module-level globals before forking ─────────────
    global _EVAL_DF, _SAMPLE_INDICES, _TEST_DF
    _EVAL_DF        = eval_df
    _SAMPLE_INDICES = sample_indices
    _TEST_DF        = test_df

    # ── Run majority voting in parallel over impression_ids ───────────────────
    logger.info(f"[3/3] Running majority voting ({args.method}) ...")
    n_workers   = args.n_workers or mp.cpu_count()
    results     = []
    t_start     = time.time()

    worker_args = [
        (id_, args.n_samples, args.n_repeats, args.method)
        for id_ in eval_ids
    ]
    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_worker_mv, a): a[0] for a in worker_args}
        for future in as_completed(futures):
            results.extend(future.result())

    logger.info(f"  Done in {time.time() - t_start:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    # Filename encodes method and evaluation config for traceability.
    results_fname = (
        f"mv_{args.method}_results"
        f"_neval{len(eval_ids)}"
        f"_nsamp{'_'.join(str(s) for s in args.n_samples)}"
        f"_nrep{args.n_repeats}"
        f".csv"
    )
    results_path = os.path.join(args.output_dir, results_fname)
    results_df   = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    logger.info(f"  Results saved to : {results_path}")
    logger.info(f"  Rows             : {len(results_df)}")


if __name__ == '__main__':
    main()