#!/usr/bin/env python3
"""
stella_debiasing.py

STELLA-style post-processing baseline for MIND.

This implementation follows the two-stage structure from:
  "Large Language Models are Not Stable Recommender Systems" (STELLA)

Stage 1 (probing):
  Build a transition matrix T where T[i, j] estimates:
    P(predicted_top_item_prompt_position = j | ground_truth_prompt_position = i)

Stage 2 (recommendation):
  For each impression, iterate over sampled shuffled outputs, run Bayesian updates
  p <- normalize(p * T[:, y]), track entropy, and aggregate the full rankings of
  the lowest-entropy steps (Borda) to produce the final debiased ranking.
"""

import os
import ast
import time
import argparse
import logging
import random
import multiprocessing as mp
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

from utils.training_utils import analyze_ranking_quality, contains_invalid_h_ids
from recommenders.models.deeprec.deeprec_utils import cal_metric  # type: ignore
from utils.misc_utils import setup_logging

# ---------------------------------------------------------------------------
# Module-level globals for Stage-2 parallel workers
# ---------------------------------------------------------------------------
_EVAL_DF = None
_SAMPLE_INDICES = None
_TEST_DF = None
_T = None
_ARGS = None


def parse_args():
    p = argparse.ArgumentParser(description="STELLA baseline for MIND.")

    p.add_argument("--probing_csv", required=True,
                   help="Full path to the probing recommendation_lists.csv (controlled permutations).")
    p.add_argument("--eval_csv", required=True,
                   help="Full path to the evaluation recommendation_lists.csv (random permutations).")
    p.add_argument("--test_csv", required=True,
                   help="Full path to the ground-truth test CSV (columns: impression_id, label).")
    p.add_argument("--output_dir", required=True,
                   help="Directory where transition matrix and STELLA outputs are saved.")

    p.add_argument("--n_offline", type=int, default=1000,
                   help="Number of impressions used to build transition matrix.")
    p.add_argument("--n_eval", type=int, default=-1,
                   help="Number of evaluation impressions (-1 = all).")

    p.add_argument("--k_candidates", type=int, default=10,
                   help="Number of candidates per impression.")
    p.add_argument("--tm_smoothing", type=float, default=1.0,
                   help="Laplace smoothing added to transition-matrix counts.")

    p.add_argument("--n_samples", type=int, nargs="+",
                   default=[2, 3, 4, 5, 6, 7, 8, 9, 10],
                   help="Number of Bayesian update steps to use.")
    p.add_argument("--n_repeats", type=int, default=20,
                   help="Random repeats per (impression, n_sample).")
    p.add_argument("--max_steps", type=int, default=10,
                   help="Maximum update steps (cap within selected sample indices).")

    p.add_argument("--entropy_patience", type=int, default=3,
                   help="Early-stop when entropy change stays below tolerance for this many steps.")
    p.add_argument("--entropy_tol", type=float, default=1e-6,
                   help="Entropy convergence tolerance.")
    p.add_argument("--aggregate_top_m", type=int, default=3,
                   help="Aggregate the M lowest-entropy rankings via Borda.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")
    p.add_argument("--n_workers", type=int, default=None,
                   help="Number of worker processes for Stage-2 (default: all CPU cores).")
    return p.parse_args()


def load_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    for col in ["recommendations", "prompts_original", "recommendations_original"]:
        if col in df.columns:
            df[col] = df[col].apply(ast.literal_eval)
    if "permutations" in df.columns:
        df["permutations"] = df["permutations"].apply(
            lambda x: np.fromstring(x.strip("[]"), sep=" ", dtype=int).tolist()
            if isinstance(x, str) else x
        )
    return df


def aggregate_borda(rankings):
    items = list({item for ranking in rankings for item in ranking})
    position_counts = {item: Counter() for item in items}
    for ranking in rankings:
        for pos, item in enumerate(ranking):
            position_counts[item][pos] += 1

    scores = {
        item: sum(pos * count for pos, count in position_counts[item].items()) / max(1, len(rankings))
        for item in items
    }
    return sorted(items, key=lambda x: scores[x])


def compute_recommendation_metrics(generated_text: pd.DataFrame, test_df: pd.DataFrame, k: int = 10):
    ndcg_5, ndcg_10, auc, mrr = [], [], [], []
    for _, row in generated_text.iterrows():
        impression_id = row["impression_id"]
        recommendations = row["recommendations"]
        relevant_label = test_df.loc[test_df["impression_id"] == impression_id, "label"].values[0]
        quality = analyze_ranking_quality(recommendations)

        if contains_invalid_h_ids(recommendations):
            group_preds = np.array([float(k - i) for i in range(k)], dtype=np.float32)
            group_labels = np.zeros(k, dtype=np.int32)
            group_labels[random.randint(0, k - 1)] = 1
        else:
            if quality["count"] == k and not quality["has_duplicates"] and recommendations:
                group_preds = np.array([float(len(recommendations) - i) for i in range(len(recommendations))], dtype=np.float32)
                try:
                    idx = recommendations.index(relevant_label)
                    group_labels = [0 if i != idx else 1 for i in range(len(recommendations))]
                except ValueError:
                    group_preds = np.array([float(k - i) for i in range(k)], dtype=np.float32)
                    group_labels = np.zeros(k, dtype=np.int32)
                    group_labels[random.randint(0, k - 1)] = 1
            else:
                group_preds = np.array([float(k - i) for i in range(k)], dtype=np.float32)
                group_labels = np.zeros(k, dtype=np.int32)
                group_labels[random.randint(0, k - 1)] = 1

        res = cal_metric([group_labels], [group_preds], ["group_auc", "mean_mrr", "ndcg@5;10"])
        ndcg_5.append(res["ndcg@5"])
        ndcg_10.append(res["ndcg@10"])
        auc.append(res["group_auc"])
        mrr.append(res["mean_mrr"])
    return auc, mrr, ndcg_5, ndcg_10


def build_transition_matrix(probing_df: pd.DataFrame, k: int = 10, smoothing: float = 1.0):
    counts = np.full((k, k), smoothing, dtype=np.float64)
    used = 0

    for _, row in probing_df.iterrows():
        if "labels" not in row or not isinstance(row["labels"], str) or not row["labels"].startswith("C"):
            continue
        rest = row["labels"][1:]
        if not rest.isdigit():
            continue
        gt_pos = int(rest) - 1
        if gt_pos < 0 or gt_pos >= k:
            continue

        ranking = row["recommendations_original"] if "recommendations_original" in probing_df.columns else row["recommendations"]
        if not ranking:
            continue
        pred_top = ranking[0]
        prompt_order = row["prompts_original"]
        if pred_top not in prompt_order:
            continue
        pred_pos = prompt_order.index(pred_top)
        if pred_pos < 0 or pred_pos >= k:
            continue

        counts[gt_pos, pred_pos] += 1.0
        used += 1

    T = counts / counts.sum(axis=1, keepdims=True)
    return T, counts, used


def save_transition_matrix(T: np.ndarray, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "transition_matrix.npy"), T)
    pd.DataFrame(
        T,
        index=[f"gt_pos_{i+1}" for i in range(T.shape[0])],
        columns=[f"pred_pos_{j+1}" for j in range(T.shape[1])]
    ).to_csv(os.path.join(output_dir, "transition_matrix.csv"))

    plt.figure(figsize=(8, 6))
    sns.heatmap(T, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=max(0.3, float(np.max(T))))
    plt.xlabel("Predicted top-item prompt position")
    plt.ylabel("Ground-truth prompt position")
    plt.title("STELLA Transition Matrix T")
    plt.xticks([x + 0.5 for x in range(T.shape[1])], [str(i + 1) for i in range(T.shape[1])])
    plt.yticks([x + 0.5 for x in range(T.shape[0])], [str(i + 1) for i in range(T.shape[0])], rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "transition_matrix_heatmap.pdf"))
    plt.close()


def run_bayesian_steps(rows_df: pd.DataFrame, T: np.ndarray, step_indices, max_steps: int,
                       entropy_tol: float, entropy_patience: int):
    k = T.shape[0]
    p = np.ones(k, dtype=np.float64) / k
    entropies = []
    posterior_trace = []
    step_rankings = []
    step_positions = []

    patience = 0
    prev_entropy = None
    capped_indices = step_indices[:max_steps]

    for idx in capped_indices:
        row = rows_df.iloc[idx]
        ranking = row["recommendations_original"]
        prompt_order = row["prompts_original"]
        if not ranking or ranking[0] not in prompt_order:
            continue

        top_item = ranking[0]
        y = prompt_order.index(top_item)

        likelihood = T[:, y]
        p = p * likelihood
        p_sum = p.sum()
        if p_sum <= 0:
            p = np.ones(k, dtype=np.float64) / k
        else:
            p = p / p_sum

        entropy = float(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum())
        entropies.append(entropy)
        posterior_trace.append(p.copy().tolist())
        step_rankings.append(ranking)
        step_positions.append(int(y))

        if prev_entropy is not None and abs(prev_entropy - entropy) < entropy_tol:
            patience += 1
        else:
            patience = 0
        prev_entropy = entropy

        if patience >= entropy_patience:
            break

    return entropies, posterior_trace, step_rankings, step_positions


def _worker_stage2(imp_id):
    """Process one impression_id across all n_samples and repeats."""
    imp_df = _EVAL_DF[_EVAL_DF["impression_id"] == imp_id].sort_values("permutation_seed").reset_index(drop=True)
    rows = []
    recommendation_rows = []

    for n_sample in _ARGS.n_samples:
        if (imp_id, n_sample, 0) not in _SAMPLE_INDICES:
            continue
        for repeat_idx in range(_ARGS.n_repeats):
            key = (imp_id, n_sample, repeat_idx)
            if key not in _SAMPLE_INDICES:
                continue
            t_i = time.time()

            entropies, posterior_trace, step_rankings, step_positions = run_bayesian_steps(
                imp_df,
                _T,
                _SAMPLE_INDICES[key],
                max_steps=_ARGS.max_steps,
                entropy_tol=_ARGS.entropy_tol,
                entropy_patience=_ARGS.entropy_patience,
            )
            if not step_rankings:
                continue

            m = min(_ARGS.aggregate_top_m, len(step_rankings))
            best_idx = np.argsort(entropies)[:m]
            selected_rankings = [step_rankings[i] for i in best_idx]
            stella_ranking = aggregate_borda(selected_rankings)

            tmp_df = pd.DataFrame({"impression_id": [imp_id], "recommendations": [stella_ranking]})
            auc, mrr, ndcg_5, ndcg_10 = compute_recommendation_metrics(tmp_df, _TEST_DF, k=_ARGS.k_candidates)
            t_f = time.time()

            rows.append({
                "impression_id": imp_id,
                "n_samples": n_sample,
                "repeat": repeat_idx,
                "time_s": float(t_f - t_i),
                "steps_used": len(step_rankings),
                "selected_steps": best_idx.tolist(),
                "selected_pred_positions": [step_positions[i] for i in best_idx],
                "min_entropy": float(np.min(entropies)),
                "final_entropy": float(entropies[-1]),
                "entropy_trace": entropies,
                "posterior_trace": posterior_trace,
                "stella_ranking": stella_ranking,
                "ndcg_10": ndcg_10[0],
                "ndcg_5": ndcg_5[0],
                "auc": auc[0],
                "mrr": mrr[0],
            })
            recommendation_rows.append({
                "impression_id": imp_id,
                "n_samples": n_sample,
                "repeat": repeat_idx,
                "recommendations": stella_ranking,
            })

    return rows, recommendation_rows


def main():
    args = parse_args()
    logger = setup_logging()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("=" * 60)
    logger.info("STELLA Debiasing (MIND)")
    logger.info("=" * 60)
    logger.info(f"probing_csv={args.probing_csv}")
    logger.info(f"eval_csv={args.eval_csv}")
    logger.info(f"test_csv={args.test_csv}")
    logger.info(f"output_dir={args.output_dir}")

    probing_df = load_data(args.probing_csv)
    eval_df    = load_data(args.eval_csv)
    test_df    = pd.read_csv(args.test_csv, sep=",", header=0)

    probing_ids = probing_df["impression_id"].unique()[:args.n_offline]
    probing_subset = probing_df[probing_df["impression_id"].isin(probing_ids)].copy()

    T, counts, used = build_transition_matrix(
        probing_subset,
        k=args.k_candidates,
        smoothing=args.tm_smoothing
    )
    save_transition_matrix(T, args.output_dir)
    np.save(os.path.join(args.output_dir, "transition_counts.npy"), counts)
    logger.info(f"Transition matrix built from {used} valid probing rows.")

    eval_ids = eval_df["impression_id"].unique()
    if args.n_eval != -1:
        eval_ids = eval_ids[:args.n_eval]

    sample_indices = {}
    for imp_id in eval_ids:
        n_avail = len(eval_df[eval_df["impression_id"] == imp_id])
        for n_sample in args.n_samples:
            m = min(n_sample, args.max_steps)
            if m > n_avail:
                continue
            for repeat_idx in range(args.n_repeats):
                sample_indices[(imp_id, n_sample, repeat_idx)] = rng.choice(n_avail, size=m, replace=False)
    np.save(os.path.join(args.output_dir, "sample_indices_stella.npy"), sample_indices)

    start = time.time()
    rows = []
    recommendation_rows = []
    n_workers = args.n_workers or mp.cpu_count()
    logger.info(f"Stage-2 parallel workers: {n_workers}")

    global _EVAL_DF, _SAMPLE_INDICES, _TEST_DF, _T, _ARGS
    _EVAL_DF = eval_df
    _SAMPLE_INDICES = sample_indices
    _TEST_DF = test_df
    _T = T
    _ARGS = args

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_worker_stage2, int(imp_id)): imp_id for imp_id in eval_ids}
        for future in as_completed(futures):
            worker_rows, worker_recs = future.result()
            rows.extend(worker_rows)
            recommendation_rows.extend(worker_recs)

    results_df = pd.DataFrame(rows)
    recs_df = pd.DataFrame(recommendation_rows)
    results_path = os.path.join(args.output_dir, "stella_metrics.csv")
    recs_path = os.path.join(args.output_dir, "stella_recommendation_lists.csv")
    results_df.to_csv(results_path, index=False)
    recs_df.to_csv(recs_path, index=False)

    summary = {
        "n_rows": int(len(results_df)),
        "n_eval_impressions": int(len(eval_ids)),
        "mean_ndcg_10": float(results_df["ndcg_10"].mean()) if len(results_df) else float("nan"),
        "std_ndcg_10": float(results_df["ndcg_10"].std()) if len(results_df) else float("nan"),
        "mean_time_s": float(results_df["time_s"].mean()) if len(results_df) else float("nan"),
        "std_time_s": float(results_df["time_s"].std()) if len(results_df) else float("nan"),
        "elapsed_s": float(time.time() - start),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.output_dir, "stella_summary.csv"), index=False)
    logger.info(f"Saved STELLA outputs to {args.output_dir}")
    logger.info(f"Summary: {summary}")


if __name__ == "__main__":
    main()

