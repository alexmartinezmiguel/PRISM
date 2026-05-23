#!/usr/bin/env python3
"""
two_stage_pl_interaction_debiasing.py

Two-stage Plackett-Luce debiasing with a position-relevance interaction model.

Extends the additive model (s_c = q_c + β_{m_c}) with a multiplicative term:

    s_c = (1 + γ_{m_c}) * q_c  +  β_{m_c}

where:
  • β_m  — additive position bias (intercept): a general advantage/disadvantage
            for items placed at prompt position m.
  • γ_m  — relevance-scaling bias (slope): how much position m amplifies (+)
            or attenuates (−) the item's intrinsic relevance score.

Both β and γ are shared across all users and estimated once in Stage 1.
At inference, they are fixed while only the K item-quality scores q_c are
optimized, keeping the per-query cost identical to the additive model.

Stage 1 (offline):  Estimate β and γ from a held-out set of M impressions via
                    Block Coordinate Descent (BCD).

Stage 2 (inference): For each evaluation impression, fix (β̂, γ̂) and estimate
                     item-quality scores q.  Repeat n_repeats times with random
                     subsets of permutations to obtain error bars as a function
                     of the number of samples N.

Usage example
-------------
python two_stage_pl_interaction_debiasing.py \
    --offline_csv    <path/to/offline/recommendation_lists.csv> \
    --eval_csv       <path/to/eval/recommendation_lists.csv>    \
    --test_csv       <path/to/test.csv>                         \
    --n_offline  1000  \
    --n_eval     -1    \
    --R          20    \
    --n_samples  2 3 4 5 6 7 8 9 10 \
    --n_repeats  20    \
    --n_iters_bcd 30   \
    --sigma_q     5.0  \
    --sigma_beta  5.0  \
    --sigma_gamma 5.0  \
    --tol         1e-5 \
    --seed        40   \
    --output_dir  results/pl_interaction/
"""

import os
import ast
import time
import logging
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from scipy.optimize import minimize  # type: ignore
from scipy.special import logsumexp  # type: ignore

import random  # type: ignore
from utils.training_utils import analyze_ranking_quality, contains_invalid_h_ids
from recommenders.models.deeprec.deeprec_utils import cal_metric  # type: ignore
from utils.misc_utils import setup_logging

# ---------------------------------------------------------------------------
# Module-level globals for Stage 2 workers
# ---------------------------------------------------------------------------
_EVAL_DF        = None
_SAMPLE_INDICES = None
_TEST_DF        = None


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Two-stage PL debiasing with position-relevance interaction.'
    )

    p.add_argument('--offline_csv', required=True,
                   help='Full path to the offline recommendation_lists.csv (Stage 1)')
    p.add_argument('--eval_csv', required=True,
                   help='Full path to the eval recommendation_lists.csv (Stage 2)')
    p.add_argument('--test_csv', required=True,
                   help='Full path to the ground-truth test CSV (columns: impression_id, label)')
    p.add_argument('--output_dir', default='results/pl_interaction/',
                   help='Directory where result CSVs and parameter arrays are saved')

    p.add_argument('--n_offline',   type=int,   default=1000,
                   help='Impressions used for offline (β, γ) estimation (Stage 1)')
    p.add_argument('--n_eval',      type=int,   default=-1,
                   help='Impressions used for evaluation (-1 = all)')

    p.add_argument('--R',           type=int,   default=20,
                   help='Permutations per impression used in Stage 1')
    p.add_argument('--n_iters_bcd', type=int,   default=30,
                   help='Maximum BCD outer iterations')
    p.add_argument('--tol',         type=float, default=1e-5,
                   help='Convergence tolerance: stop when max(|Δβ|, |Δγ|) < tol')

    p.add_argument('--n_samples',   type=int,   nargs='+',
                   default=[2, 3, 5, 7, 10, 15, 20],
                   help='Sample sizes to evaluate at inference time')
    p.add_argument('--n_repeats',   type=int,   default=20,
                   help='Random-subset repeats per (impression, n_sample)')

    p.add_argument('--sigma_q',     type=float, default=5.0,
                   help='Prior std for item-quality scores q  (L2 regularisation)')
    p.add_argument('--sigma_beta',  type=float, default=5.0,
                   help='Prior std for additive position bias β (L2 regularisation)')
    p.add_argument('--sigma_gamma', type=float, default=5.0,
                   help='Prior std for relevance-scaling bias γ (L2 regularisation)')

    p.add_argument('--n_workers',   type=int,   default=None,
                   help='Parallel workers (default: all CPU cores)')
    p.add_argument('--seed',        type=int,   default=42,
                   help='Random seed for reproducibility')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading & preprocessing
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


def build_pl_matrices(impression_df: pd.DataFrame,
                      col_prompts: str,
                      col_ranks: str):
    """
    Convert prompt-order and output-rank columns into 2-D integer matrices.

    Returns
    -------
    input_orders : (R, N) int array — prompt position of each item per observation
    output_ranks : (R, N) int array — output rank   of each item per observation
    item2idx     : dict mapping item label to column index
    """
    R          = len(impression_df)
    base_list  = impression_df[col_prompts].iloc[0]
    N          = len(base_list)
    items_sorted = sorted(
        base_list,
        key=lambda x: int(x.replace('C', '')) if x.startswith('C') else x
    )
    item2idx = {item: idx for idx, item in enumerate(items_sorted)}

    input_orders = np.zeros((R, N), dtype=int)
    output_ranks = np.zeros((R, N), dtype=int)

    for r, (_, row) in enumerate(impression_df.iterrows()):
        for pos, item in enumerate(row[col_prompts]):
            input_orders[r, item2idx[item]] = pos
        for pos, item in enumerate(row[col_ranks]):
            output_ranks[r, item2idx[item]] = pos

    return input_orders, output_ranks, item2idx


# ---------------------------------------------------------------------------
# Core PL inner loop (unchanged — operates on pre-computed utilities)
# ---------------------------------------------------------------------------

def _pl_likelihood_and_grad(u_flat, sorted_items_list, N):
    """
    Compute PL log-likelihood and ∂L/∂u for R observations.

    Parameters
    ----------
    u_flat           : (R, N) array of pre-computed utilities
    sorted_items_list: list of R arrays — item indices in output-rank order

    Returns
    -------
    log_lik : scalar
    grad_u  : (R, N) array — ∂log_lik/∂u_{r,i}
    """
    R       = u_flat.shape[0]
    log_lik = 0.0
    grad_u  = np.zeros((R, N))

    for r in range(R):
        u            = u_flat[r]
        sorted_items = sorted_items_list[r]

        for k in range(N):
            item_k    = sorted_items[k]
            remaining = sorted_items[k:]
            u_rem     = u[remaining]
            lse       = logsumexp(u_rem)

            log_lik += u[item_k] - lse

            probs = np.exp(u_rem - lse)
            grad_u[r, item_k]    += 1.0
            grad_u[r, remaining] -= probs

    return log_lik, grad_u


# ---------------------------------------------------------------------------
# Gradient functions — interaction model
# ---------------------------------------------------------------------------

def neg_log_posterior_q(q, input_orders, output_ranks, N, beta_fixed, gamma_fixed, sigma_q):
    """
    Negative log-posterior and gradient w.r.t. q for ONE impression,
    with (β, γ) held fixed.

    Utility:   u_{r,i} = (1 + γ_{p_{r,i}}) * q_i  +  β_{p_{r,i}}
    Chain rule: ∂u_{r,i}/∂q_i = (1 + γ_{p_{r,i}})
    So:         ∂L/∂q_i = Σ_r  ∂L/∂u_{r,i} * (1 + γ_{p_{r,i}})
    """
    scale  = 1.0 + gamma_fixed[input_orders]              # (R, N)
    u_flat = scale * q[np.newaxis, :] + beta_fixed[input_orders]  # (R, N)
    sorted_items_list = [np.argsort(output_ranks[r]) for r in range(input_orders.shape[0])]

    log_lik, grad_u = _pl_likelihood_and_grad(u_flat, sorted_items_list, N)

    # Weight gradient by the position-dependent scale factor
    grad_q = (grad_u * scale).sum(axis=0)

    log_prior    = -0.5 * np.sum(q ** 2) / sigma_q ** 2
    neg_log_post = -(log_lik + log_prior)
    grad         = -grad_q + q / sigma_q ** 2

    return neg_log_post, grad


def neg_log_posterior_beta_gamma(params, input_orders_list, output_ranks_list,
                                  q_list, N, sigma_beta, sigma_gamma):
    """
    Negative log-posterior and gradient w.r.t. [β; γ] (concatenated 2K vector),
    accumulated over ALL M impressions with all q^(m) held fixed.

    Utility:   u_{r,i} = (1 + γ_{p_{r,i}}) * q_i  +  β_{p_{r,i}}
    Chain rule:
      ∂u_{r,i}/∂β_k   = 1         if p_{r,i} = k,  else 0
      ∂u_{r,i}/∂γ_k   = q_i       if p_{r,i} = k,  else 0

    So:
      ∂L/∂β_k = Σ_{u,ρ,i : p_{r,i}=k}  ∂L/∂u_{r,i}
      ∂L/∂γ_k = Σ_{u,ρ,i : p_{r,i}=k}  ∂L/∂u_{r,i} * q_i
    """
    beta  = params[:N]
    gamma = params[N:]

    log_lik    = 0.0
    grad_beta  = np.zeros(N)
    grad_gamma = np.zeros(N)

    for input_orders, output_ranks, q in zip(input_orders_list, output_ranks_list, q_list):
        scale  = 1.0 + gamma[input_orders]                        # (R, N)
        u_flat = scale * q[np.newaxis, :] + beta[input_orders]    # (R, N)
        sorted_items_list = [np.argsort(output_ranks[r]) for r in range(input_orders.shape[0])]

        ll, grad_u = _pl_likelihood_and_grad(u_flat, sorted_items_list, N)
        log_lik   += ll

        # ∂L/∂β_k: scatter grad_u by prompt-position indices (same as additive model)
        np.add.at(grad_beta, input_orders, grad_u)

        # ∂L/∂γ_k: same scatter but weighted by the item's relevance score q_i
        np.add.at(grad_gamma, input_orders, grad_u * q[np.newaxis, :])

    log_prior_beta  = -0.5 * np.sum(beta  ** 2) / sigma_beta  ** 2
    log_prior_gamma = -0.5 * np.sum(gamma ** 2) / sigma_gamma ** 2
    neg_log_post    = -(log_lik + log_prior_beta + log_prior_gamma)

    grad = np.concatenate([
        -grad_beta  + beta  / sigma_beta  ** 2,
        -grad_gamma + gamma / sigma_gamma ** 2,
    ])

    return neg_log_post, grad


# ---------------------------------------------------------------------------
# BCD worker (one q^(m) update, runs in a separate process)
# ---------------------------------------------------------------------------

def _update_q_worker(args):
    """
    Optimize q^(m) for a single impression given fixed (β, γ).
    Returns (m, q_hat).
    """
    m, input_orders, output_ranks, N, beta_fixed, gamma_fixed, sigma_q, q_init = args

    res = minimize(
        neg_log_posterior_q,
        q_init,
        args=(input_orders, output_ranks, N, beta_fixed, gamma_fixed, sigma_q),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
    )
    return m, res.x


# ---------------------------------------------------------------------------
# Stage 1 — Offline (β, γ) estimation via BCD
# ---------------------------------------------------------------------------

def estimate_beta_gamma_offline(
    input_orders_list,
    output_ranks_list,
    N           = 10,
    n_iters     = 30,
    sigma_q     = 5.0,
    sigma_beta  = 5.0,
    sigma_gamma = 5.0,
    tol         = 1e-5,
    n_workers   = None,
):
    """
    Estimate shared position-bias parameters (β, γ) via Block Coordinate Descent.

    Each BCD iteration alternates between:
      (A) Update every q^(m) in parallel, fixing (β, γ)      [M independent K-param problems]
      (B) Update (β, γ) across all impressions, fixing all q  [one 2K-param problem]

    Convergence: max(|Δβ|_∞, |Δγ|_∞) < tol.

    Returns
    -------
    beta_hat  : ndarray (N,) — estimated additive position bias
    gamma_hat : ndarray (N,) — estimated relevance-scaling bias
    q_list    : list of M (N,) arrays — nuisance parameters (can be discarded)
    """
    logger    = logging.getLogger(__name__)
    M         = len(input_orders_list)
    n_workers = n_workers or mp.cpu_count()
    ctx       = mp.get_context('fork')

    # Cold-start
    beta   = np.zeros(N)
    gamma  = np.zeros(N)
    q_list = [np.zeros(N) for _ in range(M)]

    for iteration in range(n_iters):
        t0 = time.time()

        # ── (A) Parallel q updates — fix β and γ ────────────────────────────
        args_list = [
            (m,
             input_orders_list[m],
             output_ranks_list[m],
             N,
             beta.copy(),
             gamma.copy(),
             sigma_q,
             q_list[m].copy())
            for m in range(M)
        ]

        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_update_q_worker, a): a[0] for a in args_list}
            for future in as_completed(futures):
                m, q_new = future.result()
                q_list[m] = q_new

        # ── (B) Joint (β, γ) update — fix all q^(m) ─────────────────────────
        # Concatenate into a single 2K vector for L-BFGS-B.
        params_init = np.concatenate([beta, gamma])

        res_bias = minimize(
            neg_log_posterior_beta_gamma,
            params_init,
            args=(input_orders_list, output_ranks_list, q_list, N, sigma_beta, sigma_gamma),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
        )

        params_new  = res_bias.x
        beta_new    = params_new[:N]
        gamma_new   = params_new[N:]

        delta_beta  = np.max(np.abs(beta_new  - beta))
        delta_gamma = np.max(np.abs(gamma_new - gamma))
        delta       = max(delta_beta, delta_gamma)

        beta  = beta_new
        gamma = gamma_new
        elapsed = time.time() - t0

        logger.info(
            f"BCD iter {iteration+1:3d} | Δmax = {delta:.2e} "
            f"| Δβ = {delta_beta:.2e} | Δγ = {delta_gamma:.2e} "
            f"| β = {np.round(beta, 3)} "
            f"| γ = {np.round(gamma, 3)} "
            f"| {elapsed:.1f}s"
        )

        if delta < tol:
            logger.info(f"BCD converged at iteration {iteration + 1}.")
            break

    return beta, gamma, q_list


# ---------------------------------------------------------------------------
# Stage 2 — Inference-time q estimation with fixed (β̂, γ̂)
# ---------------------------------------------------------------------------

def estimate_q_inference(input_orders, output_ranks, N, beta_hat, gamma_hat, sigma_q):
    """
    Estimate item-quality scores q for one impression with (β̂, γ̂) fixed.

    Returns
    -------
    q_hat            : ndarray (N,)
    debiased_ranking : list of str, e.g. ['C3', 'C1', ...]
    converged        : bool
    """
    res = minimize(
        neg_log_posterior_q,
        np.zeros(N),
        args=(input_orders, output_ranks, N, beta_hat, gamma_hat, sigma_q),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-7}
    )
    q_hat            = res.x
    debiased_ranking = [f'C{i+1}' for i in np.argsort(-q_hat)]
    return q_hat, debiased_ranking, res.success


# ---------------------------------------------------------------------------
# Stage 2 worker (one impression_id, all n_samples and repeats)
# ---------------------------------------------------------------------------

def _worker_stage2(args):
    """
    Process one impression_id for all (n_sample, repeat_idx) combinations.

    Uses module-level globals _EVAL_DF, _SAMPLE_INDICES, _TEST_DF (set by main
    before the ProcessPoolExecutor is created — no pickling via fork).

    Returns a list of result dicts, one per (n_sample, repeat_idx) triple.
    """
    id_, N, beta_hat, gamma_hat, n_samples, n_repeats, sigma_q = args

    imp_df = (
        _EVAL_DF[_EVAL_DF['impression_id'] == id_]
        .sort_values('permutation_seed')
        .reset_index(drop=True)
    )
    io, or_, _ = build_pl_matrices(imp_df, 'prompts_original', 'recommendations_original')

    rows = []
    for n_sample in n_samples:
        if n_sample > len(imp_df):
            continue
        for repeat_idx in range(n_repeats):
            idx = _SAMPLE_INDICES[(id_, n_sample, repeat_idx)]

            t_i = time.time()
            q_hat, debiased_ranking, converged = estimate_q_inference(
                io[idx], or_[idx],
                N         = N,
                beta_hat  = beta_hat,
                gamma_hat = gamma_hat,
                sigma_q   = sigma_q,
            )
            t_f = time.time()

            res_df = pd.DataFrame({
                'impression_id':   [id_],
                'recommendations': [debiased_ranking]
            })
            auc, mrr, ndcg_5, ndcg_10 = compute_recommendation_metrics(res_df, _TEST_DF)

            rows.append({
                'impression_id':    id_,
                'n_samples':        n_sample,
                'repeat':           repeat_idx,
                'converged':        converged,
                'time_s':           t_f - t_i,
                'ndcg_10':          ndcg_10[0],
                'ndcg_5':           ndcg_5[0],
                'auc':              auc[0],
                'mrr':              mrr[0],
                'debiased_ranking': debiased_ranking,
                'q_hat':            q_hat.tolist(),
                'beta_hat':         beta_hat.tolist(),
                'gamma_hat':        gamma_hat.tolist(),
            })
    return rows


# ---------------------------------------------------------------------------
# Compute recommendation metrics (unchanged)
# ---------------------------------------------------------------------------

def compute_recommendation_metrics(generated_text: pd.DataFrame, test_df: pd.DataFrame):
    """
    Compute NDCG@5, NDCG@10, AUC and MRR for a DataFrame of recommendations.
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
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    logger = setup_logging()
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    N   = 10

    logger.info("=" * 60)
    logger.info("Two-Stage PL Debiasing — Interaction Model")
    logger.info("=" * 60)
    logger.info(f"  offline_csv       : {args.offline_csv}")
    logger.info(f"  eval_csv          : {args.eval_csv}")
    logger.info(f"  test_csv          : {args.test_csv}")
    logger.info(f"  n_offline         : {args.n_offline}")
    logger.info(f"  R                 : {args.R}")
    logger.info(f"  n_eval            : {args.n_eval}")
    logger.info(f"  n_samples         : {args.n_samples}")
    logger.info(f"  n_repeats         : {args.n_repeats}")
    logger.info(f"  n_iters_bcd       : {args.n_iters_bcd}")
    logger.info(f"  sigma_q           : {args.sigma_q}")
    logger.info(f"  sigma_beta        : {args.sigma_beta}")
    logger.info(f"  sigma_gamma       : {args.sigma_gamma}")
    logger.info(f"  tol               : {args.tol}")
    logger.info(f"  n_workers         : {args.n_workers or mp.cpu_count()}")
    logger.info(f"  seed              : {args.seed}")
    logger.info("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("[1/4] Loading data ...")
    offline_df = load_data(args.offline_csv)
    eval_df    = load_data(args.eval_csv)
    test_df    = pd.read_csv(args.test_csv, sep=',', header=0)

    offline_ids = offline_df['impression_id'].unique()[:args.n_offline]
    eval_ids    = eval_df['impression_id'].unique()
    eval_ids    = eval_ids if args.n_eval == -1 else eval_ids[:args.n_eval]

    logger.info(f"  Offline impressions available : {offline_df['impression_id'].nunique()}")
    logger.info(f"  Offline impressions used      : {len(offline_ids)}")
    logger.info(f"  Eval impressions available    : {eval_df['impression_id'].nunique()}")
    logger.info(f"  Eval impressions used         : {len(eval_ids)}")

    # ── Stage 1: load or estimate (β, γ) ─────────────────────────────────────
    # Both arrays are stored in a single .npz so they always stay in sync.
    params_fname = (
        f"bias_params_interaction"
        f"_noffline{len(offline_ids)}"
        f"_R{args.R}"
        f"_iters{args.n_iters_bcd}"
        f"_tol{args.tol}"
        f".npz"
    )
    params_path = os.path.join(args.output_dir, params_fname)

    if os.path.exists(params_path):
        loaded    = np.load(params_path)
        beta_hat  = loaded['beta']
        gamma_hat = loaded['gamma']
        logger.info(f"[2/4] Loaded existing (β̂, γ̂) from {params_path}")
        logger.info(f"  β̂ = {np.round(beta_hat,  4)}")
        logger.info(f"  γ̂ = {np.round(gamma_hat, 4)}")
    else:
        logger.info("[2/4] Building PL matrices for offline impressions ...")
        input_orders_list = []
        output_ranks_list = []

        for id_ in offline_ids:
            imp_df = (
                offline_df[offline_df['impression_id'] == id_]
                .sort_values('permutation_seed')
                .reset_index(drop=True)
                .iloc[:args.R]
            )
            if len(imp_df) < 2:
                continue
            io, or_, _ = build_pl_matrices(imp_df, 'prompts_original', 'recommendations_original')
            input_orders_list.append(io)
            output_ranks_list.append(or_)

        logger.info(f"  Using {len(input_orders_list)} offline impressions after filtering.")
        logger.info(f"  Running BCD (max {args.n_iters_bcd} iters, tol={args.tol}) ...")

        t_stage1 = time.time()
        beta_hat, gamma_hat, _ = estimate_beta_gamma_offline(
            input_orders_list,
            output_ranks_list,
            N           = N,
            n_iters     = args.n_iters_bcd,
            sigma_q     = args.sigma_q,
            sigma_beta  = args.sigma_beta,
            sigma_gamma = args.sigma_gamma,
            tol         = args.tol,
            n_workers   = args.n_workers,
        )
        logger.info(f"  Stage 1 done in {time.time() - t_stage1:.1f}s")
        logger.info(f"  β̂ = {np.round(beta_hat,  4)}")
        logger.info(f"  γ̂ = {np.round(gamma_hat, 4)}")

        np.savez(params_path, beta=beta_hat, gamma=gamma_hat)
        logger.info(f"  (β̂, γ̂) saved to {params_path}")

    # ── Pre-generate or reload shared random indices ──────────────────────────
    indices_fname = (
        f"sample_indices"
        f"_neval{len(eval_ids)}"
        f"_nsamp{'_'.join(str(s) for s in args.n_samples)}"
        f"_nrep{args.n_repeats}"
        f"_seed{args.seed}"
        f"_noffline{len(offline_ids)}"
        f"_R{args.R}"
        f".npy"
    )
    indices_path = os.path.join(args.output_dir, indices_fname)

    if os.path.exists(indices_path):
        sample_indices = np.load(indices_path, allow_pickle=True).item()
        logger.info(f"[3/4] Loaded existing sample_indices from {indices_path}")
    else:
        logger.info(f"[3/4] Generating sample_indices (seed={args.seed}) ...")
        sample_indices = {}
        for id_ in eval_ids:
            imp_df  = eval_df[eval_df['impression_id'] == id_]
            n_avail = len(imp_df)
            for n_sample in args.n_samples:
                if n_sample > n_avail:
                    continue
                for repeat_idx in range(args.n_repeats):
                    sample_indices[(id_, n_sample, repeat_idx)] = rng.choice(
                        n_avail, size=n_sample, replace=False
                    )
        np.save(indices_path, sample_indices)
        logger.info(f"  sample_indices saved to {indices_path}")

    # ── Stage 2: inference-time q estimation ─────────────────────────────────
    logger.info("[4/4] Stage 2 — inference-time q estimation ...")
    n_workers = args.n_workers or mp.cpu_count()
    results   = []
    t_stage2  = time.time()

    global _EVAL_DF, _SAMPLE_INDICES, _TEST_DF
    _EVAL_DF        = eval_df
    _SAMPLE_INDICES = sample_indices
    _TEST_DF        = test_df

    stage2_args = [
        (id_, N, beta_hat, gamma_hat, args.n_samples, args.n_repeats, args.sigma_q)
        for id_ in eval_ids
    ]
    ctx = mp.get_context('fork')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_worker_stage2, a): a[0] for a in stage2_args}
        for future in as_completed(futures):
            results.extend(future.result())

    logger.info(f"  Stage 2 done in {time.time() - t_stage2:.1f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    results_fname = (
        f"stage2_results_interaction"
        f"_noffline{len(offline_ids)}"
        f"_R{args.R}"
        f"_neval{len(eval_ids)}"
        f"_nsamp{'_'.join(str(s) for s in args.n_samples)}"
        f"_nrep{args.n_repeats}"
        f"_seed{args.seed}"
        f".csv"
    )
    results_path = os.path.join(args.output_dir, results_fname)
    results_df   = pd.DataFrame(results)
    results_df.to_csv(results_path, index=False)

    logger.info(f"  Results saved to   : {results_path}")
    logger.info(f"  Rows               : {len(results_df)}")
    logger.info(f"  Converged          : {results_df['converged'].mean()*100:.1f}%")
    logger.info(f"  Mean NDCG@10       : {results_df['ndcg_10'].mean():.4f}")


if __name__ == '__main__':
    main()
