"""
utils/data_utils.py

Prompt-building and permutation utilities for LLM-based news recommendation.

Functions
---------
permute_candidates            — randomly shuffle a candidate list and return the
                                permutation array.
rebuild_prompt_candidates     — re-label permuted candidates as C1, C2, …, Ck.
map_labels_to_permuted        — map ground-truth C-ids to their new positions after
                                a permutation.
build_chat_template_dataset_PL — build a full chat-template dataset with random,
                                controlled, or cyclic candidate permutations;
                                used by llm_recommender_permutations.py.
"""
import re
import random
import numpy as np  # type: ignore
import pandas as pd  # type: ignore


def permute_candidates(candidates, rng=None):
    """
    Randomly permute a candidate list.

    Parameters
    ----------
    candidates : list[str]
        Original candidate strings (length k), each formatted as "C#: headline".
    rng : np.random.Generator or None
        Random generator; a fresh default_rng is created if None.

    Returns
    -------
    permuted_candidates : list[str]
        Reordered candidates (same strings, new order).
    perm : np.ndarray, shape (k,)
        Permutation array where perm[new_index] = old_index.
    """
    if rng is None:
        rng = np.random.default_rng()

    k = len(candidates)
    perm = rng.permutation(k)
    permuted_candidates = [candidates[i] for i in perm]
    return permuted_candidates, perm


def rebuild_prompt_candidates(permuted_candidates):
    """
    Re-label a permuted candidate list so identifiers run C1, C2, …, Ck.

    Parameters
    ----------
    permuted_candidates : list[str]
        Candidates after permutation (may have stale C# prefixes).

    Returns
    -------
    list[str] — re-labelled candidates, e.g. ["C1: headline_A", "C2: headline_B", …]
    """
    rebuilt = []
    for i, cand in enumerate(permuted_candidates):
        title = cand.split(":", 1)[1].strip()
        rebuilt.append(f"C{i + 1}: {title}")
    return rebuilt


def map_labels_to_permuted(labels, perm):
    """
    Translate ground-truth C-ids from original space to permuted prompt space.

    Parameters
    ----------
    labels : list[str]
        Ground-truth identifiers in original space (e.g. ['C5']).
    perm : np.ndarray, shape (k,)
        Permutation where perm[new_index] = old_index (0-based).

    Returns
    -------
    list[str] — identifiers in the permuted space (e.g. ['C3']).
    """
    k = len(perm)
    inverse_perm = np.empty_like(perm)
    inverse_perm[perm] = np.arange(k)

    mapped = []
    for cid in labels:
        old_pos = int(cid[1:]) - 1       # 'C5' → 4
        new_pos = inverse_perm[old_pos]   # 0-based new position
        mapped.append(f"C{new_pos + 1}")
    return mapped


def build_chat_template_dataset_PL(
    df: pd.DataFrame,
    permutation_type: str = 'random',
    num_permutations: int = 100,
    random_state: int = 42,
):
    """
    Build a chat-template dataset with permuted candidate orderings.

    For each impression with exactly 10 candidates, generates `num_permutations`
    variants with shuffled candidate orderings to expose the model to different
    prompt positions (used for self-consistency data collection).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: impression_id, candidate_news_id, candidate,
        label, history.
    permutation_type : str
        'random'     — fully random independent shuffles.
        'controlled' — pin the first relevant candidate at position (random_state-1);
                       the remaining candidates are shuffled randomly.
        'cyclic'     — rotate the candidate list by evenly spaced offsets.
    num_permutations : int
        Number of permuted variants per impression.
    random_state : int
        Base seed; permutation i uses seed (random_state + i).

    Returns
    -------
    ids, all_messages, new_labels, permutations, permuted_prompts
        Five parallel lists (one entry per impression × permutation).
    """
    # Prompt v8 — JSON-only output format
    system_prompt = (
        "You serve as a personalized news recommendation system. Understand the "
        "User's History News, and then generate a recommendation list from the "
        "Candidate News ranked by how well they match the user's interests "
        "(i.e., similarity/continuity with the history)."
        "Output Format: Output your answer strictly in valid JSON with this exact "
        "structure and field order:\n\n"
        "{\n"
        '  "ranking": ["C#", "C#", ..., "C#"]\n'
        "}\n\n"
        "Important:\n"
        "- The JSON must be syntactically valid (no markdown, no extra text, no commentary).\n"
        "- The ranking must be ordered from most relevant to least relevant.\n"
        "- Do not include anything outside the JSON block."
    )

    ids, all_messages, new_labels, permutations, permuted_prompts = [], [], [], [], []

    # For controlled mode the target position is fixed across all impressions/iterations
    controlled_target_pos = max(0, min(9, random_state - 1))

    for _, row in df.iterrows():
        original_candidates = row['candidate'].split('\n')
        original_labels     = row['label'].split(',')
        n_candidates        = len(row['candidate_news_id'].split('\n'))

        if n_candidates != 10:
            continue

        for i in range(num_permutations):
            rng = np.random.default_rng(random_state + i)

            if permutation_type == 'random':
                permuted, perm = permute_candidates(original_candidates, rng)

            elif permutation_type == 'controlled':
                labels_list = [lbl for lbl in original_labels if lbl]

                if not labels_list:
                    permuted, perm = permute_candidates(original_candidates, rng)
                else:
                    try:
                        primary_idx = int(labels_list[0][1:]) - 1
                        assert 0 <= primary_idx < n_candidates
                    except (ValueError, AssertionError):
                        permuted, perm = permute_candidates(original_candidates, rng)
                    else:
                        # Place the primary relevant item at the target position;
                        # shuffle the rest randomly.
                        remaining = [j for j in range(n_candidates) if j != primary_idx]
                        rng.shuffle(remaining)
                        new_order = remaining
                        new_order.insert(controlled_target_pos, primary_idx)
                        perm    = np.array(new_order, dtype=np.int64)
                        permuted = [original_candidates[j] for j in perm]

            elif permutation_type == 'cyclic':
                shift = int(round(i * (n_candidates / num_permutations)))
                base_indices = list(range(n_candidates))
                if shift == 0:
                    new_order = base_indices
                else:
                    new_order = base_indices[-shift:] + base_indices[:-shift]
                perm    = np.array(new_order, dtype=np.int64)
                permuted = [original_candidates[j] for j in perm]

            else:
                raise ValueError(f"Unknown permutation_type: {permutation_type!r}")

            prompt_candidates = '\n'.join(rebuild_prompt_candidates(permuted))
            permuted_labels   = ','.join(map_labels_to_permuted(original_labels, perm))
            permuted_prompt   = [f'C{c + 1}' for c in perm]

            user_prompt = (
                f"User's History News: {row['history']}\n"
                f"Candidate News: {prompt_candidates}\n"
                f"Important: Imperative to include all {n_candidates} ids of the Candidate News set.\n"
            )

            # Single user turn (compatible with Gemma, Llama, Qwen chat templates)
            messages = [{"role": "user", "content": system_prompt + "\n\n" + user_prompt}]

            ids.append(row['impression_id'])
            all_messages.append(messages)
            new_labels.append(permuted_labels)
            permutations.append(perm)
            permuted_prompts.append(permuted_prompt)

    return ids, all_messages, new_labels, permutations, permuted_prompts
