"""
utils/training_utils.py

Parsing and quality-checking utilities for LLM-generated rankings.

Functions
---------
_coerce_to_text          — normalise LLM output (str / dict / list) to plain text,
                           stripping markdown code fences.
analyze_ranking_quality  — check a candidate-ID list for completeness, duplicates,
                           out-of-range indices, and overall confidence.
contains_invalid_h_ids   — detect rankings that contain history (H#) IDs instead of
                           candidate (C#) IDs, which signals a hallucination.
"""
import re
import json
from typing import List, Dict


def _coerce_to_text(content) -> str:
    """
    Normalise a raw LLM output to a plain string.

    Handles str, dict (OpenAI-style message parts), and list inputs.
    Strips markdown code fences (```json ... ```) if present.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        if 'text' in content and isinstance(content['text'], str):
            return content['text']
        return json.dumps(content, ensure_ascii=False)
    if isinstance(content, list):
        parts = []
        for el in content:
            if isinstance(el, dict) and 'text' in el and isinstance(el['text'], str):
                parts.append(el['text'])
            else:
                parts.append(str(el))
        result = "\n".join(parts)
    else:
        result = str(content)

    # Strip markdown code fences
    result = re.sub(r'^```(?:json)?\s*\n?', '', result, flags=re.MULTILINE)
    result = re.sub(r'\n?```\s*$', '', result, flags=re.MULTILINE)

    return result.strip()


def analyze_ranking_quality(news_ids: List[str], expected_count: int = 10) -> dict:
    """
    Analyse the quality of an extracted candidate ranking.

    Parameters
    ----------
    news_ids       : list of candidate IDs produced by the LLM (e.g. ['C3', 'C1', ...])
    expected_count : expected number of items (default 10 for MIND)

    Returns
    -------
    dict with keys: count, expected_count, has_placeholders, has_duplicates,
                    has_out_of_range, is_sequential, max_id, confidence
    """
    quality_metrics = {
        'count': len(news_ids),
        'expected_count': expected_count,
        'has_placeholders': 'C#' in news_ids,
        'has_duplicates': len(news_ids) != len(set(news_ids)),
        'has_out_of_range': False,
        'is_sequential': False,
        'max_id': 0,
        'confidence': 'low',
    }

    valid_ids = [nid for nid in news_ids if nid != 'C#']
    if valid_ids:
        try:
            numbers = [int(nid[1:]) for nid in valid_ids]
            quality_metrics['max_id'] = max(numbers)
            quality_metrics['has_out_of_range'] = any(n < 1 or n > expected_count for n in numbers)

            if (len(set(numbers)) == len(numbers) == expected_count
                    and max(numbers) <= expected_count):
                quality_metrics['is_sequential'] = True
                quality_metrics['confidence'] = 'high'
        except (ValueError, TypeError):
            pass

    if (quality_metrics['has_placeholders'] or quality_metrics['has_duplicates']
            or quality_metrics['has_out_of_range']):
        quality_metrics['confidence'] = 'low'

    return quality_metrics


def contains_invalid_h_ids(ranking: List[str]) -> bool:
    """
    Return True if the ranking contains any H# identifiers (history IDs).

    The LLM occasionally hallucinates history article IDs (H1, H2, …) instead
    of candidate IDs (C1, C2, …). Such rankings are treated as invalid.

    Parameters
    ----------
    ranking : list of ID strings produced by the LLM

    Returns
    -------
    bool
    """
    if not ranking:
        return False

    h_pattern = re.compile(r'^H\d+$')
    return any(isinstance(nid, str) and h_pattern.match(nid) for nid in ranking)
