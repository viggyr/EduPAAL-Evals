"""Metric functions: every comparison between system probes and ground truth.

Conventions:
- levels are the strings "unknown" | "beginner" | "intermediate" | "advanced";
  ORDINAL maps them to 0..3 (imported from systems.base).
- every function is pure and unit-tested on synthetic inputs in tests/.
- missing predictions are never treated as zeros: callers pass only the
  (prediction, truth) pairs that were actually produced; a crashed leg
  aborts the run loudly instead of contributing a row.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Optional, Sequence, Tuple

from systems.base import LEVELS, ORDINAL


def mastery_exact_match(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    """Fraction of (predicted, truth) pairs with equal levels."""
    if not pairs:
        return None
    return sum(1 for p, t in pairs if p == t) / len(pairs)


def ordinal_mae(pairs: Sequence[Tuple[str, str]]) -> Optional[float]:
    """Mean |ordinal(pred) - ordinal(truth)|; unknown=0 .. advanced=3."""
    if not pairs:
        return None
    return sum(abs(ORDINAL[p] - ORDINAL[t]) for p, t in pairs) / len(pairs)


def monotonicity(topic_levels: Dict[str, List[str]]) -> Optional[float]:
    """Fraction of topics whose probed levels never decrease across probes.

    All scenario evidence is non-negative, so a coherent learner model should
    not regress. EduPAAL has no demotion by design; baselines may regress
    when retrieval or the judge wobbles.
    """
    if not topic_levels:
        return None
    ok = 0
    for levels in topic_levels.values():
        ords = [ORDINAL[l] for l in levels]
        if all(b >= a for a, b in zip(ords, ords[1:])):
            ok += 1
    return ok / len(topic_levels)


def cross_vertical_agreement(topic_levels: Dict[str, List[str]]) -> Optional[float]:
    """Fraction of topics (with >= 2 probes) whose probed levels stay within
    one ordinal step of each other — i.e. different verticals' evidence does
    not make the system contradict itself about the same concept."""
    eligible = {t: ls for t, ls in topic_levels.items() if len(ls) >= 2}
    if not eligible:
        return None
    ok = 0
    for levels in eligible.values():
        ords = [ORDINAL[l] for l in levels]
        if max(ords) - min(ords) <= 1:
            ok += 1
    return ok / len(eligible)


def grounding_precision_recall(
    pred: Dict[str, str], truth: Dict[str, str]
) -> Tuple[Optional[float], Optional[float]]:
    """precision: predicted non-unknown topics that are truly non-unknown.
    recall: truly non-unknown topics the system also reports as non-unknown.
    Catches hallucinated progress (low precision) and amnesia (low recall)."""
    pred_pos = {t for t, l in pred.items() if l != "unknown"}
    truth_pos = {t for t, l in truth.items() if l != "unknown"}
    precision = len(pred_pos & truth_pos) / len(pred_pos) if pred_pos else None
    recall = len(pred_pos & truth_pos) / len(truth_pos) if truth_pos else None
    return precision, recall


def grounding_accuracy(pred: Dict[str, str], truth: Dict[str, str]) -> Optional[float]:
    """Fraction of topics where the grounding report matches ground truth."""
    if not truth:
        return None
    return sum(1 for t in truth if pred.get(t) == truth[t]) / len(truth)


def next_topic_exact_match(
    pairs: Sequence[Tuple[Optional[str], Optional[str]]]
) -> Optional[float]:
    if not pairs:
        return None
    return sum(1 for p, t in pairs if p == t) / len(pairs)


def prereq_violation(
    returned: Optional[str],
    truth_levels: Dict[str, str],
    prereqs: Dict[str, List[str]],
) -> bool:
    """True when the returned next topic has a prerequisite that is not
    ground-truth advanced (or when a topic is returned but none should be)."""
    if returned is None:
        return False
    return any(truth_levels.get(p) != "advanced" for p in prereqs.get(returned, []))


def prereq_violation_rate(
    probes: Sequence[Tuple[Optional[str], Dict[str, str], Dict[str, List[str]]]]
) -> Optional[float]:
    if not probes:
        return None
    return sum(1 for r, lv, pq in probes if prereq_violation(r, lv, pq)) / len(probes)


def weakest_hit_rate(
    pairs: Sequence[Tuple[List[str], List[str]]]
) -> Optional[float]:
    """Mean |predicted ∩ truth| / n over weakest(n) probes. n is the truth
    list's length, so a short prediction is penalized, not excused."""
    if not pairs:
        return None
    scores = []
    for pred, truth in pairs:
        n = len(truth)
        scores.append(len(set(pred) & set(truth)) / n if n else 1.0)
    return sum(scores) / len(scores)


def provenance_coverage(
    claims: Sequence[Tuple[List[str], List[str]]],
) -> Tuple[Optional[float], Optional[float]]:
    """claims: (cited_ids, actual_evidence_ids) for each non-unknown claim.
    coverage: fraction of claims citing at least one real evidence id.
    citation precision: cited ids that are real, over all cited ids."""
    if not claims:
        return None, None
    coverage = sum(1 for cited, actual in claims if set(cited) & set(actual)) / len(claims)
    total_cited = sum(len(cited) for cited, _ in claims)
    valid_cited = sum(len(set(cited) & set(actual)) for cited, actual in claims)
    precision = valid_cited / total_cited if total_cited else None
    return coverage, precision


def latency_stats(latencies: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """p50/p95 seconds per op plus sample count. Pure summary of observed data."""
    out: Dict[str, Dict[str, float]] = {}
    for op, samples in latencies.items():
        if not samples:
            continue
        qs = statistics.quantiles(sorted(samples), n=100, method="inclusive")
        out[op] = {
            "count": float(len(samples)),
            "p50": qs[49],
            "p95": qs[94],
            "max": max(samples),
        }
    return out
