"""Metric functions for EduPAAL-Evals."""

from .metrics import (
    mastery_exact_match,
    ordinal_mae,
    monotonicity,
    cross_vertical_agreement,
    grounding_precision_recall,
    grounding_accuracy,
    next_topic_exact_match,
    prereq_violation,
    prereq_violation_rate,
    weakest_hit_rate,
    provenance_coverage,
    latency_stats,
)

__all__ = [
    "mastery_exact_match",
    "ordinal_mae",
    "monotonicity",
    "cross_vertical_agreement",
    "grounding_precision_recall",
    "grounding_accuracy",
    "next_topic_exact_match",
    "prereq_violation",
    "prereq_violation_rate",
    "weakest_hit_rate",
    "provenance_coverage",
    "latency_stats",
]
