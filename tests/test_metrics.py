"""Metric correctness on hand-computed synthetic inputs."""

from metrics import (
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


def test_mastery_exact_match():
    pairs = [("advanced", "advanced"), ("beginner", "intermediate"), ("unknown", "unknown")]
    assert mastery_exact_match(pairs) == 2 / 3
    assert mastery_exact_match([]) is None


def test_ordinal_mae():
    # unknown=0 beginner=1 intermediate=2 advanced=3
    pairs = [("advanced", "advanced"), ("beginner", "advanced"), ("unknown", "beginner")]
    assert ordinal_mae(pairs) == (0 + 2 + 1) / 3
    assert ordinal_mae([]) is None


def test_monotonicity():
    assert monotonicity({"t1": ["unknown", "beginner", "intermediate"]}) == 1.0
    assert monotonicity({"t1": ["beginner", "unknown"]}) == 0.0
    assert monotonicity({"t1": ["intermediate"], "t2": ["advanced", "beginner"]}) == 0.5
    assert monotonicity({}) is None


def test_cross_vertical_agreement():
    ok = {"t1": ["beginner", "intermediate"], "t2": ["intermediate", "intermediate"]}
    assert cross_vertical_agreement(ok) == 1.0
    bad = {"t1": ["beginner", "advanced"]}
    assert cross_vertical_agreement(bad) == 0.0
    assert cross_vertical_agreement({"t1": ["beginner"]}) is None  # single probe: ineligible


def test_grounding_precision_recall():
    pred = {"a": "beginner", "b": "unknown", "c": "intermediate"}
    truth = {"a": "beginner", "b": "beginner", "c": "unknown"}
    p, r = grounding_precision_recall(pred, truth)
    assert p == 0.5  # {a} of {a, c}
    assert r == 0.5  # {a} of {a, b}
    assert grounding_accuracy(pred, truth) == 1 / 3


def test_next_topic():
    assert next_topic_exact_match([("a", "a"), ("b", "c"), (None, None)]) == 2 / 3
    assert prereq_violation("pc", {"pa": "advanced", "pb": "beginner"}, {"pc": ["pb"]}) is True
    assert prereq_violation("pb", {"pa": "advanced"}, {"pb": ["pa"]}) is False
    assert prereq_violation(None, {}, {}) is False
    probes = [
        ("pb", {"pa": "advanced"}, {"pb": ["pa"]}),
        ("pc", {"pa": "advanced", "pb": "beginner"}, {"pc": ["pb"]}),
    ]
    assert prereq_violation_rate(probes) == 0.5


def test_weakest_hit_rate():
    pairs = [(["x", "y"], ["y", "x"]), (["x", "z"], ["x", "y"])]
    assert weakest_hit_rate(pairs) == (1.0 + 0.5) / 2
    assert weakest_hit_rate([]) is None


def test_provenance_coverage():
    claims = [(["ev-1"], ["ev-1", "ev-2"]), ([], ["ev-3"]), (["ev-9"], ["ev-3"])]
    coverage, precision = provenance_coverage(claims)
    assert coverage == 1 / 3
    assert precision == 1 / 2  # ev-1 valid of {ev-1, ev-9}
    assert provenance_coverage([]) == (None, None)


def test_latency_stats():
    stats = latency_stats({"ingest": [0.1, 0.2, 0.3, 0.4]})
    assert stats["ingest"]["count"] == 4
    assert abs(stats["ingest"]["p50"] - 0.25) < 1e-9
    assert stats["ingest"]["max"] == 0.4
    assert latency_stats({}) == {}
