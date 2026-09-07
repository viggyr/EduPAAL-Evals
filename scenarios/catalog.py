"""Scenario catalog: S1..S4 form the default suite. Each builder is
deterministic for (id, seed). S5's builder is retained and still
addressable via get_scenario("S5", seed), but it is excluded from the
default suite: learner isolation is enforced structurally by
learner-scoped records and was already covered by the multi-learner
scale test in the EduPAAL validation battery."""

from __future__ import annotations

from typing import Dict

from .common import Learner, ProbePoint, ScenarioSpec, StreamBuilder, Topic


def _s1(seed: int) -> ScenarioSpec:
    t1, t2, t3, t4 = (
        Topic("t1", "Fractions"),
        Topic("t2", "Decimals"),
        Topic("t3", "Percentages"),
        Topic("t4", "Ratios"),
    )
    L = Learner("s1-learner", "Steady Learner")
    b = StreamBuilder("s1", seed)
    A = b.add
    # day 0
    A(L.id, t1, "guide", 0.55, 0, attempts=3, hints_used=1)
    A(L.id, t1, "guide", 0.62, 0, attempts=2, hints_used=1)
    # day 3
    A(L.id, t1, "evaluator", 0.70, 3, attempts=2, hints_used=0)
    # day 6
    A(L.id, t2, "guide", 0.50, 6, attempts=3, hints_used=2)
    A(L.id, t1, "practice", 0.78, 6, attempts=2, hints_used=0)
    # day 9
    A(L.id, t1, "evaluator", 0.85, 9, attempts=1, hints_used=0)
    A(L.id, t2, "evaluator", 0.58, 9, attempts=2, hints_used=1)
    # day 12
    A(L.id, t1, "practice", 0.92, 12, attempts=1, hints_used=0)
    A(L.id, t2, "guide", 0.66, 12, attempts=2, hints_used=0)
    # day 15
    A(L.id, t3, "guide", 0.55, 15, attempts=3, hints_used=1)
    A(L.id, t2, "practice", 0.72, 15, attempts=2, hints_used=0)
    # day 18
    A(L.id, t3, "guide", 0.60, 18, attempts=2, hints_used=1)
    return ScenarioSpec(
        id="S1",
        title="Steady learner: 4-topic chain, 8 sessions, 3 verticals",
        seed=seed,
        topics=[t1, t2, t3, t4],
        learners=[L],
        plan_order=["t1", "t2", "t3", "t4"],
        prereqs={},
        evidence=b.evidence,
        probes=[ProbePoint("mid", 7), ProbePoint("final", 12)],
    )


def _s2(seed: int) -> ScenarioSpec:
    a1, a2 = Topic("a1", "Linear Equations"), Topic("a2", "Quadratic Equations")
    L = Learner("s2-learner", "Struggling Learner")
    b = StreamBuilder("s2", seed)
    A = b.add
    # Strong dialogue, weak quizzes on a1: characterizes disagreement handling.
    A(L.id, a1, "guide", 0.90, 0, attempts=1, hints_used=0)
    A(L.id, a1, "guide", 0.85, 0, attempts=1, hints_used=0)
    A(L.id, a1, "evaluator", 0.35, 4, attempts=3, hints_used=2)
    A(L.id, a1, "evaluator", 0.40, 4, attempts=3, hints_used=1)
    A(L.id, a2, "evaluator", 0.30, 8, attempts=3, hints_used=2)
    A(L.id, a2, "evaluator", 0.35, 8, attempts=3, hints_used=2)
    A(L.id, a1, "guide", 0.88, 12, attempts=1, hints_used=0)
    A(L.id, a2, "evaluator", 0.32, 12, attempts=3, hints_used=2)
    return ScenarioSpec(
        id="S2",
        title="Struggling learner: conflicting dialogue vs quiz evidence",
        seed=seed,
        topics=[a1, a2],
        learners=[L],
        plan_order=["a1", "a2"],
        prereqs={},
        evidence=b.evidence,
        probes=[ProbePoint("mid", 6), ProbePoint("final", 8)],
    )


def _s3(seed: int) -> ScenarioSpec:
    pa, pb, pc = (
        Topic("pa", "Variables"),
        Topic("pb", "Expressions"),
        Topic("pc", "Equations"),
    )
    L = Learner("s3-learner", "Prereq Learner")
    b = StreamBuilder("s3", seed)
    A = b.add
    A(L.id, pa, "guide", 0.75, 0, attempts=2, hints_used=0)
    A(L.id, pa, "evaluator", 0.80, 0, attempts=2, hints_used=0)
    A(L.id, pa, "practice", 0.85, 4, attempts=1, hints_used=0)
    A(L.id, pa, "guide", 0.82, 4, attempts=1, hints_used=0)
    A(L.id, pa, "evaluator", 0.88, 8, attempts=1, hints_used=0)
    A(L.id, pa, "practice", 0.90, 8, attempts=1, hints_used=0)
    A(L.id, pb, "guide", 0.60, 12, attempts=2, hints_used=1)
    A(L.id, pb, "guide", 0.65, 12, attempts=2, hints_used=1)
    return ScenarioSpec(
        id="S3",
        title="Prerequisite chain pa -> pb -> pc; next_topic must respect prereqs",
        seed=seed,
        topics=[pa, pb, pc],
        learners=[L],
        plan_order=["pa", "pb", "pc"],
        prereqs={"pb": ["pa"], "pc": ["pb"]},
        evidence=b.evidence,
        probes=[ProbePoint("mid", 6), ProbePoint("final", 8)],
    )


def _s4(seed: int) -> ScenarioSpec:
    p1 = Topic("p1", "Probability")
    L = Learner("s4-learner", "Plateau Learner")
    b = StreamBuilder("s4", seed)
    A = b.add
    # Performance stalls in the high-0.6s/low-0.7s: must NOT reach advanced.
    A(L.id, p1, "guide", 0.68, 0, attempts=2, hints_used=1)
    A(L.id, p1, "evaluator", 0.70, 0, attempts=2, hints_used=1)
    A(L.id, p1, "guide", 0.72, 5, attempts=2, hints_used=0)
    A(L.id, p1, "evaluator", 0.69, 5, attempts=2, hints_used=1)
    A(L.id, p1, "practice", 0.71, 10, attempts=2, hints_used=0)
    A(L.id, p1, "guide", 0.70, 10, attempts=2, hints_used=0)
    return ScenarioSpec(
        id="S4",
        title="Plateau: evidence stalls at intermediate; advanced is a false positive",
        seed=seed,
        topics=[p1],
        learners=[L],
        plan_order=["p1"],
        prereqs={},
        evidence=b.evidence,
        probes=[ProbePoint("mid", 4), ProbePoint("final", 6)],
    )


def _s5(seed: int) -> ScenarioSpec:
    x1, x2 = Topic("x1", "Photosynthesis"), Topic("x2", "Cell Division")
    LA = Learner("s5-learner-a", "Learner A")
    LB = Learner("s5-learner-b", "Learner B")
    b = StreamBuilder("s5", seed)
    A = b.add
    # Interleaved ingestion; A masters x1 while B barely starts x2.
    A(LA.id, x1, "guide", 0.80, 0, attempts=2, hints_used=0)
    A(LB.id, x2, "evaluator", 0.35, 0, attempts=3, hints_used=2)
    A(LA.id, x1, "evaluator", 0.85, 3, attempts=1, hints_used=0)
    A(LB.id, x2, "evaluator", 0.40, 3, attempts=3, hints_used=2)
    A(LA.id, x1, "practice", 0.88, 6, attempts=1, hints_used=0)
    A(LA.id, x1, "guide", 0.82, 6, attempts=1, hints_used=0)
    A(LA.id, x1, "evaluator", 0.90, 9, attempts=1, hints_used=0)
    A(LA.id, x1, "practice", 0.92, 9, attempts=1, hints_used=0)
    return ScenarioSpec(
        id="S5",
        title="Isolation: two learners interleaved; no cross-learner leakage",
        seed=seed,
        topics=[x1, x2],
        learners=[LA, LB],
        plan_order=["x1", "x2"],
        prereqs={},
        evidence=b.evidence,
        probes=[ProbePoint("mid", 4), ProbePoint("final", 8)],
    )


_BUILDERS: Dict[str, object] = {
    "S1": _s1,
    "S2": _s2,
    "S3": _s3,
    "S4": _s4,
    "S5": _s5,
}

ALL_SCENARIOS = ["S1", "S2", "S3", "S4"]
# S5 is intentionally not in the default suite (see module docstring);
# get_scenario("S5", seed) still works for explicit use.


def get_scenario(scenario_id: str, seed: int) -> ScenarioSpec:
    sid = scenario_id.upper()
    if sid not in _BUILDERS:
        raise ValueError(f"unknown scenario {scenario_id!r}; choose from {ALL_SCENARIOS}")
    return _BUILDERS[sid](seed)
