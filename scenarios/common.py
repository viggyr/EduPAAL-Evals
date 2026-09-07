"""Shared scenario machinery: spec types, seeded stream builder, ground truth.

Ground truth is a *stipulated, fixed rubric* — a measurement instrument, not a
pedagogical claim. It is deliberately close to, but not identical with,
EduPAAL's shipped heuristic defaults (K=3, T_intermediate=0.65, T_advanced=0.8,
cross-modal required for advanced). The differences are intentional: the eval
measures each system's agreement with an *independent* notion of mastery, so a
system that merely re-implements EduPAAL's exact thresholds would score 100%
by construction and the comparison would be circular.

RUBRIC (see scenarios/README.md for the full rationale):
  unknown      no evidence
  beginner     >= 1 evidence
  intermediate >= 4 evidences, mean(last 4) >= 0.60, min(last 4) >= 0.30
  advanced     >= 6 evidences spanning >= 2 activity types,
               mean(last 4) >= 0.75, min(last 4) >= 0.40

Same (scenario id, seed) always produces the identical evidence stream and
identical ground truth: streams are built from fixed performance lists, with
only intra-day timestamp minutes drawn from random.Random(seed).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

BASE_DATE = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)

SOURCE_TO_ACTIVITY = {
    "guide": "dialogue",
    "evaluator": "quiz",
    "practice": "practice",
}


@dataclass(frozen=True)
class Topic:
    id: str
    name: str


@dataclass(frozen=True)
class Learner:
    id: str
    name: str


@dataclass(frozen=True)
class ProbePoint:
    label: str
    evidence_index: int  # ingest evidence[:evidence_index], then probe


@dataclass
class ScenarioSpec:
    id: str
    title: str
    seed: int
    topics: List[Topic]
    learners: List[Learner]
    plan_order: List[str]
    prereqs: Dict[str, List[str]]
    evidence: List[Dict[str, Any]]
    probes: List[ProbePoint]


class StreamBuilder:
    """Appends evidence items in chronological order with stable ids."""

    def __init__(self, scenario_id: str, seed: int):
        self.scenario_id = scenario_id
        self.rng = random.Random(seed)
        self.evidence: List[Dict[str, Any]] = []
        self._counters: Dict[str, int] = {}

    def add(
        self,
        learner_id: str,
        topic: Topic,
        source_agent: str,
        performance: float,
        day: int,
        attempts: int = 2,
        hints_used: int = 0,
        activity_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = f"{self.scenario_id}-{learner_id}"
        self._counters[key] = self._counters.get(key, 0) + 1
        n = self._counters[key]
        minute_jitter = self.rng.randrange(0, 60)
        occurred = BASE_DATE + timedelta(days=day, minutes=minute_jitter)
        ev = {
            "id": f"ev-{self.scenario_id}-{learner_id}-{n:03d}",
            "learner_id": learner_id,
            "node_id": topic.id,
            "topic_name": topic.name,
            "source_agent": source_agent,
            "activity_type": activity_type or SOURCE_TO_ACTIVITY[source_agent],
            "occurred_at": occurred.isoformat(),
            "performance": performance,
            "attempts": attempts,
            "hints_used": hints_used,
        }
        self.evidence.append(ev)
        return ev


# ------------------------------------------------------------- ground truth

def ground_truth_mastery(evidences: List[Dict[str, Any]]) -> str:
    """Stipulated rubric; see module docstring. Independent of any system."""
    n = len(evidences)
    if n == 0:
        return "unknown"
    if n < 4:
        return "beginner"
    last4 = evidences[-4:]
    mean4 = sum(e["performance"] for e in last4) / 4
    min4 = min(e["performance"] for e in last4)
    if (
        n >= 6
        and len({e["activity_type"] for e in evidences}) >= 2
        and mean4 >= 0.75
        and min4 >= 0.40
    ):
        return "advanced"
    if mean4 >= 0.60 and min4 >= 0.30:
        return "intermediate"
    return "beginner"


def evidence_for(
    evidences: List[Dict[str, Any]], learner_id: str, node_id: str
) -> List[Dict[str, Any]]:
    return [
        e
        for e in evidences
        if e["learner_id"] == learner_id and e["node_id"] == node_id
    ]


def ground_truth_levels(spec: ScenarioSpec, upto: int) -> Dict[str, Dict[str, str]]:
    """learner_id -> {topic_id -> stipulated level} from evidence[:upto]."""
    out: Dict[str, Dict[str, str]] = {}
    for learner in spec.learners:
        levels = {}
        for topic in spec.topics:
            evs = evidence_for(spec.evidence[:upto], learner.id, topic.id)
            levels[topic.id] = ground_truth_mastery(evs)
        out[learner.id] = levels
    return out


def ground_truth_next_topic(
    spec: ScenarioSpec, levels: Dict[str, Dict[str, str]], learner_id: str
) -> Optional[str]:
    """First plan-order topic not yet advanced whose prereqs are all advanced
    (ground-truth levels). None when every topic is advanced."""
    lv = levels[learner_id]
    for tid in spec.plan_order:
        if lv[tid] == "advanced":
            continue
        if all(lv[p] == "advanced" for p in spec.prereqs.get(tid, [])):
            return tid
    return None


def ground_truth_weakest(
    spec: ScenarioSpec,
    levels: Dict[str, Dict[str, str]],
    learner_id: str,
    n: int,
) -> List[str]:
    """n lowest ground-truth topics; ties broken by plan order (deterministic)."""
    from systems.base import ORDINAL

    lv = levels[learner_id]
    ranked = sorted(spec.plan_order, key=lambda tid: (ORDINAL[lv[tid]], spec.plan_order.index(tid)))
    return ranked[:n]
