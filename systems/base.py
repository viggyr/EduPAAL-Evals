"""Common interface for every system under test.

A system under test (SUT) is one memory backend behind an identical API so a
scenario can replay the same evidence stream against EduPAAL, native Mem0, or
native MemOS and the metrics can compare answers to one ground truth.

Evidence dicts (produced by ``scenarios``) carry::

    {
        "id": "ev-s1-001",            # stable, citable evidence id
        "learner_id": "s1-learner",
        "node_id": "t1",              # scenario topic id
        "topic_name": "Fractions",    # human-readable, for narratives
        "source_agent": "guide",      # guide | evaluator | practice
        "activity_type": "dialogue",  # dialogue | quiz | practice
        "occurred_at": "2026-09-01T10:00:00+00:00",
        "performance": 0.72,          # normalized 0..1
        "attempts": 3,
        "hints_used": 1,
    }

Mastery levels are the EduPAAL vocabulary so every system answers in the same
units: ``unknown`` | ``beginner`` | ``intermediate`` | ``advanced``.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

LEVELS = ("unknown", "beginner", "intermediate", "advanced")
ORDINAL = {lvl: i for i, lvl in enumerate(LEVELS)}


class SystemUnderTest(ABC):
    """One memory backend behind the eval's uniform probe API."""

    #: short id used in result files, e.g. "edupaal", "mem0", "memos"
    name: str = "base"

    def __init__(self, scenario: "ScenarioSpec") -> None:
        self.scenario = scenario
        # op_name -> list of seconds; the runner aggregates p50/p95.
        self.latencies: Dict[str, List[float]] = {}

    # ------------------------------------------------------------------ ops

    @abstractmethod
    def ingest_evidence(self, evidence: Dict[str, Any]) -> None:
        """Store one evidence item. Must be idempotent-safe per evidence id."""

    @abstractmethod
    def mastery_of(self, learner_id: str, node_id: str) -> str:
        """Current mastery level for (learner, topic); one of LEVELS."""

    @abstractmethod
    def grounding(self, learner_id: str, node_id: str) -> Dict[str, str]:
        """Topic id -> level for every topic in the scenario plan."""

    @abstractmethod
    def next_topic(self, learner_id: str) -> Optional[str]:
        """Next topic id for the learner, or None when everything is done."""

    @abstractmethod
    def weakest(self, learner_id: str, n: int) -> List[str]:
        """The n weakest topic ids, weakest first."""

    @abstractmethod
    def cited_evidence(self, learner_id: str, node_id: str) -> List[str]:
        """Evidence ids the system can point to for its mastery claim.

        Empty list = the system makes the claim without citing evidence.
        """

    @abstractmethod
    def close(self) -> None:
        """Release resources (temp dirs, clients)."""

    # ------------------------------------------------------------- helpers

    def timed(self, op: str, fn, *args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            self.latencies.setdefault(op, []).append(time.perf_counter() - start)


# Type-only import to avoid a hard dependency cycle at import time.
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from scenarios.common import ScenarioSpec
