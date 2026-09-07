"""EduPAAL leg: the real EduPAALSkill over SQLite. No LLM anywhere in this leg."""

from __future__ import annotations

import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

from edupaal import (
    EduPAALSkill,
    Evidence,
    KnowledgeGraph,
    KnowledgeNode,
    LearnerPreferences,
    MasteryLevel,
    NodeLevel,
    SQLiteBackend,
)

from .base import SystemUnderTest


def _parse_dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso)


class EduPAALSystem(SystemUnderTest):
    name = "edupaal"

    def __init__(self, scenario) -> None:
        super().__init__(scenario)
        self._tmp = tempfile.TemporaryDirectory(prefix="edupaal-evals-")
        self.graph = self._build_graph(scenario)
        self.store = SQLiteBackend(f"{self._tmp.name}/eval.db")
        self.skills: Dict[str, EduPAALSkill] = {}
        prefs = LearnerPreferences(learning_style="mixed", pace="steady")
        topic_ids = [t.id for t in scenario.topics]
        for learner in scenario.learners:
            skill = EduPAALSkill(store=self.store, graph=self.graph)
            # Plan order follows the scenario; defaults (heuristic-v1) stay untouched
            # so the eval measures the framework as shipped.
            skill.cold_start(
                learner_id=learner.id,
                node_selection=topic_ids,
                preferences=prefs,
                order=scenario.plan_order,
                learner_name=learner.name,
            )
            self.skills[learner.id] = skill

    # ------------------------------------------------------------------ ops

    def ingest_evidence(self, evidence: Dict[str, Any]) -> None:
        def _do():
            skill = self.skills[evidence["learner_id"]]
            skill.record_evidence(
                Evidence(
                    id=evidence["id"],
                    learner_id=evidence["learner_id"],
                    node_id=evidence["node_id"],
                    source_agent=evidence["source_agent"],
                    activity_type=evidence["activity_type"],
                    occurred_at=_parse_dt(evidence["occurred_at"]),
                    performance=evidence["performance"],
                    attempts=evidence.get("attempts"),
                    hints_used=evidence.get("hints_used"),
                )
            )

        self.timed("ingest", _do)

    def mastery_of(self, learner_id: str, node_id: str) -> str:
        return self.timed(
            "mastery", self.skills[learner_id].effective_mastery, node_id
        ).value

    def grounding(self, learner_id: str, node_id: str) -> Dict[str, str]:
        def _do():
            skill = self.skills[learner_id]
            return {
                t.id: skill.effective_mastery(t.id).value for t in self.scenario.topics
            }

        return self.timed("grounding", _do)

    def next_topic(self, learner_id: str) -> Optional[str]:
        node = self.timed("next_topic", self.skills[learner_id].next_topic)
        return node.id if node else None

    def weakest(self, learner_id: str, n: int) -> List[str]:
        ranked = self.timed("weakest", self.skills[learner_id].weakest, n)
        return [node.id for node, _lvl, _score in ranked]

    def cited_evidence(self, learner_id: str, node_id: str) -> List[str]:
        def _do():
            skill = self.skills[learner_id]
            if skill.effective_mastery(node_id) == MasteryLevel.UNKNOWN:
                return []
            history = skill.mastery_history(node_id)
            if not history:
                return []
            return list(history[-1].evidence_ids)

        return self.timed("provenance", _do)

    def close(self) -> None:
        self._tmp.cleanup()

    # --------------------------------------------------------------- helpers

    def _build_graph(self, scenario) -> KnowledgeGraph:
        g = KnowledgeGraph()
        g.add_node(KnowledgeNode(id="space-eval", level=NodeLevel.SPACE, name="Eval Space"))
        g.add_node(
            KnowledgeNode(
                id="subject-eval",
                level=NodeLevel.SUBJECT,
                name="Eval Subject",
                parent_id="space-eval",
            )
        )
        g.add_node(
            KnowledgeNode(
                id=f"concept-{scenario.id}",
                level=NodeLevel.CONCEPT,
                name=f"Eval Concept {scenario.id}",
                parent_id="subject-eval",
            )
        )
        for t in scenario.topics:
            g.add_node(
                KnowledgeNode(
                    id=t.id,
                    level=NodeLevel.TOPIC,
                    name=t.name,
                    parent_id=f"concept-{scenario.id}",
                )
            )
        for tid, pres in scenario.prereqs.items():
            for p in pres:
                g.add_prerequisite(tid, p)
        return g
