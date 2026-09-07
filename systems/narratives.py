"""Natural-language evidence narratives for the baseline legs.

The baselines (native Mem0 / MemOS) are used exactly as general agent memory:
each evidence item becomes one narrative memory, stored with the framework's
native ingestion (Mem0 ``infer=True`` extraction; MemOS top-level ``add``
with its extraction pipeline). Nothing EduPAAL-specific is smuggled in —
the LLM judge later reads only what each framework retrieved.
"""

from __future__ import annotations

from typing import Any, Dict

_AGENT_DESC = {
    "guide": "the learning-guide agent held a dialogue session",
    "evaluator": "the evaluator agent ran a quiz",
    "practice": "the practice agent ran a practice set",
}


def evidence_narrative(ev: Dict[str, Any]) -> str:
    agent = _AGENT_DESC.get(ev["source_agent"], f"the {ev['source_agent']} agent recorded an activity")
    bits = [
        f"Evidence {ev['id']}.",
        f"Date: {ev['occurred_at'][:10]}.",
        f"Learner {ev['learner_id']}: {agent} on topic '{ev['topic_name']}' (activity: {ev['activity_type']}).",
        f"Normalized performance: {ev['performance']:.2f} out of 1.00.",
    ]
    if ev.get("attempts") is not None:
        bits.append(f"Attempts: {ev['attempts']}.")
    if ev.get("hints_used") is not None:
        bits.append(f"Hints used: {ev['hints_used']}.")
    return " ".join(bits)


def probe_query(learner_id: str) -> str:
    return (
        f"What is the learning progress and mastery of learner {learner_id}? "
        "List every topic studied, the evidence observed, and the learner's "
        "current level on each topic."
    )
