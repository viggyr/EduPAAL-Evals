"""Mem0 baseline: native general memory, used as-is.

Each evidence item is stored as ONE natural-language narrative via
``Memory.add(..., infer=True)`` — Mem0's native extraction path (the exact
opposite of the EduPAAL Mem0 *provider*, which writes verbatim records with
``infer=False``). Probes retrieve memories with ``Memory.search`` scoped to
the learner's ``user_id`` and hand them to the LLM judge for strict-schema
assessment.

Requires the baselines extra (``pip install "edupaal-evals[baselines]"``) and
the judge environment (``EDUPAAL_EVALS_LLM_*``). Both fail loudly.

Wiring notes (verified against the installed mem0ai source):

- Embeddings are local fastembed ``BAAI/bge-small-en-v1.5`` (384 dims); the
  vector-store config sets ``embedding_model_dims=384`` to match because
  mem0's default is 1536 (OpenAI's) and a mismatch fails every search.
- fastembed's default model cache is ``tempfile.gettempdir()`` (ephemeral),
  so the leg pins ``FASTEMBED_CACHE_PATH`` to ``~/.cache/edupaal-evals``.
  Environment-only change: mem0's behavior is untouched.
- Storage is hermetic: each instance gets its own temp-dir Qdrant folder, so
  any number of Mem0 legs can coexist in one process (local-mode Qdrant
  takes an exclusive per-folder lock). mem0's product telemetry is disabled
  (``MEM0_TELEMETRY=false``): besides the PostHog phone-home, it opens a
  second Qdrant client at the fixed path ``~/.mem0/migrations_qdrant``,
  which breaks multi-instance use. Documented opt-out; memory behavior
  untouched.
- Token accounting covers the strict-schema judge calls only; mem0's
  internal extraction LLM calls (``infer=True``) are not visible to the
  harness and are excluded from reported totals.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from .base import SystemUnderTest
from .judge import LLMJudge, _required_env
from .narratives import evidence_narrative, probe_query


class Mem0System(SystemUnderTest):
    name = "mem0"

    def __init__(self, scenario) -> None:
        super().__init__(scenario)
        # Fail loudly before touching any framework: no env => no evaluation.
        env = _required_env()
        # mem0's product telemetry (PostHog phone-home) also spins up a second
        # Qdrant client at a FIXED path (~/.mem0/migrations_qdrant), so two live
        # Memory instances in one process collide on its exclusive folder lock.
        # This is mem0's documented opt-out; memory behavior is untouched.
        # Must be set before the first mem0 import (read at import time).
        os.environ.setdefault("MEM0_TELEMETRY", "false")
        try:
            from mem0 import Memory
        except Exception as exc:
            raise RuntimeError(
                "mem0ai is not installed. Install the baselines extra: "
                'pip install "edupaal-evals[baselines]".'
            ) from exc
        self.judge = LLMJudge()  # also fail-loud on missing env
        # Persistent embedder cache (see module docstring). Must be set before
        # Memory.from_config constructs the fastembed embedder.
        os.environ.setdefault(
            "FASTEMBED_CACHE_PATH",
            os.path.join(
                os.path.expanduser("~"), ".cache", "edupaal-evals", "fastembed"
            ),
        )
        self._tmp = tempfile.TemporaryDirectory(prefix="edupaal-evals-mem0-")
        config = {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": env["EDUPAAL_EVALS_LLM_MODEL"],
                    "api_key": env["EDUPAAL_EVALS_LLM_API_KEY"],
                    "openai_base_url": env["EDUPAAL_EVALS_LLM_BASE_URL"],
                },
            },
            "embedder": {
                "provider": "fastembed",
                "config": {"model": "BAAI/bge-small-en-v1.5"},
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": os.path.join(self._tmp.name, "qdrant"),
                    # Must match the fastembed model above (bge-small-en-v1.5
                    # is 384 dims). mem0's MemoryConfig default is 1536 (the
                    # OpenAI default); without this the collection is created
                    # at 1536 while vectors are embedded at 384 and every
                    # vector-store search fails with a shape mismatch.
                    "embedding_model_dims": 384,
                },
            },
        }
        self.memory = Memory.from_config(config)
        # (learner_id, probe_label) -> (Judgment, memories_shown)
        self._judgments: Dict[Tuple[str, str], Tuple[Any, int]] = {}
        self._current_probe = ""
        self.token_usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    # ------------------------------------------------------- probe plumbing

    def set_probe(self, probe_label: str) -> None:
        """The runner sets this before each probe batch: one judge call serves
        every probe method (mastery/grounding/next/weakest/citations)."""
        self._current_probe = probe_label

    def _judgment(self, learner_id: str):
        key = (learner_id, self._current_probe)
        if key not in self._judgments:
            judgment = self.timed("judge_probe", self._run_judge, learner_id)
            self.token_usage["prompt_tokens"] += judgment.prompt_tokens
            self.token_usage["completion_tokens"] += judgment.completion_tokens
            self._judgments[key] = judgment
        return self._judgments[key]

    def _run_judge(self, learner_id: str):
        # mem0 >= 1.1: entity scoping moved from top-level kwargs to filters,
        # and the result-count knob is top_k.
        res = self.memory.search(
            probe_query(learner_id), filters={"user_id": learner_id}, top_k=30
        )
        memories = [r.get("memory", "") for r in res.get("results", []) if r.get("memory")]
        judgment = self.judge.judge(
            learner_id=learner_id,
            topics=[{"id": t.id, "name": t.name} for t in self.scenario.topics],
            prereqs=self.scenario.prereqs,
            plan_order=self.scenario.plan_order,
            memories=memories,
            probe_label=self._current_probe,
            weakest_n=2,
        )
        judgment.memories_shown = len(memories)  # informational only
        return judgment

    # ------------------------------------------------------------------ ops

    def ingest_evidence(self, evidence: Dict[str, Any]) -> None:
        def _do():
            self.memory.add(
                evidence_narrative(evidence),
                user_id=evidence["learner_id"],
                metadata={
                    "ev_id": evidence["id"],
                    "node_id": evidence["node_id"],
                    "learner_id": evidence["learner_id"],
                },
                infer=True,  # native path: let Mem0 extract facts as for any agent
            )

        self.timed("ingest", _do)

    def mastery_of(self, learner_id: str, node_id: str) -> str:
        return self._judgment(learner_id).mastery[node_id]

    def grounding(self, learner_id: str, node_id: str) -> Dict[str, str]:
        return dict(self._judgment(learner_id).mastery)

    def next_topic(self, learner_id: str) -> Optional[str]:
        return self._judgment(learner_id).next_topic

    def weakest(self, learner_id: str, n: int) -> List[str]:
        return self._judgment(learner_id).weakest[:n]

    def cited_evidence(self, learner_id: str, node_id: str) -> List[str]:
        return list(self._judgment(learner_id).cited_evidence.get(node_id, []))

    def close(self) -> None:
        self._tmp.cleanup()
