"""MemOS baseline: native general memory, used as-is.

Each evidence item is ingested as a natural-language narrative through
MemOS's top-level ``MOS.add`` — the native path that runs MemOS's extraction
pipeline (the exact opposite of the EduPAAL MemOS *provider*, which bypasses
ingestion and writes verbatim records via ``NaiveTextMemory``). Probes use
``MOS.search`` scoped to the learner and hand retrieved memories to the LLM
judge for strict-schema assessment.

Requires the baselines extra (``pip install "edupaal-evals[baselines]"``) and
the judge environment (``EDUPAAL_EVALS_LLM_*``). Both fail loudly.

MemOS wiring notes (verified against the installed MemoryOS source):

- The chat/extractor LLM is configured explicitly from
  ``EDUPAAL_EVALS_LLM_MODEL`` at ``EDUPAAL_EVALS_LLM_BASE_URL``. MemOS's
  ``MOS.simple()`` auto-config hardcodes ``gpt-4o-mini``, which does not exist
  on the Gemini endpoint, so we build the config via ``get_default`` instead.
- Embeddings come from the same base URL's OpenAI-compatible ``/embeddings``
  endpoint (``EDUPAAL_EVALS_MEMOS_EMBEDDER_MODEL``, default
  ``gemini-embedding-001``; its 3072-dim output matches MemOS's default
  ``vector_dimension``). A base URL that serves no embeddings endpoint (e.g.
  Groq) cannot run this leg — that fails loudly at ingest time, by design.
- MemOS requires explicit user registration: each scenario learner is created
  with the real ``MOS.create_user`` API and granted access to the scenario's
  cube. Arbitrary learner IDs are NOT assumed to work.
- Storage is hermetic: ``MEMOS_BASE_PATH`` points at a per-instance temp dir
  (set before importing memos, since ``memos.settings`` reads it at import
  time). The base path is process-global, so the first instance wins; each
  scenario additionally gets its own qdrant folder because QdrantClient's
  local mode takes an exclusive per-folder file lock — two open clients on
  one folder in the same process is a hard error. Per-instance cube/user ids
  keep collections separate regardless.
"""

from __future__ import annotations

import atexit
import os
import tempfile
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .base import SystemUnderTest
from .judge import LLMJudge, _required_env
from .narratives import evidence_narrative, probe_query

ENV_EMBEDDER_MODEL = "EDUPAAL_EVALS_MEMOS_EMBEDDER_MODEL"
DEFAULT_EMBEDDER_MODEL = "gemini-embedding-001"

# MEMOS_BASE_PATH is read by memos.settings at import time and is therefore
# process-global: the first MemOSSystem in the process wins, and later
# scenarios reuse that base path with per-instance cube/user ids. The tmpdirs
# must stay alive until process exit, so they are registered here and cleaned
# by atexit instead of in close().
_TMPDIRS: List[tempfile.TemporaryDirectory] = []
atexit.register(lambda: [t.cleanup() for t in _TMPDIRS])


def _memory_text(item: Any) -> str:
    for attr in ("memory", "content", "text"):
        val = getattr(item, attr, None)
        if isinstance(val, str) and val.strip():
            return val
    if isinstance(item, dict):
        for key in ("memory", "content", "text"):
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return val
    text = str(item).strip()
    return text


class MemOSSystem(SystemUnderTest):
    name = "memos"

    def __init__(self, scenario) -> None:
        super().__init__(scenario)
        # Fail loudly before touching any framework: no env => no evaluation.
        env = _required_env()
        embedder_model = os.environ.get(ENV_EMBEDDER_MODEL, DEFAULT_EMBEDDER_MODEL)
        # Hermetic storage: memos.settings reads MEMOS_BASE_PATH at import time,
        # so it must be set before the first memos import in this process.
        # Kept alive until process exit (see _TMPDIRS). Each scenario also
        # gets its own qdrant folder (see below) because local-mode Qdrant
        # takes an exclusive per-folder lock.
        self._tmp = tempfile.TemporaryDirectory(prefix="edupaal-evals-memos-")
        _TMPDIRS.append(self._tmp)
        os.environ.setdefault("MEMOS_BASE_PATH", self._tmp.name)
        try:
            from memos.mem_cube.general import GeneralMemCube
            from memos.mem_os.main import MOS
            from memos.mem_os.utils.default_config import (
                get_default_config,
                get_default_cube_config,
            )
        except Exception as exc:
            raise RuntimeError(
                "MemoryOS is not installed. Install the baselines extra: "
                'pip install "edupaal-evals[baselines]".'
            ) from exc
        self.judge = LLMJudge()  # also fail-loud on missing env
        # Per-instance ids keep cubes/collections isolated between scenarios.
        self._instance_uid = f"edupaal-evals-{uuid.uuid4().hex[:12]}"
        # get_default() would also work, but it builds the cube internally;
        # we need to set a per-scenario qdrant path on the cube config first
        # (QdrantClient local mode forbids two open clients on one folder).
        # The native ingestion/extraction/retrieval pipeline is otherwise
        # byte-identical to get_default().
        _cfg_kwargs = dict(
            openai_api_key=env["EDUPAAL_EVALS_LLM_API_KEY"],
            openai_api_base=env["EDUPAAL_EVALS_LLM_BASE_URL"],
            user_id=self._instance_uid,
            model_name=env["EDUPAAL_EVALS_LLM_MODEL"],
            embedder_model=embedder_model,
        )
        mos_config = get_default_config(**_cfg_kwargs)
        cube_config = get_default_cube_config(**_cfg_kwargs)
        cube_config.text_mem.config.vector_db.config.path = os.path.join(
            self._tmp.name, ".memos", "qdrant"
        )
        default_cube = GeneralMemCube(cube_config)
        self.mos = MOS(mos_config)
        self._cube_id = f"cube_{self._instance_uid}"
        self.mos.register_mem_cube(default_cube, mem_cube_id=self._cube_id)
        # Real user registration (MemOS rejects unknown user_ids at search).
        # Each learner is granted access to this scenario's cube.
        for learner in scenario.learners:
            self.mos.create_user(learner.id)
            self.mos.register_mem_cube(
                default_cube, mem_cube_id=self._cube_id, user_id=learner.id
            )
        self._judgments: Dict[Tuple[str, str], Any] = {}
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
        result = self.mos.search(probe_query(learner_id), user_id=learner_id, top_k=20)
        items = (result or {}).get("text_mem", []) or []
        memories = [_memory_text(i) for i in items]
        memories = [m for m in memories if m]
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
            # Native path: top-level add runs MemOS's extraction pipeline.
            self.mos.add(
                messages=[{"role": "user", "content": evidence_narrative(evidence)}],
                user_id=evidence["learner_id"],
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
        # Storage intentionally lives until process exit: MEMOS_BASE_PATH is
        # process-global and later scenarios in this process still need it.
        # Per-instance cube/user ids keep their data isolated.
        return None
