"""Mem0 leg hermetic-storage regression test.

Two Mem0System instances constructed in one process must not share vector
storage. This is the bug class that broke the MemOS leg: local-mode Qdrant
takes an exclusive per-folder file lock, so scenarios sharing one folder
cannot each hold an open client. The Mem0 leg avoids it with per-instance
temp-dir Qdrant paths; this test pins that behavior.

Uses ``infer=False`` for the probe write so no LLM calls are made —
embeddings are local fastembed. Dummy LLM env vars suffice because the
OpenAI client is constructed but never called on this path. Skips if the
mem0ai package is absent.
"""

import pytest

from scenarios import get_scenario


@pytest.fixture()
def dummy_llm_env(monkeypatch):
    monkeypatch.setenv("EDUPAAL_EVALS_LLM_API_KEY", "dummy")
    monkeypatch.setenv("EDUPAAL_EVALS_LLM_BASE_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("EDUPAAL_EVALS_LLM_MODEL", "dummy-model")


def test_two_mem0_instances_are_storage_isolated(dummy_llm_env):
    pytest.importorskip("mem0")
    from systems.mem0_system import Mem0System

    s1 = Mem0System(get_scenario("S1", 7))
    s2 = Mem0System(get_scenario("S2", 7))
    try:
        # Distinct storage folders: no shared Qdrant lock is possible.
        assert s1._tmp.name != s2._tmp.name

        s1.memory.add(
            "Learner s1-learner completed a fractions dialogue with score 0.9.",
            user_id="s1-learner",
            infer=False,  # no LLM; local embed only
        )

        # Instance 1 retrieves its own write...
        r1 = s1.memory.search(
            "fractions dialogue", filters={"user_id": "s1-learner"}, top_k=5
        )
        assert any(
            "fractions" in (r.get("memory") or "")
            for r in r1.get("results", [])
        ), "instance 1 cannot see its own write"

        # ...instance 2 sees nothing: storage is isolated.
        r2 = s2.memory.search(
            "fractions dialogue", filters={"user_id": "s1-learner"}, top_k=5
        )
        assert r2.get("results", []) == [], "instance 2 leaked instance 1's data"
    finally:
        s1.close()
        s2.close()
