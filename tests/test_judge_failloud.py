"""Fail-loud guarantees: without configuration, baseline legs must raise.

These tests prove the harness refuses to fabricate baseline evaluations:
- no EDUPAAL_EVALS_LLM_API_KEY -> LLMJudge() raises RuntimeError
- mem0/memos system constructors raise before touching any framework
  (mem0ai / MemoryOS imports are skipped if the packages are absent)

Run with the key deliberately UNSET: the CI/dev environment for these tests
must not define EDUPAAL_EVALS_LLM_API_KEY.
"""

import os

import pytest

from scenarios import get_scenario


@pytest.fixture(autouse=True)
def no_llm_key(monkeypatch):
    monkeypatch.delenv("EDUPAAL_EVALS_LLM_API_KEY", raising=False)


def test_judge_raises_without_key():
    from systems.judge import LLMJudge

    with pytest.raises(RuntimeError, match="EDUPAAL_EVALS_LLM_API_KEY"):
        LLMJudge()


def test_mem0_system_raises_without_key():
    mem0 = pytest.importorskip("mem0")
    from systems.mem0_system import Mem0System

    spec = get_scenario("S1", 7)
    with pytest.raises(RuntimeError, match="EDUPAAL_EVALS_LLM_API_KEY"):
        Mem0System(spec)


def test_memos_system_raises_without_key():
    pytest.importorskip("memos")
    from systems.memos_system import MemOSSystem

    spec = get_scenario("S1", 7)
    with pytest.raises(RuntimeError, match="EDUPAAL_EVALS_LLM_API_KEY"):
        MemOSSystem(spec)


def test_edupaal_system_needs_no_key():
    """The EduPAAL leg is fully functional with no LLM configuration."""
    pytest.importorskip("edupaal")
    from systems.edupaal_system import EduPAALSystem

    spec = get_scenario("S4", 7)
    system = EduPAALSystem(spec)
    try:
        for ev in spec.evidence:
            system.ingest_evidence(ev)
        assert system.mastery_of("s4-learner", "p1") in ("beginner", "intermediate")
        assert system.next_topic("s4-learner") == "p1"
        assert system.grounding("s4-learner", "p1")["p1"] != "unknown"
        assert system.weakest("s4-learner", 2) == ["p1"]
    finally:
        system.close()
