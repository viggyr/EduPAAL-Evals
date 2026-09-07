"""Scenario determinism and ground-truth sanity.

Same (scenario, seed) must always produce the identical evidence stream and
identical ground truth. Ground-truth values for the fixed streams are
asserted explicitly so a rubric change fails loudly here, not silently in a
report.
"""

from scenarios import get_scenario, ground_truth_levels, ground_truth_next_topic, ground_truth_weakest


def _levels(sid, seed=7, upto=None):
    spec = get_scenario(sid, seed)
    upto = len(spec.evidence) if upto is None else upto
    return spec, ground_truth_levels(spec, upto)


def test_same_seed_identical_stream():
    a = get_scenario("S1", 7)
    b = get_scenario("S1", 7)
    assert a.evidence == b.evidence
    assert [t.id for t in a.topics] == [t.id for t in b.topics]


def test_evidence_ids_stable_and_unique():
    for sid in ("S1", "S2", "S3", "S4", "S5"):
        spec = get_scenario(sid, 7)
        ids = [e["id"] for e in spec.evidence]
        assert len(ids) == len(set(ids)), sid
        assert all(i.startswith(f"ev-{sid.lower()}-") for i in ids), sid


def test_s1_ground_truth_final():
    spec, lv = _levels("S1")
    L = "s1-learner"
    assert lv[L] == {"t1": "advanced", "t2": "intermediate", "t3": "beginner", "t4": "unknown"}
    assert ground_truth_next_topic(spec, lv, L) == "t2"
    assert ground_truth_weakest(spec, lv, L, 2) == ["t4", "t3"]


def test_s1_ground_truth_mid():
    spec = get_scenario("S1", 7)
    lv = ground_truth_levels(spec, 7)
    L = "s1-learner"
    assert lv[L]["t1"] == "intermediate"
    assert lv[L]["t2"] == "beginner"
    assert lv[L]["t3"] == "unknown"
    assert ground_truth_next_topic(spec, lv, L) == "t1"


def test_s2_conflict_ground_truth():
    spec, lv = _levels("S2")
    L = "s2-learner"
    # strong dialogue outweighs two weak quizzes under the stipulated rubric
    assert lv[L] == {"a1": "intermediate", "a2": "beginner"}
    assert ground_truth_next_topic(spec, lv, L) == "a1"


def test_s3_prereq_ground_truth():
    spec, lv = _levels("S3")
    L = "s3-learner"
    assert lv[L] == {"pa": "advanced", "pb": "beginner", "pc": "unknown"}
    assert ground_truth_next_topic(spec, lv, L) == "pb"
    assert ground_truth_weakest(spec, lv, L, 2) == ["pc", "pb"]


def test_s4_plateau_never_advanced():
    spec = get_scenario("S4", 7)
    for upto in (4, 6):
        lv = ground_truth_levels(spec, upto)
        assert lv["s4-learner"]["p1"] == "intermediate", upto


def test_s5_isolation_ground_truth():
    spec, lv = _levels("S5")
    assert lv["s5-learner-a"] == {"x1": "advanced", "x2": "unknown"}
    assert lv["s5-learner-b"] == {"x1": "unknown", "x2": "beginner"}
    assert ground_truth_next_topic(spec, lv, "s5-learner-a") == "x2"
    assert ground_truth_next_topic(spec, lv, "s5-learner-b") == "x1"
