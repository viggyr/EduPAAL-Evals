# Scenarios

Deterministic, seeded scenario scripts. Each scenario defines:

- a fixed topic set, learner set, plan order, and prerequisite DAG,
- a chronological evidence stream (stable evidence ids like `ev-s1-s1-learner-001`),
- probe points: `(label, evidence_index)` — ingest `evidence[:evidence_index]`,
  then probe every system identically.

**Same `(scenario id, seed)` → identical stream, identical ground truth, always.**
Streams are built from fixed performance lists; only intra-day timestamp
minutes are drawn from `random.Random(seed)`, so the seed is load-bearing but
the pedagogy-relevant content is fully inspectable in `catalog.py`.

## Ground-truth rubric (stipulated, not pedagogical)

Ground truth is a *fixed measurement instrument*, deliberately close to but
**not identical with** EduPAAL's shipped heuristic defaults. The differences
are intentional: if ground truth re-implemented EduPAAL's exact thresholds
(K=3, T_intermediate=0.65, T_advanced=0.80, cross-modal advanced), EduPAAL
would score 100% by construction and the comparison would be circular. The
eval measures agreement with an *independent* notion of mastery.

For a topic's evidence list E (chronological):

| Level | Condition |
|---|---|
| `unknown` | \|E\| = 0 |
| `beginner` | \|E\| ≥ 1 |
| `intermediate` | \|E\| ≥ 4, mean(last 4 performances) ≥ 0.60, min(last 4) ≥ 0.30 |
| `advanced` | \|E\| ≥ 6 **and** ≥ 2 distinct activity types in E **and** mean(last 4) ≥ 0.75 **and** min(last 4) ≥ 0.40 |

Checked top-down: advanced first, then intermediate, else beginner.

Derived ground truths:

- **next_topic**: first topic in plan order that is not `advanced` and whose
  prerequisites are all `advanced` (ground-truth levels); `None` when every
  topic is advanced. Note: this is stricter than EduPAAL's shipped
  `prereq_gate` (INTERMEDIATE) — the gap is measured, not hidden.
- **weakest(n)**: n topics with the lowest ground-truth ordinal
  (unknown=0 … advanced=3); ties broken by plan order.

## Catalog

| ID | Title | Learners | Topics | Evidence | Probes | What it stresses |
|---|---|---|---|---|---|---|
| S1 | Steady learner | 1 | t1 Fractions, t2 Decimals, t3 Percentages, t4 Ratios | 12 over 21 days, 3 verticals | mid(7), final(12) | Cross-vertical accumulation; final GT: t1 advanced, t2 intermediate, t3 beginner, t4 unknown |
| S2 | Struggling learner | 1 | a1 Linear Equations, a2 Quadratic Equations | 8; strong dialogue vs weak quizzes on a1 | mid(6), final(8) | Disagreement handling; GT a1 intermediate despite two weak quizzes |
| S3 | Prerequisites | 1 | pa Variables → pb Expressions → pc Equations | 8; pa mastered, pb started, pc untouched | mid(6), final(8) | `next_topic` must never return a topic whose prereqs aren't advanced |
| S4 | Plateau | 1 | p1 Probability | 6, all 0.68–0.72 | mid(4), final(6) | Advanced predictions here are false positives; GT stays intermediate |
| S5 | Isolation | 2 (A, B) | x1 Photosynthesis, x2 Cell Division | 8 interleaved; A masters x1, B barely starts x2 | mid(4), final(8) | No cross-learner leakage: A's x2 and B's x1 must stay unknown |

## Evidence dicts

Every stream item carries: `id`, `learner_id`, `node_id`, `topic_name`,
`source_agent` (guide | evaluator | practice), `activity_type`
(dialogue | quiz | practice), `occurred_at` (ISO-8601), `performance`
(0..1), `attempts`, `hints_used`. This is the exact contract
`SystemUnderTest.ingest_evidence` consumes.
