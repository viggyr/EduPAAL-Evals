#!/usr/bin/env python3
"""EduPAAL-Evals runner.

Usage:
    python run.py --systems edupaal,mem0,memos [--scenarios S1,S2] [--seed 7]

Each (system, scenario) pair runs independently: the scenario's evidence
stream is replayed, every probe point is evaluated against the scenario's
stipulated ground truth, and results land in results/<timestamp>/ as
raw.jsonl (every probe) plus report.md (aggregated tables).

Rules:
- One process per invocation. MemOS reads MEMOS_BASE_PATH at import time
  (process-global); per-instance cube/user ids keep scenarios isolated.
- A leg that cannot run (missing LLM key, missing baseline packages) fails
  loudly: its error is recorded in the report and the process exits non-zero
  at the end. Other legs still complete; nothing is filled with zeros.
- Every number in the report comes from a real run. No mocks anywhere.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from metrics import (  # noqa: E402
    mastery_exact_match,
    ordinal_mae,
    monotonicity,
    cross_vertical_agreement,
    grounding_precision_recall,
    grounding_accuracy,
    next_topic_exact_match,
    prereq_violation_rate,
    weakest_hit_rate,
    provenance_coverage,
    latency_stats,
)
from scenarios import (  # noqa: E402
    ALL_SCENARIOS,
    ScenarioSpec,
    evidence_for,
    get_scenario,
    ground_truth_levels,
    ground_truth_next_topic,
    ground_truth_weakest,
)
from systems.edupaal_system import EduPAALSystem  # noqa: E402

WEAKEST_N = 2

SYSTEM_BUILDERS = {
    "edupaal": EduPAALSystem,
    "mem0": None,  # lazy: importing systems.mem0_system is harmless, but the
    "memos": None,  # constructors fail loudly on missing key/packages.
}


def _load_system(name: str):
    if name == "edupaal":
        return EduPAALSystem
    if name == "mem0":
        from systems.mem0_system import Mem0System

        return Mem0System
    if name == "memos":
        from systems.memos_system import MemOSSystem

        return MemOSSystem
    raise ValueError(f"unknown system {name!r}; choose from edupaal, mem0, memos")


def run_pair(system_name: str, spec: ScenarioSpec, out: dict) -> dict:
    """Run one system on one scenario. Returns the summary dict."""
    builder = _load_system(system_name)
    system = builder(spec)  # raises loudly when the leg cannot run
    set_probe = getattr(system, "set_probe", None)

    mastery_pairs = []
    topic_levels: dict = {}
    grounding_accs = []
    precisions, recalls = [], []
    next_pairs = []
    violation_probes = []
    weakest_pairs = []
    prov_claims = []
    probe_records = []

    ingested = 0
    try:
        for probe in spec.probes:
            new_items = spec.evidence[ingested : probe.evidence_index]
            for ev in new_items:
                system.ingest_evidence(ev)
            ingested = probe.evidence_index
            if set_probe:
                set_probe(f"{spec.id}-{probe.label}")

            gt = ground_truth_levels(spec, probe.evidence_index)
            for learner in spec.learners:
                lid = learner.id
                pred_levels = {}
                for topic in spec.topics:
                    pred = system.mastery_of(lid, topic.id)
                    truth = gt[lid][topic.id]
                    pred_levels[topic.id] = pred
                    mastery_pairs.append((pred, truth))
                    topic_levels.setdefault((lid, topic.id), []).append(pred)
                    actual_ids = [
                        e["id"]
                        for e in evidence_for(
                            spec.evidence[: probe.evidence_index], lid, topic.id
                        )
                    ]
                    if pred != "unknown":
                        prov_claims.append(
                            (system.cited_evidence(lid, topic.id), actual_ids)
                        )
                grounding = system.grounding(lid, spec.topics[0].id)
                p, r = grounding_precision_recall(grounding, gt[lid])
                if p is not None:
                    precisions.append(p)
                if r is not None:
                    recalls.append(r)
                grounding_accs.append(grounding_accuracy(grounding, gt[lid]))

                gt_next = ground_truth_next_topic(spec, gt, lid)
                pred_next = system.next_topic(lid)
                next_pairs.append((pred_next, gt_next))
                violation_probes.append((pred_next, gt[lid], spec.prereqs))

                pred_weak = system.weakest(lid, WEAKEST_N)
                gt_weak = ground_truth_weakest(spec, gt, lid, WEAKEST_N)
                weakest_pairs.append((pred_weak, gt_weak))

                probe_records.append(
                    {
                        "system": system_name,
                        "scenario": spec.id,
                        "seed": spec.seed,
                        "probe": probe.label,
                        "learner": lid,
                        "predicted_levels": pred_levels,
                        "truth_levels": gt[lid],
                        "predicted_next": pred_next,
                        "truth_next": gt_next,
                        "predicted_weakest": pred_weak,
                        "truth_weakest": gt_weak,
                    }
                )
    finally:
        system.close()

    coverage, cite_prec = provenance_coverage(prov_claims)
    tokens = getattr(system, "token_usage", {"prompt_tokens": 0, "completion_tokens": 0})
    summary = {
        "system": system_name,
        "scenario": spec.id,
        "seed": spec.seed,
        "n_evidence": len(spec.evidence),
        "mastery_exact_match": mastery_exact_match(mastery_pairs),
        "ordinal_mae": ordinal_mae(mastery_pairs),
        "monotonicity": monotonicity(topic_levels),
        "cross_vertical_agreement": cross_vertical_agreement(topic_levels),
        "grounding_accuracy": _mean(grounding_accs),
        "grounding_precision": _mean(precisions),
        "grounding_recall": _mean(recalls),
        "next_topic_exact_match": next_topic_exact_match(next_pairs),
        "prereq_violation_rate": prereq_violation_rate(violation_probes),
        "weakest_hit_rate": weakest_hit_rate(weakest_pairs),
        "provenance_coverage": coverage,
        "citation_precision": cite_prec,
        "latency": latency_stats(system.latencies),
        "prompt_tokens": tokens.get("prompt_tokens", 0),
        "completion_tokens": tokens.get("completion_tokens", 0),
    }
    out["probe_records"].extend(probe_records)
    return summary


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_report(path: str, summaries, failures, args) -> None:
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# EduPAAL-Evals report",
        "",
        f"- timestamp (UTC): {ts}",
        f"- seed: {args.seed}",
        f"- systems: {', '.join(args.systems)}",
        f"- scenarios: {', '.join(args.scenarios)}",
        "- ground truth: stipulated rubric in scenarios/README.md (independent of all systems)",
        "",
    ]
    metric_rows = [
        ("mastery_exact_match", "Mastery exact-match"),
        ("ordinal_mae", "Ordinal MAE (0..3)"),
        ("monotonicity", "Monotonicity"),
        ("cross_vertical_agreement", "Cross-vertical agreement"),
        ("grounding_accuracy", "Grounding accuracy"),
        ("grounding_precision", "Grounding precision"),
        ("grounding_recall", "Grounding recall"),
        ("next_topic_exact_match", "Next-topic exact-match"),
        ("prereq_violation_rate", "Prereq-violation rate"),
        ("weakest_hit_rate", f"Weakest({WEAKEST_N}) hit rate"),
        ("provenance_coverage", "Provenance coverage"),
        ("citation_precision", "Citation precision"),
        ("prompt_tokens", "Prompt tokens"),
        ("completion_tokens", "Completion tokens"),
    ]
    by_pair = {(s["system"], s["scenario"]): s for s in summaries}
    for scen in args.scenarios:
        lines += [f"## {scen}", ""]
        header = "| metric | " + " | ".join(args.systems) + " |"
        lines += [header, "|" + "---|" * (len(args.systems) + 1)]
        for key, label in metric_rows:
            cells = [_fmt(by_pair.get((sys, scen), {}).get(key)) for sys in args.systems]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines += [""]
        # latency detail per system
        for sys in args.systems:
            s = by_pair.get((sys, scen))
            if not s:
                continue
            lat = s.get("latency") or {}
            if lat:
                lines.append(f"Latency {sys} (seconds):")
                lines.append("")
                lines.append("| op | count | p50 | p95 | max |")
                lines.append("|---|---|---|---|---|")
                for op, st in sorted(lat.items()):
                    lines.append(
                        f"| {op} | {int(st['count'])} | {st['p50']:.3f} | "
                        f"{st['p95']:.3f} | {st['max']:.3f} |"
                    )
                lines.append("")
    if failures:
        lines += ["## Leg failures (loud, not zero-filled)", ""]
        for f in failures:
            lines += [f"### {f['system']} × {f['scenario']}", "", "```", f["error"][:2000], "```", ""]
    lines += [
        "## Notes",
        "",
        "- Every number above comes from a real run; see raw.jsonl for every probe.",
        "- Baseline legs (mem0, memos) store evidence as natural-language narratives",
        "  with native extraction and are probed through a strict real-LLM judge.",
        "  Baseline quality jointly reflects the memory system's extraction, its",
        "  retrieval, and the judge model — the harness does not isolate the three.",
        "- Cost is reported only when EDUPAAL_EVALS_LLM_PRICE_PROMPT_PER_1M and",
        "  EDUPAAL_EVALS_LLM_PRICE_COMPLETION_PER_1M are both set.",
        "",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="EduPAAL-Evals runner")
    ap.add_argument("--systems", default="edupaal",
                    help="comma-separated: edupaal,mem0,memos")
    ap.add_argument("--scenarios", default=",".join(ALL_SCENARIOS),
                    help="comma-separated: S1..S4 (S5 builder exists but is excluded from the default suite)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    args.systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    args.scenarios = [s.strip().upper() for s in args.scenarios.split(",") if s.strip()]

    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", stamp)
    os.makedirs(outdir, exist_ok=True)

    out = {"probe_records": []}
    summaries, failures = [], []
    for sys_name in args.systems:
        for scen_id in args.scenarios:
            spec = get_scenario(scen_id, args.seed)
            try:
                summaries.append(run_pair(sys_name, spec, out))
                print(f"ok   {sys_name} x {scen_id}", flush=True)
            except Exception as exc:  # fail loud, keep other legs' results
                err = f"{type(exc).__name__}: {exc}"
                failures.append({"system": sys_name, "scenario": scen_id, "error": err})
                print(f"FAIL {sys_name} x {scen_id}: {err}", flush=True)
                traceback.print_exc()

    with open(os.path.join(outdir, "raw.jsonl"), "w") as fh:
        for rec in out["probe_records"]:
            fh.write(json.dumps(rec) + "\n")
        for s in summaries:
            fh.write(json.dumps({"_summary": s}) + "\n")
        for f in failures:
            fh.write(json.dumps({"_failure": f}) + "\n")
    write_report(os.path.join(outdir, "report.md"), summaries, failures, args)
    print(f"results -> {outdir}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
