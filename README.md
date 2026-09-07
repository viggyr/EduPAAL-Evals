# EduPAAL-Evals

Adversarial evaluation harness comparing **EduPAAL's structured educational
memory** against general agent-memory infrastructure used natively:

| Leg | What it is | How evidence is stored |
|-----|-----------|------------------------|
| `edupaal` | EduPAAL skill (SQLite, no LLM) | Structured `Evidence` records with authoritative mastery semantics |
| `mem0` | Native [Mem0](https://github.com/mem0ai/mem0) (`infer=True`) | Natural-language narratives; Mem0's own extraction |
| `memos` | Native [MemOS](https://github.com/MemTensor/MemOS) (`MOS.add`) | Natural-language narratives; MemOS's own extraction pipeline |

The baselines are probed through a **strict real-LLM judge** that reads only
the memories each system *retrieved* and returns strict-schema JSON
(mastery per topic, next topic, weakest topics, cited evidence IDs). No
heuristic fallback, no mock scores: if the judge is unconfigured or returns
invalid JSON after retries, the run fails loudly.

> **No fabricated numbers.** Every score in `results/` comes from a real run.
> Blocked or skipped legs are reported with their actual error, never as
> zeros.

## Requirements

- Python ≥ 3.10
- EduPAAL (pinned to a validated commit in `pyproject.toml`)
- For the baseline legs: `pip install -e ".[baselines]"` (adds `mem0ai`,
  `MemoryOS`, `openai`)
- For the baseline legs: a free **Google Gemini API key** (see below).
  The EduPAAL-only leg needs no key and no baseline packages.

## Get a free Gemini API key (baseline legs only)

1. Go to **Google AI Studio**: https://aistudio.google.com
2. Sign in with a Google account.
3. Click **"Get API key"** (or go directly to
   https://aistudio.google.com/apikey).
4. Click **"Create API key"** and copy it.
5. No credit card is required for the free tier.

Store the key in the Secure Vault (never commit it, never paste it into
chat or logs). The harness reads it from the environment at run time:

```bash
export EDUPAAL_EVALS_LLM_API_KEY="<key from vault>"
```

## Configuration

Only **one** variable is required. The rest have working defaults.

| Variable | Required | Default | Purpose |
|----------|----------|---------|---------|
| `EDUPAAL_EVALS_LLM_API_KEY` | **yes** (baselines) | — | Gemini API key. Missing → the baseline legs raise `RuntimeError` immediately; nothing is mocked. |
| `EDUPAAL_EVALS_LLM_MODEL` | no | `gemini-3.8-flash` | Judge chat model. Verified in Google's official OpenAI-compatibility docs (https://ai.google.dev/gemini-api/docs/openai, checked 2026-09-07). Override with any current model id from those docs. |
| `EDUPAAL_EVALS_LLM_BASE_URL` | no | `https://generativelanguage.googleapis.com/v1beta/openai/` | OpenAI-compatible endpoint (verified in the same docs). Override for an alternative provider (see below). |
| `EDUPAAL_EVALS_LLM_PRICE_PROMPT_PER_1M` | no | — | USD per 1M prompt tokens. Cost is reported **only** when this and the next are both set; otherwise `n/a`. Free tiers cost $0, so usually unnecessary. |
| `EDUPAAL_EVALS_LLM_PRICE_COMPLETION_PER_1M` | no | — | USD per 1M completion tokens. |
| `EDUPAAL_EVALS_MEMOS_EMBEDDER_MODEL` | no | `gemini-embedding-001` | Embedding model for the MemOS leg. Must be served by the base URL's OpenAI-compatible `/embeddings` endpoint, with 3072-dim output (MemOS's default `vector_dimension`). |

Free-tier notes:

- Gemini's free tier is rate-limited (per-day request caps vary by model).
  If a run hits HTTP 429, wait for the quota window or set
  `EDUPAAL_EVALS_LLM_MODEL` to a higher-quota model such as a
  `gemini-*-flash-lite` id from the docs.
- On Gemini 3 models, thinking tokens are billed as output tokens and are
  included in the reported `completion_tokens`.

### Alternative provider: Groq (documented, not default)

The judge speaks plain OpenAI protocol, so any OpenAI-compatible base URL
works. For Groq's free tier (key at https://console.groq.com):

```bash
export EDUPAAL_EVALS_LLM_BASE_URL="https://api.groq.com/openai/v1"
export EDUPAAL_EVALS_LLM_MODEL="llama-3.1-8b-instant"
export EDUPAAL_EVALS_LLM_API_KEY="<groq key>"
```

Limitation: Groq serves **no embeddings endpoint**, so the `memos` leg
cannot run against Groq — it fails loudly at ingest time. The `mem0` leg
works (it uses a local FastEmbed embedder; only its LLM calls go to the
base URL).

## Scenarios

Deterministic, seeded scenarios with documented latent truth, in
`scenarios/` (see `scenarios/README.md` for the fixed seeds and the
evidence→latent-mastery rubric):

- **S1 steady learner** — 4-topic chain, 8 sessions (dialogue, quiz, practice)
- **S2 struggling learner** — strong dialogue vs weak quiz on the same
  concept; characterizes disagreement handling
- **S3 prerequisites** — `A → B → C`; `next_topic` must never pick a topic
  whose prerequisites are unmet
- **S4 plateau** — evidence stops at Intermediate; Advanced predictions are
  false positives
- **S5 isolation** — two learners, interleaved evidence, no cross-learner
  leakage

Ground truth is generated independently of EduPAAL's outputs.

## Metrics

For each system × scenario: mastery exact-match rate, ordinal MAE
(unknown=0 … advanced=3), cross-vertical agreement after each source
vertical's evidence, grounding precision/recall, next-topic
prerequisite-violation rate, `weakest(n)` hit rate, provenance coverage
(mastery claims citing specific evidence IDs), p50/p95 operation latency,
LLM prompt/completion tokens, and estimated LLM cost (only when prices are
configured). Absent baseline values are never filled with zeros.

## Running

```bash
pip install -e .
# baseline legs:
pip install -e ".[baselines]"
export EDUPAAL_EVALS_LLM_API_KEY="<gemini key>"

# EduPAAL only (no key, no baseline packages needed):
python run.py --systems edupaal

# Full comparison:
python run.py --systems edupaal,mem0,memos
```

Each run writes `results/<timestamp>/raw.jsonl` (every probe, latency, and
token count) and `results/<timestamp>/report.md` (aggregated metrics).
Every baseline LLM call's token usage is logged; the report shows totals
per system and per scenario.

## Design notes

- The EduPAAL leg uses the real `EduPAALSkill` on SQLite with **no LLM**.
- The Mem0 leg uses native `Memory.add(..., infer=True)` per evidence
  narrative and `Memory.search` per probe — the opposite of EduPAAL's
  exact-storage Mem0 provider.
- The MemOS leg uses top-level `MOS.add(...)` / `MOS.search(...)` with an
  explicitly configured chat model (MemOS's `MOS.simple()` auto-config
  hardcodes `gpt-4o-mini`, which does not exist on the Gemini endpoint),
  Gemini embeddings, explicit `create_user` registration per learner, and
  hermetic local storage per process (MemOS reads MEMOS_BASE_PATH at import
  time, so one process = one base path; per-instance cube/user ids keep
  scenarios isolated).
- Baseline quality depends jointly on the native memory system's
  extraction, its retrieval, and the judge model — a weak baseline score
  does not isolate which of the three is at fault. The harness reports
  retrieved-memory counts alongside judgments so this can be inspected.

## License

MIT — see `LICENSE`.
