"""The LLM judge used by the baseline legs (native Mem0 / MemOS).

The baselines store evidence as natural-language narratives in general memory
infrastructure, exactly as a vanilla agent would. The judge reads the
*retrieved* memories and answers the same probes the EduPAAL leg answers
natively, returning a STRICT-SCHEMA JSON object. There is deliberately no
heuristic fallback: if the environment is not configured, or the model does
not return valid JSON after retries, the judge raises instead of inventing a
score. Fabricated evaluations are worse than no evaluations.

Default provider: Google Gemini free tier via its OpenAI-compatible endpoint.
Get a free key at https://aistudio.google.com (sign in -> "Get API key").
No credit card required for the free tier.

Environment::

    EDUPAAL_EVALS_LLM_API_KEY   your Gemini API key (required; fail-loud.
                                The user keeps this in the Secure Vault; the
                                harness reads it from the environment at run
                                time and never persists it.)
    EDUPAAL_EVALS_LLM_MODEL     default: "gemini-3.8-flash" (verified in
                                Google's OpenAI-compatibility docs,
                                https://ai.google.dev/gemini-api/docs/openai,
                                2026-09-07). Override with any current model id;
                                verify against the model list in those docs.
    EDUPAAL_EVALS_LLM_BASE_URL  default:
                                "https://generativelanguage.googleapis.com/v1beta/openai/"
                                Override for an alternative OpenAI-compatible
                                provider, e.g. Groq's free tier:
                                EDUPAAL_EVALS_LLM_BASE_URL=https://api.groq.com/openai/v1
                                EDUPAAL_EVALS_LLM_MODEL=llama-3.1-8b-instant
                                (Groq key at https://console.groq.com). Note:
                                Groq serves no embeddings endpoint, so the
                                MemOS baseline leg requires the Gemini default
                                (or another base URL that serves embeddings).

Free-tier notes: Gemini's free tier is rate-limited (requests/day caps that
vary by model; flash-lite models allow more). If a run hits 429, either wait
for the quota window or set EDUPAAL_EVALS_LLM_MODEL to a higher-quota model
such as a gemini-*-flash-lite id from the docs. On Gemini 3 models, thinking
tokens are billed as output tokens and appear inside completion_tokens.

Token accounting: every judge call records prompt/completion tokens from the
provider's usage block. Cost is reported only when both price variables are
set; the free tiers cost $0, so these are usually unnecessary.

Optional::

    EDUPAAL_EVALS_LLM_PRICE_PROMPT_PER_1M      USD per 1M prompt tokens
    EDUPAAL_EVALS_LLM_PRICE_COMPLETION_PER_1M  USD per 1M completion tokens
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import LEVELS

ENV_MODEL = "EDUPAAL_EVALS_LLM_MODEL"
ENV_KEY = "EDUPAAL_EVALS_LLM_API_KEY"
ENV_BASE_URL = "EDUPAAL_EVALS_LLM_BASE_URL"
ENV_PRICE_PROMPT = "EDUPAAL_EVALS_LLM_PRICE_PROMPT_PER_1M"
ENV_PRICE_COMPLETION = "EDUPAAL_EVALS_LLM_PRICE_COMPLETION_PER_1M"

# Gemini free tier via its OpenAI-compatible endpoint. Model id verified in
# Google's official OpenAI-compatibility docs (ai.google.dev/gemini-api/docs/openai,
# checked 2026-09-07); re-verify there before depending on it long-term.
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_MODEL = "gemini-3.8-flash"

JUDGE_SCHEMA_HINT = """\
Respond with ONLY a JSON object with exactly these keys:
{
  "mastery": {"<topic_id>": "unknown|beginner|intermediate|advanced", ... (every topic listed exactly once)},
  "next_topic": "<topic_id>" or null,
  "weakest": ["<topic_id>", ...] (up to N ids, weakest first),
  "cited_evidence": {"<topic_id>": ["<evidence_id>", ...]} (evidence ids you actually saw in the memories above; empty list when a level is unknown or uncited)
}
No prose, no markdown fences, no extra keys."""


def _required_env() -> Dict[str, str]:
    """Only the API key is strictly required; model/base URL have Gemini defaults."""
    if not os.environ.get(ENV_KEY):
        raise RuntimeError(
            "LLM judge is not configured (fail-loud by design; no mock fallback). "
            f"Missing environment variable: {ENV_KEY}. Get a free Gemini key at "
            "https://aistudio.google.com (sign in -> Get API key), store it in "
            f"the Secure Vault, and export {ENV_KEY} from the vault at run time. "
            f"Optional overrides: {ENV_MODEL} (default {DEFAULT_MODEL!r}), "
            f"{ENV_BASE_URL} (default {DEFAULT_BASE_URL!r})."
        )
    return {
        ENV_MODEL: os.environ.get(ENV_MODEL, DEFAULT_MODEL),
        ENV_KEY: os.environ[ENV_KEY],
        ENV_BASE_URL: os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL),
    }


@dataclass
class Judgment:
    mastery: Dict[str, str]
    next_topic: Optional[str]
    weakest: List[str]
    cited_evidence: Dict[str, List[str]]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    memories_shown: int = 0  # informational: how many memories the judge saw


@dataclass
class LLMJudge:
    """Strict-schema judge over retrieved memories. Fails loudly, never mocks."""

    model: str = field(default="")
    api_key: str = field(default="")
    base_url: str = field(default="")
    max_retries: int = 3

    def __post_init__(self) -> None:
        env = _required_env()
        self.model = self.model or env[ENV_MODEL]
        self.api_key = self.api_key or env[ENV_KEY]
        self.base_url = self.base_url or env[ENV_BASE_URL]
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover - surfaced on use
            raise RuntimeError(
                "The 'openai' package is required for the LLM judge. Install the "
                "baselines extra: pip install 'edupaal-evals[baselines]'."
            ) from exc
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------ api

    def judge(
        self,
        *,
        learner_id: str,
        topics: List[Dict[str, str]],
        prereqs: Dict[str, List[str]],
        plan_order: List[str],
        memories: List[str],
        probe_label: str,
        weakest_n: int,
    ) -> Judgment:
        topic_lines = "\n".join(
            f"- {t['id']}: {t['name']} (prerequisites: {', '.join(prereqs.get(t['id'], [])) or 'none'})"
            for t in topics
        )
        mem_block = "\n".join(f"[{i+1}] {m}" for i, m in enumerate(memories)) or "(no memories retrieved)"
        prompt = f"""\
You are assessing a learner's mastery from their learning memories. Learner: {learner_id}.
Probe point: {probe_label}.

Topics (in plan order {plan_order}):
{topic_lines}

Retrieved memories (these are ALL you may use; cite their Evidence ids like ev-s1-003):
{mem_block}

Mastery levels: unknown (no accepted evidence yet), beginner, intermediate, advanced.
Rules: a topic's prerequisites must be mastered before it; next_topic is the first topic in plan order
that is not yet advanced and whose prerequisites are all advanced, or null when all are advanced.
weakest: the {weakest_n} least-mastered topic ids, weakest first.

{JUDGE_SCHEMA_HINT}"""
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            raw, usage = self._complete(prompt, attempt)
            try:
                return self._parse(raw, topics, plan_order, weakest_n, usage)
            except ValueError as exc:
                last_err = exc
        raise RuntimeError(
            f"LLM judge failed to return valid JSON after {self.max_retries} attempts "
            f"(probe {probe_label}). Last error: {last_err}. Refusing to fabricate a judgment."
        )

    # -------------------------------------------------------------- internals

    def _complete(self, prompt: str, attempt: int):
        if attempt:
            prompt += "\n\nYour previous response was not valid JSON matching the schema. Return ONLY the JSON object."
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        text = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        return text, {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        }

    def _parse(self, raw: str, topics, plan_order, weakest_n, usage) -> Judgment:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"not JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("top-level JSON is not an object")
        topic_ids = {t["id"] for t in topics}
        mastery = data.get("mastery")
        if not isinstance(mastery, dict) or set(mastery) != topic_ids:
            raise ValueError(f"mastery must map exactly these topics: {sorted(topic_ids)}")
        for tid, lvl in mastery.items():
            if lvl not in LEVELS:
                raise ValueError(f"bad level {lvl!r} for topic {tid}")
        nxt = data.get("next_topic")
        if nxt is not None and nxt not in topic_ids:
            raise ValueError(f"bad next_topic {nxt!r}")
        weakest = data.get("weakest")
        if not isinstance(weakest, list) or len(weakest) > weakest_n:
            raise ValueError(f"weakest must be a list of at most {weakest_n} topic ids")
        for tid in weakest:
            if tid not in topic_ids:
                raise ValueError(f"bad weakest id {tid!r}")
        cited = data.get("cited_evidence")
        if not isinstance(cited, dict):
            raise ValueError("cited_evidence must be an object")
        for tid, ids in cited.items():
            if tid not in topic_ids or not isinstance(ids, list):
                raise ValueError(f"bad cited_evidence entry for {tid!r}")
        return Judgment(
            mastery=dict(mastery),
            next_topic=nxt,
            weakest=list(weakest),
            cited_evidence={k: list(v) for k, v in cited.items()},
            prompt_tokens=usage["prompt_tokens"],
            completion_tokens=usage["completion_tokens"],
        )


def estimate_cost(prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    """USD estimate, or None when the user did not supply prices (no guessing)."""
    pp = os.environ.get(ENV_PRICE_PROMPT)
    pc = os.environ.get(ENV_PRICE_COMPLETION)
    if not pp or not pc:
        return None
    return (prompt_tokens / 1e6) * float(pp) + (completion_tokens / 1e6) * float(pc)
