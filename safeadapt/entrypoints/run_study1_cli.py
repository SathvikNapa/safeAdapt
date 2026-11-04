"""Composition root / driving adapter for Study 1 (§4). Wires adapters to the
application core based on --provider, runs the full condition matrix, writes
one JSONL record per session, and prints a metrics summary aligned to RQ1-RQ5.

    uv run python -m safeadapt.entrypoints.run_study1_cli
    uv run python -m safeadapt.entrypoints.run_study1_cli --provider openai --n-per-condition 10
    uv run python -m safeadapt.entrypoints.run_study1_cli --provider ollama \
        --ollama-model qwen2.5:7b-instruct --ollama-base-url http://127.0.0.1:11434 \
        --n-per-condition 45 --n-masking 30 --concurrency 4

`--provider openai`/`ollama` make real model calls — openai is billed
(needs OPENAI_API_KEY); ollama is free but needs a reachable `ollama serve`
(see README's "Open-weight (Ollama) runs" section for the CLSP setup used
for this project). The default `mock` provider is free and deterministic,
for pipeline validation rather than publishable results.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
from collections import defaultdict
from pathlib import Path

from safeadapt.adapters.results_jsonl import JsonlResultsSink, session_to_record
from safeadapt.application.run_session import run_session
from safeadapt.application.study1_design import build_study1_plan
from safeadapt.application.topic_guide import build_guide
from safeadapt.domain.models import SessionResult

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "study1"


def _build_adapters(args: argparse.Namespace):
    if args.provider == "mock":
        from safeadapt.adapters.llm_mock import MockRespondentSimulator, MockToMEstimator

        return MockRespondentSimulator(), MockToMEstimator()
    if args.provider == "openai":
        from safeadapt.adapters.llm_openai import OpenAIRespondentSimulator, OpenAIToMEstimator

        return OpenAIRespondentSimulator(), OpenAIToMEstimator()
    if args.provider == "ollama":
        from safeadapt.adapters.llm_ollama import OllamaRespondentSimulator, OllamaToMEstimator

        return (
            OllamaRespondentSimulator(
                model=args.ollama_model, base_url=args.ollama_base_url, timeout=args.ollama_timeout
            ),
            OllamaToMEstimator(
                model=args.ollama_tom_model or args.ollama_model,
                base_url=args.ollama_base_url,
                timeout=args.ollama_timeout,
            ),
        )
    raise ValueError(f"unknown provider '{args.provider}'")


async def _run_all(args: argparse.Namespace) -> list[SessionResult]:
    respondent, tom = _build_adapters(args)
    guide = build_guide(args.instrument)
    plan = build_study1_plan(n_per_condition=args.n_per_condition, n_masking=args.n_masking)

    sink = JsonlResultsSink(RESULTS_DIR / args.out)
    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[SessionResult] = []

    async def _one(spec):
        async with semaphore:
            result = await run_session(
                guide=guide,
                profile=spec.profile,
                persona=spec.persona,
                condition=spec.condition,
                ablation_arm=spec.ablation_arm,
                respondent=respondent,
                tom=tom,
            )
            sink.write(session_to_record(result))
            return result

    results = await asyncio.gather(*(_one(spec) for spec in plan))
    sink.close()
    return list(results)


def _summarize(results: list[SessionResult]) -> None:
    by_condition_persona = defaultdict(list)
    for r in results:
        by_condition_persona[(r.condition.value, r.persona.value if r.persona else "none")].append(r)

    print(f"\n{len(results)} sessions run.\n")
    print("-- RQ1/RQ2: escalation rate by condition x persona --")
    for key, group in sorted(by_condition_persona.items()):
        non_masking = [r for r in group if not r.profile.masking]
        if not non_masking:
            continue
        rate = sum(r.escalated for r in non_masking) / len(non_masking)
        print(f"  {key[0]:>16} / {key[1]:<10}  n={len(non_masking):<4}  escalation_rate={rate:.2f}")

    print("\n-- RQ3: safety firing + recovery rate by persona (SafeAdapt-full only) --")
    full = [r for r in results if r.condition.value == "safeadapt_full" and not r.profile.masking]
    by_persona = defaultdict(list)
    for r in full:
        by_persona[r.persona.value if r.persona else "none"].append(r)
    for persona, group in sorted(by_persona.items()):
        fired = [r for r in group if r.safety_fired_count > 0]
        firing_rate = len(fired) / len(group) if group else 0.0
        recovery_rate = sum(r.recovered for r in fired) / len(fired) if fired else float("nan")
        print(f"  {persona:<10} n={len(group):<4}  firing_rate={firing_rate:.2f}  recovery_rate={recovery_rate:.2f}")

    print("\n-- RQ4: ToM mean social-bias (S) score by persona --")
    for persona, group in sorted(by_persona.items()):
        s_scores = [r.mean_social_bias for r in group]
        print(f"  {persona:<10} mean_S={statistics.mean(s_scores):.2f}")

    print("\n-- RQ5: masking-profile escalation detection rate by persona --")
    masking = [r for r in results if r.profile.masking]
    by_persona_masking = defaultdict(list)
    for r in masking:
        by_persona_masking[r.persona.value if r.persona else "none"].append(r)
    for persona, group in sorted(by_persona_masking.items()):
        detected = sum(r.escalated for r in group) / len(group) if group else 0.0
        print(f"  {persona:<10} n={len(group):<4}  detected_rate={detected:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SafeAdapt Study 1 (simulated).")
    parser.add_argument("--provider", choices=["mock", "openai", "ollama"], default="mock")
    parser.add_argument("--instrument", choices=["phq9", "climate_distress"], default="phq9")
    parser.add_argument("--n-per-condition", type=int, default=180)
    parser.add_argument("--n-masking", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--out", default="sessions.jsonl")
    parser.add_argument("--ollama-model", default="qwen2.5:7b-instruct", help="model for both respondent + ToM")
    parser.add_argument("--ollama-tom-model", default=None, help="override: use a different model for ToM only")
    parser.add_argument("--ollama-base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-timeout", type=float, default=120.0)
    args = parser.parse_args()

    results = asyncio.run(_run_all(args))
    _summarize(results)
    print(f"\nWrote {len(results)} records to {RESULTS_DIR / args.out}")


if __name__ == "__main__":
    main()
