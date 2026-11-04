# SafeAdapt

Simulation harness for **Study 1** of *Safety-Constrained Adaptive
Interviewing with Persona Modulation* — validates, with 900+ LLM-simulated
respondents and no human subjects, whether a persona-aware safety constraint
reduces distress escalation relative to an unconstrained adaptive interviewer
and a static baseline, and whether safety behavior itself differs by
interviewer persona (Researcher vs. Peer).

The paper this implements lives alongside the production survey platform at
`SurveyAI/research/Safety-Constrained Adaptive Interviewing for Mental
Health and Climate Distress Research v4.docx`. This repo is the experiment
code and results, kept separate from that production codebase.

## Architecture (hexagonal / ports & adapters)

```
safeadapt/
├── domain/          entities + pure logic — no I/O, no framework imports
│   ├── models.py          Persona, SystemCondition, AblationArm, DistressProfile,
│   │                      MentalState, Probe, ScoredProbe, Turn, SessionResult
│   ├── personas.py        the interviewer's relational-frame system prompts
│   ├── safety_policy.py   persona-aware threshold τ_p, fire/no-fire decision
│   ├── utility.py         U(a_t) = α·Coverage + β·Emergence, s.t. A'(a_t) ≤ τ_p
│   └── instruments.py     vendored PHQ-9 / climate-distress item text
├── ports/           Protocols the application core depends on
│   ├── llm_ports.py        RespondentSimulator, ToMEstimator
│   └── results_port.py     ResultsSink
├── application/     use cases — depend on ports + domain, not on adapters
│   ├── topic_guide.py      builds fixed-core + adaptive-probe + grounding probes
│   ├── interviewer.py      SafeAdaptInterviewer: one turn's decide-and-ask logic
│   ├── run_session.py      drives one full session to completion
│   └── study1_design.py    builds the full Study 1 condition matrix (§4)
├── adapters/        concrete port implementations — all I/O lives here
│   ├── llm_mock.py         deterministic, free, default
│   ├── llm_openai.py       real OpenAI calls — costs money, not default
│   └── results_jsonl.py    JSONL writer
└── entrypoints/     driving adapters (composition roots)
    └── run_study1_cli.py   wires adapters → application, runs, summarizes
```

Dependency direction is inward: `entrypoints` → `adapters`/`application` →
`ports`/`domain`. `domain` never imports `ports`, `application`, or
`adapters`. `application` imports `ports` (abstractions) and `domain`, never
a concrete adapter — which adapter gets used is decided only in
`entrypoints`, at the edge.

## Running

```bash
# zero-cost, deterministic — the default. Full Study-1-scale run (~1100 sessions).
uv run python -m safeadapt.entrypoints.run_study1_cli

# smaller smoke run
uv run python -m safeadapt.entrypoints.run_study1_cli --n-per-condition 10 --n-masking 10

# real OpenAI calls (needs OPENAI_API_KEY; not free) — confirm scale before running at full N
uv run python -m safeadapt.entrypoints.run_study1_cli --provider openai --n-per-condition 10
```

Results land in `results/study1/<--out, default sessions.jsonl>` — one JSON
record per session — plus an RQ1-RQ5-aligned summary printed to stdout.
`results/study1/mock_dry_run.jsonl` is a committed example from a full-scale
(1,100-session) `mock` run — a pipeline validation artifact, not real
experimental data (see Caveats).

## Tests

```bash
uv run pytest tests/
```

## Caveats

- `domain/instruments.py`'s climate-distress items are **not** a verbatim
  reproduction of Latkin et al. (2022)'s published, validated scale — swap in
  the licensed item text before treating results as anything beyond pipeline
  validation. The PHQ-9 items are public domain (Pfizer) and used verbatim.
- The `mock` provider's respondent/ToM dynamics are a simplified, internally
  consistent simulation for exercising the full pipeline (persona effects,
  safety firing/recovery, ToM S-score differentiation, masking detection) at
  zero cost — they are not a substitute for real LLM-simulated respondents.
  Study 1's actual claims require the `openai` (or an `anthropic`) adapter.
