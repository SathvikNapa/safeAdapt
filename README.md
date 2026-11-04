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
│   ├── llm_ollama.py       real open-weight models via local/remote Ollama — free, not default
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

# real open-weight models via Ollama (free, but needs a reachable `ollama serve`)
uv run python -m safeadapt.entrypoints.run_study1_cli --provider ollama \
    --ollama-model qwen2.5:7b-instruct --ollama-base-url http://127.0.0.1:11435 \
    --n-per-condition 45 --n-masking 30 --concurrency 4
```

### Open-weight (Ollama) runs

The `ollama` provider uses `safeadapt/adapters/llm_ollama.py` for both roles:
`OllamaRespondentSimulator` (temperature 0.8, in-character replies) and
`OllamaToMEstimator` — the latter is a genuine structured-reasoning LLM call,
not a keyword lookup: it never sees the respondent's ground-truth profile,
reasons step-by-step via a schema-enforced `reasoning` field (Ollama
structured outputs), and `project()` does real prospective reasoning about a
not-yet-asked question rather than reusing the current estimate plus a fixed
bump.

This project's open-weight runs use a GPU node on the CLSP SLURM cluster:

```bash
# one-time: install + sbatch job (deploy/clsp_ollama.sbatch), see that file
ssh login.clsp.jhu.edu squeue -u <you>          # find the running node, e.g. c26
ssh -L 11435:c26:11434 login.clsp.jhu.edu        # tunnel: local 11435 -> node's 11434
# then point --ollama-base-url at http://127.0.0.1:11435
```

**Known packaging bug**: Ollama **v0.40.14**'s `ollama-linux-amd64.tar.zst`
is missing `lib/ollama/libggml-base.so.0` (and `llama-server`), so it
silently falls back to CPU-only inference even with a GPU present and a
correct `--gres=gpu:1` allocation (`nvidia-smi` shows the GPU; the ollama log
shows `inference compute id=cpu ... ` with no `id=cuda` line). **v0.40.13**'s
tarball is complete — that's what `deploy/clsp_ollama.sbatch` installs.
Re-check with `tar -tf` before upgrading past this.

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
  Study 1's actual claims require the `openai` or `ollama` adapter.
- The `ollama` provider's respondent simulator does not thread prior
  conversation turns into its prompt (matches `OpenAIRespondentSimulator`'s
  existing single-turn-per-call convention — see `respond()`'s docstring in
  `llm_ollama.py`); only the ToM estimator sees the full transcript.
