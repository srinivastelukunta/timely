# timely

Simulation and evaluation code for studying whether a human can still
correct delegated agent work **in time**: not only whether a corrective
action remains possible, but whether it remains possible for long enough
to be used.

This repository contains code only. Result files and figures are
produced by running it.

## Layout

```
sim/
  common/                 shared symbolic workflow, admissibility filters, witness cache
  study1_reliability.py   layered protection under a widening capability gap
  study2_window_collapse.py   single agent: feasibility filter vs timely filter
  study3_concurrent.py    two or three agents, stale views, reservation
  study4_llm.py           local language models (Ollama) self-assessing their margin
  study4/                 the target application, episode loop and LLM client for Study 4
tests/                    pytest suite
scripts/                  print compact tables from the result files
```

## Setup

Python 3.13. Studies 1 to 3 are CPU only.

```
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
pytest
```

## Run

Run from the repository root. Every script sets an explicit seed
(20260919) and writes `results/studyN.json` and its figures under
`figures/`.

```
python sim/study1_reliability.py
python sim/study2_window_collapse.py      # about 25 s, 2000 trials per cell
python scripts/show_study2.py
python sim/study3_concurrent.py           # about 30 s on 8 workers
python scripts/show_study3.py
```

Studies 1 to 3 reproduce exactly, apart from measured wall times.
Study 3 reuses the Study 2 trials, so its single-agent case is a paired
reference.

### Study 4 (GPU)

Needs [Ollama](https://ollama.com) and a GPU with about 8 GB of memory.
No extra Python dependencies.

```
ollama pull qwen2.5-coder:1.5b
ollama pull qwen2.5-coder:3b
ollama pull qwen2.5-coder:7b
python sim/study4_llm.py run --only 1.5b      # one tier at a time keeps one model in memory
python sim/study4_llm.py run --only :3b
python sim/study4_llm.py run --only :7b
python sim/study4_llm.py analyze
python scripts/show_study4.py
```

The run is resumable. GPU inference is not bit-reproducible, so every
prompt and reply is kept locally under `results/study4_raw/`.

The target application is a small real Python project created on disk,
with executable checks. The model never edits it: it proposes a
structured action and the execution layer applies it. External
operations such as deployment are simulated as log entries.

If Ollama runs on the CPU, check that only the installed Ollama service
is listening on port 11434. A second `ollama serve` started from a shell
can hold the port without GPU support.

## Conventions

- Task performance and revision success are reported separately and are
  never combined into one score. An agent that preserves every window by
  refusing to work scores zero on the task, and the metrics show it.
- The feasibility filter is a strong baseline: it is the timely filter
  with a threshold of zero and the same machinery.
- Conditions are compared on paired trajectories: same initial state and
  seed, different request times.
- In Study 4, capability (model tier, context length, samples) and
  authority (permissions, execution speed, mediation) never share a
  knob, and the deadline is a simulated step clock.
