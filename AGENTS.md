# Agent entry

This repository measures TT-Metal eltwise accuracy and timing, publishes a GitHub report,
and diffs one run against another. People browsing `reports/**` and running
`measure` / `compare` / `check` are the audience. Do not add reward scalars, RL policies,
or kernel-generating JSON unless asked.

Style: [CLAUDE.md](CLAUDE.md). Roadmap: [docs/roadmap.md](docs/roadmap.md).
Contract twin of `score.py::verdict`: [analyze-report/contract.md](analyze-report/contract.md).

## Cycle

research → propose → wait for approval → new branch → implement → review → improve → review → `CLAUDE.md` + ruff + pytest → commit only if asked.

Skill: `.cursor/skills/improvement-cycle/SKILL.md`.

## Hard rules

- Cite arch/dtype/op/variant and the tt-metal commit. Never average across arch or dtype.
- A sampled `max_ulp` is a lower bound. A timing is not a regression across hosts.
- Never penalise `faithful`. That is tie-breaking (`NearestAway` vs even), not error.
- `compare` scores `rounded_frac` (higher is better) once both indexes have it. Do not
  subtract `max_ulp` from a pre-`rounded_frac` index against one that has it.
- `contract.md` and `score.py::verdict` move together. `ask.md` is generated; do not hand-edit.
- Propose a new file or package before creating it.
