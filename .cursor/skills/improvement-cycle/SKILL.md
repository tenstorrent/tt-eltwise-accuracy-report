---
name: improvement-cycle
description: Runs the research-propose-branch-implement-review-improve-review-style-commit cycle for this repo. Use when improving ttnn-accuracy, changing metrics, adding CLI flags, editing reports, or when the user asks to work a roadmap item.
---

# Improvement cycle

Copy and track:

```
- [ ] research
- [ ] propose (stop for approval)
- [ ] branch
- [ ] implement
- [ ] review
- [ ] improve
- [ ] review
- [ ] style: CLAUDE.md, uvx ruff check ., uvx ruff format ., pytest
- [ ] commit (only if the user asked)
```

## research

Read the files. No throwaway scripts to discover a file's contents. For a metric or verdict change, read `analyze-report/contract.md` and `src/ttnn_accuracy/report/score.py` together. For timing, read `measure/perf.py` and `config.py` (`SIDE`, `NOISE_PCT`). For coverage, read `analyze-report/uncovered.md` and `ops/overrides.py`.

## propose

Short. What, why, files you will edit, files you will not create. Call out any new module — wait.

Stop. Do not branch until the user approves.

## branch

From up-to-date `main`: `feat/`, `fix/`, `perf/`, `docs/` plus one slug. One concern.

## implement

Minimum that works. Stay inside existing files unless the proposal named a new one.

## review, improve, review

First pass: correctness vs contract, baselines, host-aware perf, sampled maxima as lower bounds. Second pass: after the fixes, review again. Do not treat the first review as the last.

## style

CLAUDE.md ladder. Then:

```bash
uvx ruff check .
uvx ruff format .
pytest
```

All three green, or do not offer a commit.

## commit

Only when the user asked to commit or to finish the cycle. Message: why, 1–2 sentences, matching recent `git log` style (`feat:`, `fix:`, `perf:`, `docs:`). Never `--no-verify`. Never amend a pushed commit.
