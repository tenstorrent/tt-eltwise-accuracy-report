# LLM workflows

The pipeline produces the data; LLMs read and maintain it. Every judgment is predefined
in a committed artifact, so a model retrieves decisions — it never invents them.

## Where coverage and data are defined

| Question | Answered by |
|---|---|
| what exists, what is measurable, why not | `stats/ops_manifest.json` — `ops`, `unprobeable`, `rejected`, `EXCLUDED` reasons |
| at which parameters, against which golden | `src/ttnn_accuracy/ops/overrides.py` — the only hand-edited surface |
| what was measured, on what silicon | `data/{arch}/{dtype}/{op}/{variant}.csv` + `stats/runs/{id}.json` |
| the numbers an answer cites | `report_index.json` — per arch/dtype/op/variant, with provenance |
| the human view | `reports/**` pages and charts |
| the reader's definitions | [contract.md](contract.md) — metrics, outcomes, verdict rules, answering rules |

## 1 — Answer a customer

```mermaid
sequenceDiagram
    participant C as customer
    participant L as LLM
    participant I as report_index.json

    C->>L: is exp usable on WH bf16?
    L->>I: wh/bf16/exp/default + _runs commit
    I-->>L: max 1 ULP, usable_to 3.39e38, specials, a5cd86212e1
    L-->>C: cited answer — fast_approx variant listed beside it
    Note over L: not in the index → say so, never extrapolate
```

Built, so answers need no judgment: a `verdict` per index entry (five fixed phrases,
rules in `report/charts.py::verdict`), a `rationale` per override, and
[contract.md](contract.md) — the committed definition of every metric, threshold,
outcome label and answering rule.

## 2 — Cover an uncovered op

The recorded reason is the work order; `overrides.py` is the only file to write;
`probe` is the validator. Backlog: [uncovered.md](uncovered.md).

```mermaid
flowchart TD
    S["scan: unprobeable · rejected ·<br/>in ttnn but absent from manifest"] --> W{"reason?"}
    W -->|missing scalar kwarg| A["LLM drafts override:<br/>canonical value, ttnn + golden spellings<br/>from nanobind and golden source"]
    W -->|no golden attached| B["LLM drafts torch golden"]
    W -->|moves data, not math| X["EXCLUDED entry"]
    W -->|integer-only · complex| O["stays out, reason stands"]
    A & B --> V["discover + probe:<br/>classifies? layouts found?"]
    V -->|yes| M["measure → report"]
    V -->|no — new reason recorded| W
```

## 3 — Fix coverage when an op changes

`discover` diffs ttnn against the committed manifest on every run — the diff is the trigger.

```mermaid
flowchart TD
    D["discover diff:<br/>added · removed · changed"] -->|added| U["workflow 2"]
    D -->|changed| R{"still classifies<br/>and probes?"}
    R -->|yes| M["re-measure the op,<br/>regenerate its pages"]
    R -->|no| F["LLM revises its override<br/>against the new signature"] --> U
    D -->|removed| P["prune data + pages,<br/>manifest already prunes itself"]
```
