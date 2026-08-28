# Nightly report workflow

[nightly-report.yml](nightly-report.yml) — the epic's no-human-intervention bullet.
Inert here; it runs once copied to `.github/workflows/` on a repo with self-hosted
TT runners labelled `wh` and `bh`.

## One night

```mermaid
flowchart LR
    CRON["cron 02:00<br/>or manual dispatch"] --> WH
    subgraph serial["one job at a time — both push the same branch"]
        WH["measure on wh runner"] --> BH["measure on bh runner"]
    end
    BH --> DONE["report updated,<br/>no human involved"]
```

## One job

```mermaid
flowchart TD
    A["checkout"] --> B["rebuild tt-metal at main"]
    B --> C["discover · derive · probe<br/>manifest refreshed, per-arch layouts"]
    C --> D["measure: 6 categories × both dtypes"]
    D --> E["charts + report"]
    E --> F["commit reports/ report_index.json stats/<br/>pull --rebase, push"]
```

| Property | Choice |
|---|---|
| trigger | nightly cron + `workflow_dispatch` |
| runners | `[self-hosted, wh]` / `[self-hosted, bh]`, matrix `max-parallel: 1` |
| tt-metal version | whatever `main` is that night — recorded per run in `stats/runs/` |
| gate | none — regenerate and publish; `compare` exists as a manual tool |
| runner provides | `TT_METAL_HOME` (built checkout), `PYTHON_ENV` |
| timeout | 12 h per arch |

## After it lands

```mermaid
flowchart LR
    N["nightly-report"] -->|report_index.json| A["analyze-report/analyze.yml<br/>(draft) — LLM verdicts + digest"]
    N -->|reports/**| G["GitHub-browsable pages"]
```
