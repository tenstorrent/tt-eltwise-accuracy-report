# Technical overview

## What is measured

```
ulp_error = |y_ref − y| / ULP(y_ref)
```

| | Source | Precision |
|---|---|---|
| `x` | a representable value of the target dtype | bf16 / fp32 |
| `y_ref` | PyTorch on CPU | float64 |
| `y` | Tenstorrent device | bf16 / fp32 |

- ULP, not absolute error: 0.001 wrong about `exp(-5)` is catastrophic, about `exp(80)` exact
- golden is fp64 — the reference is never the error source
- ULP imported from tt-metal — the whole org measures one thing
- bf16 exhaustive (all 2¹⁶); fp32 all 2³² in 1,024 blocks
- subnormals flushed on both sides, because the hardware flushes them

## Hardware path

```mermaid
flowchart LR
    subgraph host["Host — CPU"]
        A["all bf16 codes"] --> B["mask domain<br/>drop subnormals"]
        B --> C["reshape N × 128"]
        C --> G["golden<br/>float64"]
    end

    subgraph ttnn["ttnn"]
        C --> D["from_torch<br/>TILE_LAYOUT"]
        D --> E["ttnn.exp"]
        E --> F["to_torch"]
    end

    subgraph dev["Wormhole B0 — 8 × 9 Tensix"]
        D -.->|PCIe| H["tiles in DRAM"]
        H --> I["reader → CB"]
        I --> J["SFPU compute"]
        J --> K["writer → DRAM"]
        K -.->|PCIe| F
    end

    G --> L["compare"]
    F --> L
    L --> M[("CSV")]
```

32 × 32 tiles — hence the 128-column reshape. WH and BH run different kernels for the same
op, which is why architecture is an axis.

## Pipeline

```mermaid
flowchart TD
    T["ttnn runtime<br/>~500 registered ops"] --> D1["discover<br/>classify, keep eltwise"]
    D1 --> M1[("ops_manifest.json<br/>212 eltwise + refusal reasons")]
    M1 --> D2["derive"]
    D2 -->|fp64 sweep| M1
    M1 --> D3["probe"]
    D3 -->|on device, per arch| M1

    OVR["overrides.py<br/>parameters · supplied goldens · exclusions"] --> R1
    M1 --> R1

    R1["measure"] -->|on device| CSV[("data/**.csv")]
    R1 --> PROV[("runs/{id}.json")]

    CSV --> C1["charts"] --> SVG[("**.svg")]
    C1 --> IDX[("report_index.json")]

    SVG --> P1["report"]
    IDX --> P1
    P1 --> MD[("markdown pages")]

    style R1 fill:#0b6e7a,color:#fff
```

Only `probe` and `measure` need silicon; everything else runs from the goldens on CPU.

## Modules

```mermaid
flowchart TB
    CLI["cli.py"]

    subgraph ops["ops/ — what"]
        INT["introspect"] --> ARI["arity"]
        MAN["manifest"]
        OVR["overrides"]
    end

    subgraph dom["domain/ — where valid"]
        DER["derive"]
    end

    subgraph meas["measure/ — run"]
        SCH["schema"]
        MET["metrics"]
        SWP["sweeps"]
        DEV["device"]
        STO["store"]
        RUN["runner"]
    end

    subgraph rep["report/ — publish"]
        CHT["charts"]
        PAG["pages"]
        CMP["compare"]
    end

    CLI --> MAN & RUN & CHT & PAG & CMP
    MAN --> ARI & DER & OVR
    RUN --> DEV & SWP & STO & SCH
    SWP --> MET
    STO --> SCH
    DER --> MET
    CHT --> MET
    PAG --> OVR
    TTM["tt-metal ulp"] --> MET
```

`schema.py` is the only definition of a result row. `metrics.py` is pure numerics and
carries the tests.

## One run

```mermaid
sequenceDiagram
    participant R as runner
    participant S as sweeps
    participant G as torch fp64
    participant H as device
    participant M as metrics

    R->>H: open, check_arch, write provenance
    loop op × dtype × variant
        R->>S: sweep(lo, hi)
        S->>G: golden(x.float64)
        S->>H: from_torch → op → to_torch
        S->>M: compare(x, y_ref, y)
        M-->>S: DataFrame
        S-->>R: rows, or None
        R->>R: write CSV, or count failure
    end
    R-->>R: exit code = failures
```

## Configuration

| Flag | Default | Effect |
|---|---|---|
| `--arch` | required | `wh`/`bh`, **verified against the device** |
| `--ops` / `--category` | `all` | which ops |
| `--dtype` | `bf16` | `bf16`, `fp32`, `both` |
| `--output-dir` | `data/` | redirect to fast local disk |
| `--device-id` | `0` | multi-card hosts |

| Axis | Values | Status |
|---|---|---|
| Architecture | WH, BH | active |
| Data type | bf16 exhaustive, fp32 in blocks | active |
| Op variant | `overrides.py`: `exp`/`gelu` `fast_approx`, per-op scalars | active |
| Layout | per op, from `probe` | active |
| Math fidelity | LoFi … HiFi4 | reserved, not varied |
| `fp32_dest_acc_en` | on/off | reserved, not varied |

Reserved axes are null columns in every run record — free now, unbackfillable later.

| Constant | Value | Meaning |
|---|---|---|
| `TILE_WIDTH` | 128 | columns per row, multiple of the 32-wide tile |
| `FP32_BLOCK` | 2²² | fp32 values per round-trip → 1,024 blocks |
| `FP32_GROUP` | 2¹⁶ | fp32 CSV row = worst of a group; bf16 rows are per input |
| `MIN_NORMAL` | 2⁻¹²⁶ | subnormal boundary |
| `ULP_CLIP` | 1000 | chart clamp; clipped points counted |
| `USABLE_ULP` | 2 | what "accurate to \|x\| ≤ B" means |
| `SPECIAL_VALUES` | ±0, ±inf, NaN, ±min | measured per op, no ULP, printed device-vs-golden |

## Artifacts

| Path | Committed | From | Contents |
|---|---|---|---|
| `stats/ops_manifest.json` | yes | discover, derive, probe | classification with reasons, derived domains, per-arch layouts. No timestamps — a diff means ttnn changed |
| `ops/overrides.py` | yes | hand-edited | every editorial decision: scalars, variants, supplied goldens, exclusions |
| `stats/runs/{id}.json` | yes | measure | tt-metal commit, versions, device |
| `data/**.csv` | no | measure | per-input rows; symlink to local disk |
| `report_index.json` | yes | charts | summary stats per arch/dtype/op/variant — what an LLM consumes |
| `reports/**` | yes | charts, report | SVGs and markdown |
