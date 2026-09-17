# Technical overview

## What is measured

```
ulp_error = |y_ref − y| / ULP(y_ref)
```

| | Source | Precision |
|---|---|---|
| `x` | a representable value of the target dtype | bf16, fp32 |
| `y_ref` | PyTorch on CPU | float64 |
| `y` | Tenstorrent device | bf16, fp32 |

| Choice | Reason |
|---|---|
| ULP, not absolute error | being 0.001 wrong about `exp(-5)` is catastrophic; about `exp(80)` it is exact |
| golden in fp64 | the reference can never be the error source |
| ULP imported from tt-metal | the whole org measures one thing |
| bf16 exhaustive, fp32 in 1,024 blocks | 2¹⁶ fits; 2³² fits if streamed |
| subnormals flushed on both sides | the hardware flushes them |

## Hardware path

```mermaid
flowchart LR
    subgraph host["Host, CPU"]
        A["all bf16 codes"] --> B["mask domain,<br/>drop subnormals"]
        B --> C["reshape N × 128"]
        C --> G["golden<br/>float64"]
    end

    subgraph ttnn["ttnn"]
        C --> D["from_torch<br/>TILE_LAYOUT"]
        D --> E["ttnn.exp"]
        E --> F["to_torch"]
    end

    subgraph dev["Wormhole B0, 8 × 9 Tensix"]
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

Tiles are 32 × 32, which is why the reshape is 128 columns wide. WH and BH run different
kernels for the same op, so architecture is an axis.

## Pipeline

```mermaid
flowchart TD
    T["ttnn runtime<br/>~500 registered ops"] --> D1["discover<br/>classify, keep eltwise"]
    D1 --> M1[("ops_manifest.json<br/>221 eltwise, 64 refusals")]
    M1 --> D2["derive"]
    D2 -->|fp64 sweep| M1
    M1 --> D3["probe"]
    D3 -->|on device, per arch| M1

    OVR["overrides.py<br/>parameters, supplied goldens, exclusions"] --> R1
    M1 --> R1

    R1["measure"] -->|on device| CSV[("data/**.csv")]
    R1 --> PROV[("runs/{id}.json")]

    M1 --> R2["perf"]
    R2 -->|on device, one resident tensor| PRF[("stats/perf/{arch}.json")]

    CSV --> C1["charts"] --> SVG[("**.svg")]
    PRF --> C1
    C1 --> IDX[("report_index.json")]

    SVG --> P1["report"]
    IDX --> P1
    P1 --> MD[("markdown pages")]

    style R1 fill:#0b6e7a,color:#fff
```

Only `probe`, `measure` and `perf` need silicon. Everything else runs from the goldens on
CPU. Accuracy is deterministic and comparable anywhere. A timing belongs to its host, so
`perf` records which host, and `compare` never scores the field.

## Modules

```mermaid
flowchart TB
    CLI["cli.py"]

    subgraph ops["ops/ · what"]
        INT["introspect"] --> ARI["arity"]
        MAN["manifest"]
        PLN["plan"]
        OVR["overrides"]
    end

    subgraph dom["domain/ · where valid"]
        DER["derive"]
    end

    subgraph meas["measure/ · run"]
        SCH["schema"]
        MET["metrics"]
        SWP["sweeps"]
        DEV["device"]
        STO["store"]
        RUN["runner"]
        PRF["perf"]
    end

    subgraph rep["report/ · publish"]
        SCO["score"]
        CHT["charts"]
        TRE["tree"]
        PAG["pages"]
        NAV["navigation"]
        ASK["ask"]
        IDX["index"]
        CMP["compare"]
    end

    CFG["config"]
    CLI --> MAN & RUN & PRF & CHT & TRE & CMP
    MAN --> ARI & DER & OVR
    RUN --> DEV & SWP & STO & SCH & PRF & SCO
    PRF --> DEV & SCH & PLN
    STO --> SCH
    CHT --> SCO
    SCO --> SCH & PLN & OVR
    TRE --> PAG & NAV & ASK & IDX
    PAG & NAV & ASK --> IDX
    IDX --> PLN
    PLN --> MAN & OVR
    CFG --> MET & SWP & PRF & DER & SCO & CHT & PAG
    TTM["tt-metal ulp"] --> MET
    SWP --> MET
```

`schema.py` is the only definition of a result row. `metrics.py` is pure numerics and
carries the tests. `ttnn` is imported inside the functions that use it, so every module
here imports on a machine with no device.

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
        R->>R: write CSV, or count the failure
    end
    R-->>R: exit code = failures
```

## Configuration

| Flag | Default | Effect |
|---|---|---|
| `--arch` | required | `wh` or `bh`, **verified against the device** |
| `--ops`, `--category` | `all` | which ops |
| `--dtype` | `bf16` | `bf16`, `fp32`, `both` |
| `--output-dir` | `data/` | redirect to a fast local disk |
| `--device-id` | `0` | multi-card hosts |

## Axes

| Axis | Values | Status |
|---|---|---|
| architecture | WH, BH | varied |
| dtype | bf16 exhaustive, fp32 in blocks | varied |
| op variant | `overrides.py`: `fast_approx` for `exp` and `gelu`, per-op scalars | varied |
| layout | every accepted layout recorded by `probe`; sweeps take tile | recorded, not varied |
| shape | one tile-aligned block, `TILE_WIDTH` columns | fixed |
| math fidelity | HiFi4 | fixed by the kernel |
| `fp32_dest_acc_en` | follows the dtype | not free |

160 of 215 wh ops accept row-major and none is measured in it. That is the largest
untested surface in the report.

The last two rows are not axes for eltwise, whatever they are for matmul.
`unary_program_factory.cpp` hardcodes `.math_fidelity = MathFidelity::HiFi4`, and
`unary.cpp` derives `fp32_dest_acc_en` from `preserve_fp32_precision = (input_dtype ==
FLOAT32)`. Varying the dtype is the only way either one moves, and no eltwise op takes a
`compute_kernel_config`.

## Constants

Every tunable number lives in [config.py](../src/ttnn_accuracy/config.py). Structure — the
outcome labels, the CSV columns, which sweep each arity takes — stays with its own code.

| Constant | Value | Meaning |
|---|---|---|
| `TILE_WIDTH` | 128 | columns per row, a multiple of the 32-wide tile |
| `FP32_BLOCK` | 2²² | fp32 values per round-trip, so 1,024 blocks |
| `FP32_GROUP` | 2¹⁶ | an fp32 CSV row is the worst of a group. bf16 rows are per input |
| `MIN_NORMAL` | 2⁻¹²⁶ | subnormal boundary |
| `ULP_CLIP` | 1000 | chart clamp, and the CDF's right edge. Clipped points are counted |
| `USABLE_ULP` | 2 | what "accurate to \|x\| ≤ B" means |
| `ULP_LINES` | 1, 3, 10, 100 | chart reference lines, drawn once the data reaches the one before |
| `N_BINS`, `MIN_BIN` | 32, 8 | per-bin percentile panel; a bin under 8 points is hidden, not believed |
| `CDF_POINTS` | 200 | log grid for the CDF, so an fp32 sweep's 65k distinct ULPs draw as 200 |
| `SPECIAL_VALUES` | ±0, ±inf, NaN, ±min | measured per op, no ULP, printed device vs golden |
| `ELEMENTS` | 2²⁴ | one timed dispatch. At 2²⁰ host jitter moved it 70% |
| `NOISE_PCT` | 5 | `us_min` moved 3.1% between two runs of one build |

## Artifacts

| Path | Committed | From | Contents |
|---|---|---|---|
| `stats/ops_manifest.json` | yes | discover, derive, probe | classification with reasons, derived domains, per-arch layouts. No timestamps, so a diff means ttnn changed |
| `ops/overrides.py` | yes | hand-edited | every editorial decision: scalars, variants, supplied goldens, exclusions |
| `stats/runs/{id}.json` | yes | measure | tt-metal commit, versions, device |
| `stats/perf/{arch}.json` | yes | perf | µs per variant with its host. The full row, of which a page shows two figures |
| `data/**.csv` | no | measure | per-input rows. Symlink to a local disk |
| `report_index.json` | yes | charts | stats, verdict and rationale per arch/dtype/op/variant. What an assistant consumes, under [contract.md](../analyze-report/contract.md) |
| `reports/**` | yes | charts, report | SVGs and markdown |
