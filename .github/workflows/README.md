# Workflows

| Workflow | Trigger | Device | Publishes |
|---|---|---|---|
| [nightly-report.yml](nightly-report.yml) | cron 02:37, dispatch | wh and bh | commits `reports/`, `report_index.json`, `stats/` |
| [analyze-report.yml](analyze-report.yml) | after a nightly | none | commits `analyze-report/findings.md`, opens an issue on a regression |
| [validate-kernel.yml](validate-kernel.yml) | dispatch | one arch | artifact only. Builds a commit you name, checks the ops it touches, fails on a finding |
| [perf-report.yml](perf-report.yml) | dispatch | one arch | artifact only. Timings for one commit, or the diff between two |
| [custom-report.yml](custom-report.yml) | dispatch | one arch | run page and artifact. A report at parameters you name |

Only the nightly writes to the repository, so no other run can overwrite a baseline.

## Runner and image

Every device job runs on the shared org pool in a tt-metal container image, so nothing is
assumed about the machine. Two shapes:

| | Image | Runner | Why |
|---|---|---|---|
| nightly, custom-report | `tt-metalium-ubuntu-22.04-release-amd64:<tag>` | `tt-ubuntu-2204-n300-stable`, `-p150b-stable` | they need *a* tt-metal, so a released one serves and no build is needed |
| perf-report, validate-kernel | `tt-metalium/ubuntu-22.04-ci-build-amd64` | same | building the commit you name is the feature, and a release image only answers for commits already released |

Each container needs **both** of these, or the device is unusable:

```yaml
options: --device /dev/tenstorrent
volumes:
  - /dev/hugepages-1G:/dev/hugepages-1G
```

Hugepages are how the card gets system memory. Without the mount, UMD opens the device and
then dies on `Broadcasts not available without system memory`.

## The shared setup

[tt-metal-env](../actions/tt-metal-env/action.yml) provisions every device job, so the four
workflows cannot drift apart.

| Step | What and why |
|---|---|
| tt-metal checkout | `models` holds tt-metal's own ULP, which this report imports rather than reimplements. Shallow, unless the job builds |
| `safe.directory` | the workspace belongs to the runner user and the container runs as root |
| paths | `TT_METAL_HOME`, `TT_METAL_CACHE`, `DATA_DIR`, `PYTHONPATH` |
| dependencies | `torch` pinned to tt-metal's version, the project itself, plus `pytest` and `typing_extensions`, which `models/common/utility_functions.py` imports |
| preflight | imports every stage module and opens the device |

Its one input, `builds`, is true for the two workflows that compile tt-metal. It selects
full history and submodules, and skips the ttnn half of the preflight, because the
ci-build image has no compiled ttnn yet.

### Three things that are easy to get wrong

| Trap | Rule |
|---|---|
| `${{ github.workspace }}` and `${{ runner.temp }}` are **host** paths | inside a container job use the shell variables `$GITHUB_WORKSPACE` and `$RUNNER_TEMP`. `${{ env.RUNNER_TEMP }}` is not a substitute: it expands to nothing, and an artifact path built from it silently uploads zero files. Where an action input needs a path, write the file into the workspace and name it relatively. A step-level `env:` also outranks `GITHUB_ENV`, so a wrong value there wins silently |
| the release image's `/opt/venv` has no pip | it is built by `uv venv`, so a bare `pip` is a different interpreter installing where `python3` cannot see it. Use `uv pip install --python "$(command -v python3)"` |
| the CLI exit code is a **count**, not an error | `measure`, `perf` and `check` return how many variants produced no data or how many findings there were. Under `set -e` a bare call kills the step, so always `|| status=$?`. A status of 128 or more is a signal. Tolerating the count is not the same as tolerating zero results: a job that produced nothing must still fail |
| a built tt-metal is not importable from `PYTHONPATH` alone | `$TT_METAL_HOME/ttnn` is the C++ source directory with no `__init__.py`, so `import ttnn` becomes a namespace package whose `__file__` is `None`; the real package is `ttnn/ttnn`, `tracy` is `tools/tracy`, and nine dependencies (`graphviz`, `networkx`, `seaborn` and the rest) are declared in tt-metal's `pyproject.toml`. Adding paths cannot supply those, so a building job runs `uv pip install -e "$TT_METAL_HOME"` after `build_metal.sh`, exactly as tt-metal's own `create_venv.sh` does. The release image needs none of this, because it ships a real `ttnn` in site-packages |

## Validating a change

```mermaid
flowchart LR
    I["commit + ops"] --> B["build tt-metal<br/>at that commit"]
    B --> C["check: measure those ops"]
    C --> A["diff vs the published report:<br/>accuracy, and timings on this host"]
    A --> V{"anything worse?"}
    V -->|yes| F["job fails, findings on the run page"]
    V -->|no| P["job passes"]
```

The published report is the baseline, never the output. Accuracy is deterministic, so it is
read from git rather than re-measured.

## One night

```mermaid
flowchart LR
    CRON["cron 02:37<br/>or manual dispatch"] --> WH
    subgraph serial["one job at a time, both push the same branch"]
        WH["measure on n300"] --> BH["measure on p150b"]
    end
    BH --> DONE["report updated,<br/>no human involved"]
```

```mermaid
flowchart TD
    A["checkout + tt-metal source at the pinned tag"] --> B["release image supplies ttnn"]
    B --> C["discover, derive, probe:<br/>manifest refreshed, per-arch layouts"]
    C --> D["measure: 6 categories × both dtypes"]
    D --> P["perf: time every variant<br/>on the build just measured"]
    P --> E["charts + report"]
    E --> F["commit reports/ report_index.json stats/,<br/>pull --rebase, push"]
```

| Property | Choice |
|---|---|
| trigger | nightly cron and `workflow_dispatch` |
| runners | `tt-ubuntu-2204-n300-stable`, `-p150b-stable`, matrix `max-parallel: 1` |
| tt-metal version | the latest release tag, resolved once per night so both archs measure one version. Recorded in `stats/runs/` |
| gate | none. Regenerate and publish; `compare` exists as a manual tool |
| image provides | `ttnn`. `models` from a checkout at the same tag, `torch` installed by the action |
| timeout | 12 h per arch |

`DATA_DIR` is the runner's scratch and does not survive the job, so a night cannot resume
from the last one. Each category is retried once, because the device can go away under a
sweep and a second attempt resumes from the CSVs the first one wrote.

## After it lands

```mermaid
flowchart LR
    N["nightly-report"] -->|report_index.json| A["analyze-report:<br/>buckets the diff, commits findings.md"]
    A -->|regression| I["opens an issue"]
    A -.->|ANTHROPIC_API_KEY set| M["a model adds one note<br/>on top of the tables"]
    N -->|reports/**| G["GitHub-browsable pages"]
```

Produced by rules, not by a model: `compare`'s buckets are the classification. The model is
optional and adds one note.
