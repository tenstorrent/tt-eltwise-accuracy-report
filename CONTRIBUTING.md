# Contributing

Thanks for your interest in improving `tt-eltwise-accuracy-report`.

## Reporting bugs

Report bugs via [GitHub Issues](https://github.com/tenstorrent/tt-eltwise-accuracy-report/issues).
Include the command you ran, the TT-Metal commit, architecture, and dtype involved.

## Submitting changes

Bug fixes and new functionality are submitted via Pull Requests. Pull requests are reviewed on a
weekly cadence. By participating in this project, you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

Before opening a pull request:

1. Follow the conventions in [CLAUDE.md](CLAUDE.md) — this is the canonical style guide for this
   repository.
2. Run `uvx ruff check .` and `uvx ruff format .` and confirm both are clean.
3. Run `pytest` and confirm the suite is green.
4. Keep changes surgical: propose a new file or package before adding one.

## Reporting security issues

Do not use public GitHub Issues for security vulnerabilities — see [SECURITY.md](SECURITY.md).
