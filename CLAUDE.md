# Code standards

Short, precise, functional. Every line earns its place.

Cycle: research → propose → branch → implement → review → improve → review → this file → commit.
See `AGENTS.md`.

## Ladder — walk it before writing anything

1. Does this need to exist?   → no: skip it (YAGNI)
2. Already in this codebase?  → reuse it, don't rewrite
3. Stdlib does it?            → use it
4. Native platform feature?   → use it
5. Installed dependency?      → use it
6. One line?                  → one line
7. Only then: the minimum that works

## Rules

- Never reimplement what a module here already provides — import it.
- Use everything you add. Unused import, variable, argument, config key or dependency: delete it.
- Newest Python available. tt-metal pins 3.10, so: PEP 604 unions, `match`, dataclass `slots`,
  parenthesized context managers. Nothing from 3.11+.
- Comments are short and detailed: record the decision a reader cannot infer from the line.
  A comment that restates its line is deleted.
- No decorative comments. Banner rules (`# ----------`), boxed headers, section dividers and
  ASCII art are deleted on sight. If a file needs signposting it is too long — split it.
- No defensive branches for states that cannot occur.
- Refactor inside existing files. Propose new structure and wait for agreement before creating it.

## Inspecting the repo

Read the file. Do not write throwaway scripts or shell pipelines to discover what a file contains.

## Run before ending every turn

1. Ladder walked at each step?
2. Anything added that is now unused?
3. Anything reimplemented that already exists?
4. Any comment that only restates its line?
5. `uvx ruff check .` and `uvx ruff format .` clean?
6. `pytest` green?
