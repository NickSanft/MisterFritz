# Contributing to Mister Fritz

Thanks for the interest. This is a personal project that's growing into something
others can use; pull requests are welcome.

## Development setup

```bash
git clone https://github.com/NickSanft/MisterFritz.git
cd MisterFritz

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install pytest pytest-cov pytest-asyncio ruff
```

You will also need [Ollama](https://ollama.com) running locally. See
[README.md](README.md) for the full list of models to pull.

## Running tests

```bash
pytest tests/ -v
```

For coverage:

```bash
pytest tests/ --cov=. --cov-report=term-missing
```

CI requires ≥ 60% coverage to pass.

## Linting

```bash
ruff check .
```

CI runs the same command. Fix any reported issues before opening a PR.

## Branching and commits

- Branch off `master`. Use a short descriptive name (`fix-chroma-lock`,
  `add-weather-tool`).
- Keep commits focused. One logical change per commit is ideal but not required.
- Commit messages: first line is a short imperative summary (under 72 chars).
  Body explains the *why* if it's not obvious from the diff.
- Don't squash PRs into a single uninformative commit. Multiple meaningful
  commits are better than one mega-commit.

## Pull requests

- Open against `master`.
- Fill in the PR template — at minimum a one-paragraph summary and a
  test-plan checklist.
- All checks must pass: tests, coverage gate, lint.
- Don't include regenerated artifacts in commits — the Mermaid diagrams
  (`mister_fritz_diagram.png`, `document_engine_diagram.png`) are rewritten
  on every import and should usually be reverted before staging.

## Adding a new agent tool

Tools live in [agent_tools.py](agent_tools.py) and are registered in
`get_conversation_tools_description()` at the bottom of the file. Add a new
`@tool(parse_docstring=True)` function, write its docstring carefully (the
LLM uses it to decide when to call), and add a `(tool, "short description")`
entry to the registry dict. Drop-in tools that live in `skills/` and expose a
`register() -> dict` function are auto-discovered.

## Adding a new slash command

Slash commands live in [bot_commands.py](bot_commands.py), inside the
`FritzCommands` cog. Use `@app_commands.command(...)`. If the command mutates
state and should only be available to admins, wrap the body with
`if not await _require_root(interaction): return`.

## Adding a new env-var knob

Don't hardcode tunables. Put them in [fritz_utils.py](fritz_utils.py) with
a sensible default, document them in [.env.example](.env.example), and add
them to the Tunables section of the README if user-facing.

## Reporting bugs

Open a GitHub issue using the **Bug report** template. Include the bot
version (`/about` in Discord shows it), the relevant log lines, and a
reproduction if possible.
