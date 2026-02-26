## Cursor Cloud specific instructions

### Overview

Beets is a Python CLI music library management system. The main entry point is `beet` (installed as a Poetry script). See `CONTRIBUTING.rst` for full development setup and workflow details.

### Development tools

- **Poetry** (`>=1.8,<2`) — dependency/virtualenv management
- **Poethepoet** (`poe`, `>=0.26`) — task runner

Both are installed via `pipx` and live in `~/.local/bin` (already on `PATH`).

### Common commands (via `poe`)

| Task | Command |
|---|---|
| Run tests | `poe test` |
| Run tests (skip slow) | `SKIP_SLOW_TESTS=1 poe test` |
| Lint | `poe lint` |
| Check formatting | `poe check-format` |
| Auto-format | `poe format` |
| Type check | `poetry run mypy beets/plugins.py beets/metadata_plugins.py` |
| Run CLI | `poetry run beet <command>` |

All `poe` tasks are defined in `pyproject.toml` under `[tool.poe.tasks.*]`.

### Non-obvious notes

- `poe check-types` (mypy) requires explicit file/module arguments; it does not auto-discover targets. The CI workflow targets specific typed modules.
- The `test_art.py` tests require `imagemagick` system package; without it those tests fail (11 failures). It is installed in the VM environment.
- `BEETSDIR` env var overrides the beets configuration directory (useful for isolated testing with a custom `config.yaml`).
- The test suite uses `pytest`; `setup.cfg` contains pytest configuration including markers (`integration_test`, `on_lyrics_update`).
- Some optional plugin tests are skipped unless all plugin dependencies are installed (76 tests in `test_plugins.py`).
- Pre-commit hooks (`poe format`, `poe format-docs`) are configured in `.pre-commit-config.yaml` but pre-commit itself is not auto-installed; run `poe lint` and `poe check-format` before committing.
