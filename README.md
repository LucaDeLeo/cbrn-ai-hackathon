# RobustCBRN Eval

Toolkit to robustify CBRN MCQA benchmarks: consensus/shortcut detection, verified cloze variants, statistical bias battery; deterministic and fail‑graceful.

[Python 3.10+], [MIT]

## What it is
Practical toolkit to evaluate and improve robustness of AI models on CBRN (Chemical, Biological, Radiological, Nuclear) multiple‑choice QA. Implements choices‑only consensus screens, verified cloze scoring, and a heuristics battery (position bias, longest‑answer), with reproducible, fail‑graceful execution.

## Quick Start
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Create venv and install deps:
  - `uv venv && uv pip install -r requirements.txt`
- Run a sample:
  - `make setup && make sample`
- Full pipeline (recommended):
  - `make pipeline`

More pipeline options are documented in `scripts/PIPELINE_README.md`.

## Docs
- Overview and rationale: `overview.md`
- Getting started and CLI usage: `docs/getting-started/usage.md`
- Architecture: `docs/architecture/architecture.md`
- Security & release policy: `docs/safety/security-considerations.md`
- Results/report templates: `docs/results/`
- Full docs index: `docs/README.md`

## Development
- Lint: `.venv/bin/ruff check robustcbrn tests`
- Tests: `.venv/bin/pytest -q`
- Pre‑commit hooks:
  - `pip install pre-commit`
  - `pre-commit install` (or `bash scripts/install-hooks.sh`)

## Notes
- Cross‑platform pipeline (Windows/macOS/Linux) with robust error handling.
- Public artifacts are sanitized (no raw questions/choices or per‑item exploit labels). See `scripts/validate_release.sh`.

[Python 3.10+]: https://www.python.org/
[MIT]: LICENSE
- **Release Checklist**: [`docs/safety/release-checklist.md`](docs/safety/release-checklist.md)

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code of conduct
- Development workflow
- Commit message conventions
- Pull request process
- Testing requirements

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{robustcbrn-eval,
  title = {RobustCBRN Eval: Toolkit for Robustifying CBRN AI Benchmarks},
  author = {[Authors]},
  year = {2024},
  url = {https://github.com/apart-research/robustcbrn-eval}
}
```

## Safety & Release Policy

See [`docs/safety/security-considerations.md`](docs/safety/security-considerations.md) for anonymization and public artifact rules.
