# Contributing to Calm Aurora

Thanks for your interest in contributing.

## Scope

Calm Aurora is a research-oriented CIDOC CRM + RDF assistant. We welcome:

- bug fixes
- test improvements
- documentation improvements
- retrieval/prompting/policy improvements
- RDF ingestion and editing workflow improvements

## Getting Started

1. Fork the repository.
2. Create a feature branch from `main`.
3. Set up your environment:

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

4. Run tests before opening a PR:

```bash
make test
```

If your change affects end-to-end behavior, also run smoke tests:

```bash
make smoke-no-api
# or
make smoke
```

## Development Guidelines

- Keep changes focused and minimal.
- Add or update tests for behavior changes.
- Preserve existing CLI behavior unless change is intentional and documented.
- Update `README.md` when user-facing behavior changes.
- Prefer clear commit messages and small logical commits.

## Pull Request Checklist

- [ ] Code builds and tests pass locally.
- [ ] New behavior is covered by tests.
- [ ] Docs are updated for user-visible changes.
- [ ] PR description explains motivation and expected impact.

## Reporting Issues

When opening an issue, include:

- operating system
- Python version
- exact command used
- expected behavior
- observed behavior and logs
- minimal reproducible input (if possible)

## Conduct

By participating, you agree to follow the project Code of Conduct in `CODE_OF_CONDUCT.md`.
