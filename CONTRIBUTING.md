# Contributing

Thanks for considering a contribution.

1. Open an issue describing the problem or proposed change.
2. Keep pull requests focused and include tests.
3. For performance changes, include hardware, software versions, model ID, and benchmark settings.
4. Do not commit model weights, private datasets, credentials, or generated environment files.
5. Check model and dataset licenses before adding them to examples.

Run checks:

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
