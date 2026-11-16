# Contributing to anyjev

Thanks for helping. The bar for a change is: does it make the first five minutes
with anyjev easier, or the results more trustworthy?

## Setup

```bash
git clone https://github.com/ubairnisar/anyjev && cd anyjev
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

The tests need no GPU and no model: `tests/fakes.py` scripts the logprobs, and
`tests/test_http_e2e.py` runs the real `openai` client against a tiny
vLLM-shaped HTTP server.

## Good first contributions

- A new preset in `src/anyjev/presets.py` (with a test and a README line).
- A report that a model family answers with something other than a letter
  first (include the model, the vLLM version and `anyjev doctor` output).
- Docs and examples for a real use case.

## Style

`ruff check src tests` must pass. Keep runtime dependencies to `openai` and `pyyaml`.
