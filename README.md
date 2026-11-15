# Local ML Toolkit

A small, practical starter toolkit for running pretrained NLP models locally.

**First iteration:** sentiment inference with a Hugging Face model, batch prediction,
JSONL streaming from stdin, a FastAPI endpoint, and a reproducible latency benchmark.

The goal is to make local inference easy to try and easy to measure. This is an early
project scaffold, not a claim that CPU/GPU performance or quantization is already
optimized.

## Features

- Run a pretrained text-classification model locally.
- Predict one text or a batch of texts.
- Stream newline-delimited JSON records through the CLI.
- Expose inference through FastAPI.
- Benchmark latency, throughput, and process memory.
- Select CPU or CUDA when supported.
- Keep model loading separate from API and CLI code for future backends.

## Quick start

Python 3.10+ is recommended.

```bash
git clone https://github.com/<YOUR-USERNAME>/local-ml-toolkit.git
cd local-ml-toolkit

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -e ".[dev]"
```

Run a prediction (the first run downloads the model from Hugging Face):

```bash
local-ml predict "This toolkit is surprisingly easy to use."
```

Run a batch:

```bash
local-ml predict \
  "The experience was excellent." \
  "The package crashes every time."
```

Run the API:

```bash
uvicorn local_ml_toolkit.api:app --reload
```

Then:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts":["I love this product","This is frustrating"]}'
```

Stream JSONL from stdin:

```bash
printf '%s\n' \
  '{"text":"Really useful"}' \
  '{"text":"Not working as expected"}' \
  | local-ml stream
```

## Benchmark

```bash
local-ml benchmark --samples 64 --batch-size 8 --warmup 2 --repeats 5
```

Choose a device explicitly:

```bash
local-ml benchmark --device cpu
local-ml benchmark --device cuda
```

CUDA requires a CUDA-enabled PyTorch installation and a compatible GPU. The benchmark
reports the selected device and does not silently claim a GPU run when CUDA is absent.

For meaningful comparisons, keep the model, input data, batch size, warmup count,
repeat count, software versions, and hardware constant. Run CPU and GPU tests separately.
Small models and small batches can be slower on GPU because of transfer and launch overhead.

## Python API

```python
from local_ml_toolkit import LocalTextClassifier

model = LocalTextClassifier()
result = model.predict("A clear and useful example.")
print(result)
```

Batch:

```python
results = model.predict_many([
    "A great experience.",
    "A disappointing experience.",
])
```

## Default model

`distilbert/distilbert-base-uncased-finetuned-sst-2-english`

This is a public Hugging Face text-classification checkpoint. Review its model card,
license, intended use, and limitations before redistribution or production use.
The first run requires internet access to download model files; subsequent runs can
use the local Hugging Face cache. This project does not upload input text to an API.

## Project structure

```text
src/local_ml_toolkit/
  __init__.py
  model.py
  cli.py
  api.py
tests/
  test_model.py
  test_api.py
docs/
  roadmap.md
  benchmark-notes.md
```

## Current limitations

- The first backend uses the Hugging Face Transformers PyTorch pipeline.
- ONNX Runtime, quantization, model comparisons, and streaming HTTP responses are planned,
  not implemented in this initial version.
- The default model is English sentiment classification only.
- Benchmark memory is process-level RSS; GPU memory is reported only when CUDA is active.
- Benchmark numbers are machine-dependent and should not be compared across different setups
  without recording hardware and software details.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) and [docs/benchmark-notes.md](docs/benchmark-notes.md).

## Contributing

Issues and pull requests are welcome. Please include reproducible steps, environment details,
and benchmark settings when reporting performance or correctness issues.

## License

MIT. See [LICENSE](LICENSE).
