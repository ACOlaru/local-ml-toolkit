# Roadmap

## v0.1 — Local inference baseline

- [x] Hugging Face text-classification wrapper
- [x] Single and batch predictions
- [x] JSONL stdin streaming
- [x] FastAPI endpoint
- [x] CPU/CUDA selection
- [x] Basic latency and memory benchmark
- [x] Tests that do not require model downloads

## v0.2 — Benchmark quality

- [ ] Record Python, PyTorch, Transformers, OS, CPU, and GPU metadata
- [ ] Add warmup and percentile latency (p50/p95)
- [ ] Compare batch sizes 1, 4, 8, 16, 32
- [ ] Save benchmark results as JSON/CSV
- [ ] Add a benchmark-results template and reproducibility guide

## v0.3 — Efficient runtimes

- [ ] Add ONNX Runtime backend
- [ ] Compare PyTorch and ONNX Runtime on identical inputs
- [ ] Add dynamic quantization where supported
- [ ] Measure model size, process RSS, and inference latency
- [ ] Document numerical differences and quality trade-offs

## v0.4 — More useful model interfaces

- [ ] Add configurable task/model pairs
- [ ] Add embedding inference example
- [ ] Add bounded async/batch queue for API requests
- [ ] Add optional streaming HTTP response
- [ ] Add Docker example

## Design principles

1. Local-first: no remote inference API is required.
2. Reproducible: benchmark settings and hardware must be recorded.
3. Honest: distinguish measured results from estimates.
4. Small surface area: keep the public Python API easy to understand.
5. Model-aware: document each model's license, intended use, and limitations.
