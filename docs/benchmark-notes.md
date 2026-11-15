# Benchmark notes

Do not publish a CPU-versus-GPU chart without recording:

- CPU model and core count
- GPU model and VRAM, if applicable
- RAM
- Operating system
- Python, PyTorch, Transformers, and CUDA versions
- Exact model ID and revision
- Batch size, number of samples, warmup count, and repeats
- Whether model loading and tokenization are included in the timed section

The current benchmark measures repeated inference after model initialization. It reports
mean wall-clock duration, average milliseconds per sample, throughput, process RSS delta,
and peak CUDA allocated memory when CUDA is active.

This first version does not yet report p50/p95 latency or model file size. Add these before
using the results for a serious runtime comparison.

GPU results are sensitive to batch size. Always test several batch sizes and avoid drawing
conclusions from a single run. CPU and GPU should use the same model and input set.
