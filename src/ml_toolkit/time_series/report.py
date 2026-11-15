"""Create readable benchmark reports."""


def format_report(results):
    """Format benchmark results as a Markdown table."""
    if not results:
        raise ValueError("No benchmark results supplied.")

    lines = [
        "# Time-Series Benchmark Report",
        "",
        "| Model | MAE | RMSE | MAPE (%) | Runtime (s) |",
        "|---|---:|---:|---:|---:|",
    ]

    for result in results:
        lines.append(
            f"| {result['model']} "
            f"| {result['mae']:.4f} "
            f"| {result['rmse']:.4f} "
            f"| {result['mape']:.2f} "
            f"| {result['runtime_seconds']:.6f} |"
        )

    best = min(results, key=lambda item: item["mae"])

    lines.extend([
        "",
        f"Best model by MAE: **{best['model']}**",
    ])

    return "\n".join(lines)