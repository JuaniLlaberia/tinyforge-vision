"""
Timing run for scripts/prepare_cauldron.py: measures download speed and per-image processing
cost on a few light and heavy subsets, then estimates how long the full preprocessing would take
on this machine. Nothing is pushed and nothing large is kept on disk.

Usage:
    uv run python scripts/benchmark_prepare.py [--rows 200] [--num-proc 8] [--subsets ai2d docvqa]
"""
import argparse
import os

from tfv.config import PrepareConfig
from tfv.data.prepare import benchmark_download, benchmark_processing, fetch_subset_sizes

def parse_args() -> argparse.Namespace:
    """
    Parses command-line overrides of PrepareConfig's benchmark settings.

    Returns:
        argparse.Namespace: subsets, rows and num_proc.
    """
    cfg = PrepareConfig()
    parser = argparse.ArgumentParser(description="Estimate the full Cauldron preprocessing time")
    parser.add_argument("--subsets", nargs="+", default=list(cfg.benchmark_subsets))
    parser.add_argument("--rows", type=int, default=cfg.benchmark_rows, help="Rows timed per subset")
    parser.add_argument("--num-proc", type=int, default=cfg.num_proc or os.cpu_count())
    return parser.parse_args()

def format_hours(seconds: float) -> str:
    """
    Formats a duration for the summary table.

    Args:
        seconds: Duration in seconds.
    Returns:
        str: e.g. "1h 23m".
    """
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m"

def main() -> None:
    """
    Runs the benchmark and prints per-subset timings plus the full-run estimate.
    """
    args = parse_args()
    cfg = PrepareConfig()
    sizes = fetch_subset_sizes(cfg.source_path)
    total_rows = sum(sizes[name]["rows"] for name in cfg.subsets)
    total_bytes = sum(sizes[name]["bytes"] for name in cfg.subsets)
    print(f"{len(cfg.subsets)} subsets: {total_rows:,} rows, {total_bytes / 1e9:.1f} GB to download")
    print(f"Using {args.num_proc} processes (os.cpu_count() = {os.cpu_count()})\n")

    print(f"{'subset':24s} {'ms/row':>8s} {'s/MB in':>8s} {'KB/row out':>10s}")
    measured = {}
    for name in args.subsets:
        measured[name] = benchmark_processing(name, cfg, args.rows)
        m = measured[name]
        print(f"{name:24s} {m['sec_per_row'] * 1e3:8.1f} {m['sec_per_input_mb']:8.3f} {m['output_bytes_per_row'] / 1e3:10.1f}")

    # Unmeasured subsets are estimated from their size: processing cost scales with image bytes.
    mean_sec_per_mb = sum(m["sec_per_input_mb"] for m in measured.values()) / len(measured)
    mean_out_per_row = sum(m["output_bytes_per_row"] for m in measured.values()) / len(measured)
    process_seconds = 0.0
    for name in cfg.subsets:
        if name in measured:
            process_seconds += measured[name]["sec_per_row"] * sizes[name]["rows"]
        else:
            process_seconds += mean_sec_per_mb * sizes[name]["bytes"] / 1e6
    process_seconds /= args.num_proc

    mb_per_second = benchmark_download(cfg.source_path, args.subsets[0], cfg.work_dir)
    download_seconds = total_bytes / 1e6 / mb_per_second
    output_gb = mean_out_per_row * total_rows / 1e9

    print(f"\nDownload speed: {mb_per_second:.0f} MB/s")
    print(f"Estimated download:   {format_hours(download_seconds)}")
    print(f"Estimated processing: {format_hours(process_seconds)}  ({args.num_proc} processes)")
    print(f"Estimated total:      {format_hours(download_seconds + process_seconds)}")
    print(f"Estimated output size: {output_gb:.0f} GB (upload time not measured)")
    print("Not included: Arrow conversion in load_dataset and the upload; expect roughly +15-30%.")

if __name__ == "__main__":
    main()
