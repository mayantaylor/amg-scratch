"""
AMG Hierarchy SpMV Benchmark using cuSPARSE
--------------------------------------------
Reads AMG level matrices from a directory of .npz files and benchmarks
SpMV at each level using cuSPARSE via CuPy.

Expected directory layout:
    <matrix-dir>/
        level_0.npz
        level_1.npz
        ...
        meta.json      (optional)

Each .npz file is loaded with scipy.sparse.load_npz().

Requirements:
    pip install cupy-cuda12x scipy numpy
    (adjust cupy-cuda12x to match your CUDA version)

Usage examples:
    # Single directory
    python amg_spmv_bench.py eval-spmv/matrices256/block_dom_decomp/

    # Multiple directories
    python amg_spmv_bench.py eval-spmv/matrices256/block_dom_decomp/ \\
                              eval-spmv/matrices256/ruge_stuben/

    # Override trial counts
    python amg_spmv_bench.py eval-spmv/matrices256/block_dom_decomp/ \\
                              --cpu-trials 50 --gpu-trials 200

    # Run in 32-bit mode
    python amg_spmv_bench.py eval-spmv/matrices256/block_dom_decomp/ --dtype float32
"""

import json
import numpy as np
import scipy.sparse as sp
import time
import os
import glob
import argparse

# ── Try to import CuPy (cuSPARSE backend) ────────────────────────────────────
try:
    import cupy as cp
    import cupyx.scipy.sparse as cpsp
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False
    print("CuPy not found — will run CPU-only reference timings.\n")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load matrices from a directory of .npz files
# ─────────────────────────────────────────────────────────────────────────────

def load_levels(matrix_dir: str, np_dtype: np.dtype) -> tuple[list[dict], dict | None]:
    """
    Scan *matrix_dir* for files named level_<N>.npz (any N), load each as a
    scipy sparse CSR matrix, and return a sorted list of level dicts.

    Also loads meta.json from the same directory if present.

    Returns
    -------
    levels : list of dicts with keys: level, A, shape, nnz
    meta   : dict from meta.json, or None if not found
    """
    pattern = os.path.join(matrix_dir, "level_*.npz")
    npz_paths = sorted(glob.glob(pattern))

    if not npz_paths:
        raise FileNotFoundError(
            f"No files matching 'level_*.npz' found in '{matrix_dir}'.\n"
            f"Check that the path is correct and the files are present."
        )

    # Load optional metadata
    meta = None
    meta_path = os.path.join(matrix_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as fh:
            meta = json.load(fh)

    levels = []
    for path in npz_paths:
        # Extract level index from filename (level_3.npz → 3)
        basename = os.path.splitext(os.path.basename(path))[0]  # "level_3"
        try:
            lvl_idx = int(basename.split("_", 1)[1])
        except (IndexError, ValueError):
            lvl_idx = len(levels)  # fallback: sequential

        A = sp.load_npz(path)
        A = A.tocsr().astype(np_dtype)

        levels.append({
            "level": lvl_idx,
            "A":     A,
            "shape": A.shape,
            "nnz":   A.nnz,
            "path":  path,
        })

    # Sort by level index in case glob ordering differs
    levels.sort(key=lambda d: d["level"])

    return levels, meta


# ─────────────────────────────────────────────────────────────────────────────
# 2.  CPU reference SpMV
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_cpu(A_scipy: sp.csr_matrix, n_trials: int = 50) -> dict:
    n = A_scipy.shape[1]
    val_bytes = A_scipy.dtype.itemsize          # 4 (float32) or 8 (float64)
    x = np.random.rand(n).astype(A_scipy.dtype)

    # Warmup
    for _ in range(5):
        _ = A_scipy @ x

    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        y = A_scipy @ x          # noqa: F841
        t1 = time.perf_counter()
        times.append(t1 - t0)

    times = np.array(times)
    bytes_moved = (A_scipy.nnz * val_bytes       # values  (float32/64)
                   + A_scipy.nnz * 4             # col_ind (int32)
                   + (A_scipy.shape[0]+1)*4      # row_ptr (int32)
                   + n * val_bytes               # x vector
                   + A_scipy.shape[0] * val_bytes)  # y vector
    bw = bytes_moved / times.mean() / 1e9    # GB/s

    return {
        "mean_ms":   float(times.mean()      * 1e3),
        "median_ms": float(np.median(times)  * 1e3),
        "min_ms":    float(times.min()       * 1e3),
        "bw_GBs":    float(bw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3.  cuSPARSE SpMV via CuPy
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_cusparse(A_scipy: sp.csr_matrix, n_trials: int = 200) -> dict:
    """
    Uploads the matrix once, then times SpMV on-device with CUDA events
    for accurate GPU timing.
    """
    n = A_scipy.shape[1]
    val_bytes = A_scipy.dtype.itemsize          # 4 (float32) or 8 (float64)
    cp_dtype  = cp.dtype(A_scipy.dtype)

    # Upload to device
    A_gpu = cpsp.csr_matrix(A_scipy)
    x_gpu = cp.random.rand(n, dtype=cp_dtype)

    # Warmup — triggers JIT compilation / cuSPARSE handle init
    for _ in range(20):
        y_gpu = A_gpu @ x_gpu   # noqa: F841
    cp.cuda.Stream.null.synchronize()

    start_ev = cp.cuda.Event()
    stop_ev  = cp.cuda.Event()

    start_ev.record()
    for _ in range(n_trials):
        y_gpu = A_gpu @ x_gpu   # noqa: F841
    stop_ev.record()
    stop_ev.synchronize()
    mean_ms = cp.cuda.get_elapsed_time(start_ev, stop_ev) / n_trials

    bytes_moved = (A_scipy.nnz * val_bytes
                   + A_scipy.nnz * 4
                   + (A_scipy.shape[0]+1)*4
                   + n * val_bytes
                   + A_scipy.shape[0] * val_bytes)
    bw = bytes_moved / (mean_ms * 1e-3) / 1e9  # GB/s

    return {
        "mean_ms": float(mean_ms),
        "bw_GBs":  float(bw),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Formatting helpers
# ─────────────────────────────────────────────────────────────────────────────

_HEADER = (f"{'Lvl':>3}  {'Rows':>10}  {'NNZ':>11}  "
           f"{'CPU mean(ms)':>13}  {'CPU BW(GB/s)':>13}  "
           f"{'GPU mean(ms)':>13}  {'GPU BW(GB/s)':>13}  "
           f"{'Speedup':>8}")
_SEP    = "-" * len(_HEADER)


def print_results(results: list):
    print(_HEADER)
    print(_SEP)
    for r in results:
        cpu     = r["cpu"]
        gpu     = r.get("gpu")
        speedup = f"{cpu['mean_ms']/gpu['mean_ms']:>7.1f}x" if gpu else "     N/A"
        gpu_ms  = f"{gpu['mean_ms']:>13.4f}"  if gpu else f"{'N/A':>13}"
        gpu_bw  = f"{gpu['bw_GBs']:>13.2f}"   if gpu else f"{'N/A':>13}"
        print(f"{r['level']:>3}  {r['shape'][0]:>10,}  {r['nnz']:>11,}  "
              f"{cpu['mean_ms']:>13.4f}  {cpu['bw_GBs']:>13.2f}  "
              f"{gpu_ms}  {gpu_bw}  {speedup}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Per-directory benchmark driver
# ─────────────────────────────────────────────────────────────────────────────

def run_directory(matrix_dir: str, cpu_trials: int, gpu_trials: int,
                  np_dtype: np.dtype) -> list:
    """Load all levels from *matrix_dir*, benchmark, return result list."""
    levels, meta = load_levels(matrix_dir, np_dtype)

    if meta:
        print(f"  meta.json: {meta}")

    results = []
    for lvl in levels:
        A = lvl["A"]
        i = lvl["level"]
        print(f"  Benchmarking level {i}  "
              f"({A.shape[0]:,} × {A.shape[1]:,}, nnz={A.nnz:,}) ...")

        cpu_stats = benchmark_cpu(A, n_trials=cpu_trials)
        gpu_stats = benchmark_cusparse(A, n_trials=gpu_trials) if HAVE_CUPY else None

        results.append({
            "level": i,
            "shape": A.shape,
            "nnz":   A.nnz,
            "cpu":   cpu_stats,
            "gpu":   gpu_stats,
        })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpMV benchmark over AMG level matrices stored as .npz files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "matrix_dirs", nargs="+", metavar="MATRIX_DIR",
        help="One or more directories containing level_<N>.npz files",
    )
    parser.add_argument("--cpu-trials", type=int, default=20,
                        help="Number of CPU SpMV trials per level (default: 20)")
    parser.add_argument("--gpu-trials", type=int, default=50,
                        help="Number of GPU SpMV trials per level (default: 50)")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64",
                        help="Floating-point precision for SpMV (default: float64)")

    args = parser.parse_args()
    np_dtype = np.dtype(args.dtype)

    for matrix_dir in args.matrix_dirs:
        matrix_dir = matrix_dir.rstrip("/")
        label = os.path.basename(matrix_dir) or matrix_dir

        print()
        print("=" * len(_HEADER))
        print(f"Directory : {matrix_dir}")
        print(f"SpMV Benchmark — {args.dtype} CSR  |  "
              f"CPU trials={args.cpu_trials}  GPU trials={args.gpu_trials}")
        print("=" * len(_HEADER))

        try:
            results = run_directory(matrix_dir, args.cpu_trials, args.gpu_trials,
                                    np_dtype)
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
            continue

        print()
        print_results(results)
        print()

        if HAVE_CUPY:
            print(f"  GPU effective bandwidth by level (label: {label}):")
            for r in results:
                if r["gpu"]:
                    print(f"    Level {r['level']}: {r['gpu']['bw_GBs']:.2f} GB/s  "
                          f"(mean {r['gpu']['mean_ms']:.4f} ms)")

    print()


if __name__ == "__main__":
    main()
