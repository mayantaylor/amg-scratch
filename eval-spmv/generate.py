"""
generate_amg_hierarchies.py
----------------------------
Generates 6-level AMG hierarchies for four sparse matrix types spanning a range
of sparsity structures, and writes each level to disk in scipy .npz format (CSR).

Matrix types
------------
1. poisson         -- 5-point 2D Poisson on a regular grid. Perfectly banded with
                      bandwidth ~grid_size, ~5 NNZ/row. Baseline structured case.
                      AMG method: Smoothed Aggregation (SA).

2. linear_elasticity -- 2D linear elasticity, 2 DOF per node. Block-structured:
                      the matrix has a natural 2x2 dense block pattern inside the
                      same banded envelope as Poisson but with ~18 NNZ/row. Classic
                      AMG test case for vector-valued PDEs.
                      AMG method: SA with near-nullspace candidates (rigid body modes).

3. delaunay        -- Graph Laplacian of a 2D Delaunay triangulation on uniformly
                      random points. Irregular sparsity within a rough bandwidth —
                      same ~7 NNZ/row as Poisson but with no regular column-index
                      pattern, mimicking unstructured FEM meshes.
                      AMG method: Ruge-Stuben (RS), which coarsens better than SA
                      on this geometry.

4. block_dom_decomp -- Block-diagonal matrix (64 tridiagonal blocks) with weak
                      random inter-block connections. Models a domain-decomposed or
                      multi-physics system: dense local coupling within blocks,
                      sparse random coupling between them. ~13 NNZ/row.
                      AMG method: RS, which respects the block structure.

All randomness is seeded so the same grid_size always produces the same matrices.

Usage
-----
    python generate_amg_hierarchies.py --grid_size 256 --output_dir ./amg_matrices
    python generate_amg_hierarchies.py --grid_size 256 --output_dir ./amg_matrices --dtype float32

    Note: grid_size must be >= 256 for all cases to reach 6 AMG levels.
          linear_elasticity produces a matrix of size (2*grid_size^2) x (2*grid_size^2)
          due to its 2-DOF-per-node structure.

Output layout
-------------
    <output_dir>/
        poisson/
            level_0.npz       # finest (= input matrix A)
            level_1.npz
            ...
            level_5.npz       # coarsest
            meta.json         # shape, nnz, nnz_per_row for each level
        linear_elasticity/
            ...
        delaunay/
            ...
        block_dom_decomp/
            ...
        summary.json

Each .npz file can be reloaded with:
    import scipy.sparse as sp
    A = sp.load_npz("level_0.npz")
"""

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
import pyamg
import pyamg.gallery as gallery

RANDOM_SEED = 0
TARGET_LEVELS = 6
AMG_MAX_LEVELS = TARGET_LEVELS + 2  # allow a little headroom; we truncate to TARGET_LEVELS


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------

def build_poisson(grid_size, rng):
    """
    5-point 2D Poisson on a (grid_size x grid_size) grid.
    Perfectly regular banded structure, ~5 NNZ/row. Fully deterministic,
    no random state consumed.
    """
    _ = rng
    return gallery.poisson((grid_size, grid_size), format="csr")


def build_linear_elasticity(grid_size, rng):
    """
    2D linear elasticity on a (grid_size x grid_size) grid, 2 DOF per node.
    Returns (A, B) where B contains the rigid-body-mode near-nullspace candidates
    needed by SA for optimal coarsening.
    Matrix size: (2*grid_size^2) x (2*grid_size^2), ~18 NNZ/row.
    """
    _ = rng
    A, B = gallery.linear_elasticity((grid_size, grid_size))
    return A.tocsr(), B


def build_delaunay(grid_size, rng):
    """
    Graph Laplacian of a 2D Delaunay triangulation on (grid_size^2) uniformly
    random points in [0,1]^2. Irregular column indices within a rough bandwidth —
    same low NNZ/row as Poisson (~7) but no regular stencil pattern.
    Point coordinates are drawn from rng for reproducibility.
    """
    n_points = grid_size * grid_size
    pts = rng.random((n_points, 2))
    tri = Delaunay(pts)

    rows, cols = [], []
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(3):
                if i != j:
                    rows.append(simplex[i])
                    cols.append(simplex[j])

    rows = np.array(rows)
    cols = np.array(cols)
    data = -np.ones(len(rows))

    A = sp.csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    # Diagonal = degree of each node (makes row sums zero, i.e. proper Laplacian)
    degree = np.array(-A.sum(axis=1)).flatten()
    A = A + sp.diags(degree, format="csr")
    return A


def build_block_dom_decomp(grid_size, rng):
    """
    Block-diagonal matrix with 64 tridiagonal blocks plus weak random
    inter-block connections. Models a domain-decomposed system:
    - Strong intra-block coupling (tridiagonal, coefficient = 1)
    - Weak inter-block coupling (random, coefficient ~ 0.01)
    NNZ/row ~13. The block structure causes AMG to coarsen in two phases:
    first within blocks, then across them.
    """
    n = grid_size * grid_size
    n_blocks = 64
    block_size = n // n_blocks

    # Build block-diagonal part (64 tridiagonal blocks)
    blocks = []
    for _ in range(n_blocks):
        diag_main = 2.0 * np.ones(block_size)
        diag_off  = -1.0 * np.ones(block_size - 1)
        B = sp.diags(
            [diag_off, diag_main, diag_off], [-1, 0, 1],
            shape=(block_size, block_size), format="csr",
        )
        blocks.append(B)
    A = sp.block_diag(blocks, format="csr")

    # Add weak random inter-block connections (~5 per row on average)
    n_connections = 5 * n
    row_idx = rng.integers(0, n, n_connections)
    col_idx = rng.integers(0, n, n_connections)
    # Keep only entries that cross block boundaries
    cross = (row_idx // block_size) != (col_idx // block_size)
    row_idx, col_idx = row_idx[cross], col_idx[cross]
    vals = rng.random(len(row_idx)) * 0.01
    off = sp.csr_matrix((vals, (row_idx, col_idx)), shape=(n, n))
    off = off + off.T  # symmetrise

    A = A + off
    return A


# ---------------------------------------------------------------------------
# AMG hierarchy builder
# ---------------------------------------------------------------------------

def build_hierarchy(case_name, A, rng, B=None):
    """
    Run AMG and return exactly TARGET_LEVELS level matrices.

    Uses Smoothed Aggregation (SA) for structured matrices (poisson,
    linear_elasticity) and Ruge-Stuben (RS) for irregular ones (delaunay,
    block_dom_decomp), matching each solver to the matrix structure.

    np.random.seed is set before each solver call so that any internal numpy
    usage inside pyamg is also reproducible.
    """
    np.random.seed(RANDOM_SEED)

    if case_name in ("poisson", "linear_elasticity"):
        ml = pyamg.smoothed_aggregation_solver(A, B=B, max_levels=AMG_MAX_LEVELS)
    else:
        ml = pyamg.ruge_stuben_solver(A, max_levels=AMG_MAX_LEVELS)

    n_levels = len(ml.levels)
    if n_levels < TARGET_LEVELS:
        raise RuntimeError(
            f"[{case_name}] Only {n_levels} AMG levels generated (need {TARGET_LEVELS}). "
            f"Increase grid_size (minimum recommended: 256)."
        )

    return [lvl.A.tocsr() for lvl in ml.levels[:TARGET_LEVELS]]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_hierarchy(matrices, case_name, output_dir, np_dtype):
    case_dir = os.path.join(output_dir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    meta = {"case": case_name, "dtype": np_dtype.str, "levels": []}

    for i, A in enumerate(matrices):
        A = A.astype(np_dtype)                  # cast values; indices stay int32
        path = os.path.join(case_dir, f"level_{i}.npz")
        sp.save_npz(path, A)

        nnz_per_row = A.nnz / A.shape[0]
        meta["levels"].append({
            "level": i,
            "rows": A.shape[0],
            "cols": A.shape[1],
            "nnz": A.nnz,
            "nnz_per_row": round(nnz_per_row, 2),
            "file": f"level_{i}.npz",
        })

        print(
            f"  level {i}: shape=({A.shape[0]:>7,}, {A.shape[1]:>7,})  "
            f"nnz={A.nnz:>10,}  nnz/row={nnz_per_row:6.1f}"
        )

    meta_path = os.path.join(case_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return case_dir


# ---------------------------------------------------------------------------
# Cases registry
# ---------------------------------------------------------------------------

# Each entry: (name, builder_fn, description)
# builder_fn signature: (grid_size, rng) -> A  or  (grid_size, rng) -> (A, B)
CASES = [
    (
        "poisson",
        build_poisson,
        "5-pt 2D Poisson — regular banded, ~5 NNZ/row",
    ),
    (
        "linear_elasticity",
        build_linear_elasticity,
        "2D linear elasticity — 2-DOF block-banded, ~18 NNZ/row",
    ),
    (
        "delaunay",
        build_delaunay,
        "Delaunay mesh Laplacian — irregular unstructured, ~7 NNZ/row",
    ),
    (
        "block_dom_decomp",
        build_block_dom_decomp,
        "Block tridiagonal + weak random off-block — block-sparse, ~13 NNZ/row",
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 6-level AMG hierarchies for SpMV benchmarking."
    )
    parser.add_argument(
        "--grid_size",
        type=int,
        default=256,
        help="Side length of the 2D grid. Must be >= 256 for 6 levels. (default: 256)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./amg_matrices",
        help="Root directory for output files. (default: ./amg_matrices)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Floating-point dtype for saved matrix values (default: float64)",
    )
    args = parser.parse_args()

    grid_size = args.grid_size
    output_dir = args.output_dir
    np_dtype = np.dtype(args.dtype)

    if grid_size < 256:
        print(
            f"WARNING: grid_size={grid_size} may not produce {TARGET_LEVELS} AMG levels "
            "for all cases. Recommended minimum is 256.",
            file=sys.stderr,
        )

    # Single RNG instance seeded once. Passed to every builder in fixed order,
    # so the same grid_size always produces identical outputs regardless of
    # which cases are run or in what environment.
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"grid_size     : {grid_size}  (N = {grid_size**2:,})")
    print(f"output_dir    : {os.path.abspath(output_dir)}")
    print(f"target levels : {TARGET_LEVELS}")
    print(f"random seed   : {RANDOM_SEED}")
    print(f"dtype         : {args.dtype}")
    print()

    for case_name, builder, description in CASES:
        print(f"=== {case_name}  ({description}) ===")

        result = builder(grid_size, rng)
        if isinstance(result, tuple):
            A, B = result
        else:
            A, B = result, None

        matrices = build_hierarchy(case_name, A, rng, B=B)
        save_hierarchy(matrices, case_name, output_dir, np_dtype)
        print()

    summary = {
        "grid_size": grid_size,
        "random_seed": RANDOM_SEED,
        "target_levels": TARGET_LEVELS,
        "dtype": args.dtype,
        "cases": [name for name, _, _ in CASES],
        "amg_methods": {
            "poisson": "smoothed_aggregation",
            "linear_elasticity": "smoothed_aggregation (with near-nullspace B)",
            "delaunay": "ruge_stuben",
            "block_dom_decomp": "ruge_stuben",
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Done. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
