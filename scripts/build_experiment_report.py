#!/usr/bin/env python3
"""Regenerate final_results/EXPERIMENT_REPORT.md (description + 3 pivot tables)."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]

PROGRAMS = [
    "ibp",
    "deeppoly",
    "zono",
    "crownibp",
    "zid",
    "reuse",
    "polyzono",
    "deeppoly_efficient",
]
NETWORKS = ["n10", "n11", "n14", "n22", "n30", "n32"]

DESCRIPTION = """# Sparse matmul format comparison — final report

## What we did

We **did not** swap out the real runtime. ConstraintFlow still does matmul the same way it always has — our block-sparse `SparseTensor` path. What we added is a **benchmark hook** that runs after each matmul: take the same operands, try the multiply in PyTorch **CSR**, **COO**, and **BSR (8×8 blocks)**, and time it against what we already computed. Matmul is where most of the time goes, so if generic sparse formats lose there, we’re not missing much by not using them everywhere.

The flow is: run the normal block-sparse matmul → then (for comparison) densify that pair of tensors, encode to each format, run `torch.matmul`, check the answer matches. We split **encode** vs **multiply** in the logs; the tables below are **multiply only** (slowdown vs block sparse). BSR variable/derived block size is left out. Failed runs and timeouts are blank in the tables.

We ran **8 programs** (ibp, deeppoly, zono, crownibp, zid, reuse, polyzono, deeppoly_efficient) × **6 networks** (n10, n11, n14, n22, n30, n32) on different machines; results were merged from `final_results/results*.json`. Batch size is whatever that run used (not all cells used the same batch for a given network — see consolidated CSV if you need the number per cell).

## Bottom line

Block-sparse wins basically everywhere. **CSR multiply** is often only a few× slower than us. **COO** and **BSR 8×8** are a different story — usually much worse, sometimes hundreds of times slower on multiply. So the current backend isn’t just “fine”; for these workloads it’s clearly the right shape for the hot path.

---

## How to read the tables

- **Rows** = network  
- **Columns** = program  
- Number = how many times **slower than block-sparse multiply** (e.g. `10×` = 10× slower)  
- **—** = failed, timeout, or no valid BSR number  
- **Row avg / Col avg** = mean over non-empty cells in that row or column  
- Corner = mean over all filled cells in that table  

---

"""


def _fmt(v) -> str:
    return "—" if v is None else f"{v:.1f}×"


def _load(consolidated: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(consolidated.open()):
        if r.get("status") != "ok":
            continue
        rows.append(
            {
                "program": r["program"].replace(".cf", ""),
                "network": r["network"],
                "csr": _float_or_none(r.get("csr_mul_vs_block")),
                "coo": _float_or_none(r.get("coo_mul_vs_block")),
                "bsr": _float_or_none(r.get("bsr_fixed_mul_vs_block"), allow_zero=False),
            }
        )
    return rows


def _float_or_none(x, allow_zero=True):
    if x in ("", None):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not allow_zero and v <= 0:
        return None
    return v if allow_zero or v > 0 else None


def _pivot_table(rows: list[dict], key: str, title: str) -> str:
    grid = {
        (r["network"], r["program"]): r[key]
        for r in rows
        if r[key] is not None and r[key] > 0
    }
    col_vals: dict[str, list[float]] = defaultdict(list)
    lines = [f"### {title}", ""]
    lines.append("| Network | " + " | ".join(PROGRAMS) + " | **Row avg** |")
    lines.append("|---------|" + "|".join(["------:"] * len(PROGRAMS)) + "|----------:|")
    all_vals: list[float] = []
    for net in NETWORKS:
        rval = []
        cells = []
        for prog in PROGRAMS:
            v = grid.get((net, prog))
            cells.append(_fmt(v))
            if v is not None:
                rval.append(v)
                col_vals[prog].append(v)
                all_vals.append(v)
        lines.append(
            f"| **{net}** | " + " | ".join(cells) + f" | {_fmt(mean(rval) if rval else None)} |"
        )
    col_cells = [_fmt(mean(col_vals[p]) if col_vals[p] else None) for p in PROGRAMS]
    lines.append(
        f"| **Col avg** | "
        + " | ".join(col_cells)
        + f" | {_fmt(mean(all_vals) if all_vals else None)} |"
    )
    lines.append("")
    return "\n".join(lines)


def _gaps_section() -> str:
    return """---

## Gaps in the grid

These did not produce ok results (blank in tables above):

- **n32**: ibp, deeppoly, crownibp, reuse, deeppoly_efficient — failed; polyzono — timeout; only zono ok  
- **n22**: zid — timeout  

Raw numbers and batch sizes per run: `consolidated.csv` in this folder.
"""


def build_report(consolidated: Path, out_path: Path) -> None:
    rows = _load(consolidated)
    body = DESCRIPTION
    body += _pivot_table(rows, "csr", "CSR multiply slowdown vs block sparse")
    body += _pivot_table(rows, "coo", "COO multiply slowdown vs block sparse")
    body += _pivot_table(rows, "bsr", "BSR (8×8) multiply slowdown vs block sparse")
    body += _gaps_section()
    out_path.write_text(body)
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--consolidated",
        type=Path,
        default=REPO_ROOT / "final_results" / "consolidated.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "final_results" / "EXPERIMENT_REPORT.md",
    )
    args = p.parse_args()
    build_report(args.consolidated, args.out)


if __name__ == "__main__":
    main()
