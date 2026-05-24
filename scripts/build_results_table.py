#!/usr/bin/env python3
"""Build presentation results table from final_results/consolidated.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fmt_ratio(x) -> str:
    if x == "" or x is None:
        return "—"
    return f"{float(x):.1f}×"


def build(consolidated: Path, out_dir: Path) -> int:
    rows_in = list(csv.DictReader(consolidated.open()))
    out = []
    for r in rows_in:
        if r.get("status") != "ok":
            continue
        try:
            csr_f = float(r["csr_mul_vs_block"])
            coo_f = float(r["coo_mul_vs_block"])
        except (TypeError, ValueError, KeyError):
            continue
        if csr_f <= 0 or coo_f <= 0:
            continue
        bsr_raw = r.get("bsr_fixed_mul_vs_block") or ""
        try:
            bsr_f = float(bsr_raw) if bsr_raw not in ("", "0", "0.0") else None
        except ValueError:
            bsr_f = None
        if bsr_f is not None and bsr_f <= 0:
            bsr_f = None

        out.append(
            {
                "program": r["program"].replace(".cf", ""),
                "network": r["network"],
                "batch_size": r["batch_size"],
                "csr_slowdown": csr_f,
                "coo_slowdown": coo_f,
                "bsr_fixed_slowdown": bsr_f if bsr_f is not None else "",
            }
        )

    out.sort(key=lambda x: (x["network"], x["program"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    fields = [
        "program",
        "network",
        "batch_size",
        "csr_slowdown",
        "coo_slowdown",
        "bsr_fixed_slowdown",
    ]
    csv_path = out_dir / "results_table.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)

    lines = [
        "# Sparse matmul slowdown vs block sparse",
        "",
        "Block-sparse matmul is the baseline (fastest). Values show how many times "
        "**slower** CSR, COO, and BSR (8×8) multiply are on the same workload.",
        "",
        "BSR derived/dynamic omitted. Failed and timeout runs omitted.",
        "",
        "| Program | Network | Batch | CSR | COO | BSR (8×8) |",
        "|---------|---------|------:|----:|----:|----------:|",
    ]
    for r in out:
        lines.append(
            f"| {r['program']} | {r['network']} | {r['batch_size']} | "
            f"{_fmt_ratio(r['csr_slowdown'])} | {_fmt_ratio(r['coo_slowdown'])} | "
            f"{_fmt_ratio(r['bsr_fixed_slowdown'])} |"
        )

    by_net: dict[tuple, list] = defaultdict(list)
    for r in out:
        by_net[(r["network"], r["batch_size"])].append(r)

    lines.extend(
        [
            "",
            "## By network (median across programs)",
            "",
            "| Network | Batch | CSR | COO | BSR (8×8) |",
            "|---------|------:|----:|----:|----------:|",
        ]
    )
    for net, bs in sorted(by_net.keys()):
        grp = by_net[(net, bs)]
        bsr_vals = [g["bsr_fixed_slowdown"] for g in grp if g["bsr_fixed_slowdown"] != ""]
        lines.append(
            f"| {net} | {bs} | "
            f"{_fmt_ratio(median([g['csr_slowdown'] for g in grp]))} | "
            f"{_fmt_ratio(median([g['coo_slowdown'] for g in grp]))} | "
            f"{_fmt_ratio(median(bsr_vals) if bsr_vals else None)} |"
        )

    md_path = out_dir / "results_table.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(out)} rows → {csv_path} and {md_path}")
    return len(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--consolidated",
        type=Path,
        default=REPO_ROOT / "final_results" / "consolidated.csv",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "final_results",
    )
    args = p.parse_args()
    build(args.consolidated, args.out_dir)


if __name__ == "__main__":
    main()
