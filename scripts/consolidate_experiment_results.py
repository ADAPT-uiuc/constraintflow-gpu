#!/usr/bin/env python3
"""Merge results*.json from final_results/ into one deduplicated table."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PROGRAMS = [
    "ibp.cf",
    "deeppoly.cf",
    "zono.cf",
    "crownibp.cf",
    "zid.cf",
    "reuse.cf",
    "polyzono.cf",
    "deeppoly_efficient.cf",
]
NETWORKS = ["n10", "n11", "n14", "n22", "n30", "n32"]

EXPECTED = {
    "n10": {"eps": 0.005, "batch_size": 100, "dataset": "mnist"},
    "n11": {"eps": 0.004, "batch_size": 100, "dataset": "mnist"},
    "n14": {"eps": 0.1, "batch_size": 100, "dataset": "mnist"},
    "n22": {"eps": 4e-5, "batch_size": 100, "dataset": "cifar"},
    "n30": {"eps": 5e-4, "batch_size": 10, "dataset": "cifar"},
    "n32": {"eps": 0.002, "batch_size": 10, "dataset": "cifar"},
}

CSV_COLUMNS = [
    "program",
    "network",
    "dataset",
    "batch_size",
    "expected_batch_size",
    "batch_match",
    "eps",
    "status",
    "wall_time_s",
    "block_sparse_s",
    "csr_encode_s",
    "csr_mul_s",
    "coo_encode_s",
    "coo_mul_s",
    "bsr_fixed_encode_s",
    "bsr_fixed_mul_s",
    "bsr_dynamic_encode_s",
    "bsr_dynamic_mul_s",
    "verify_s",
    "csr_mul_vs_block",
    "coo_mul_vs_block",
    "bsr_fixed_mul_vs_block",
    "bsr_dynamic_mul_vs_block",
    "dedup_note",
    "source_file",
]


def _norm(row: dict) -> dict:
    r = dict(row)
    for k in ("eps", "batch_size"):
        if k in r and r[k] not in ("", None):
            r[k] = int(float(r[k])) if k == "batch_size" else float(r[k])
    for k in list(r.keys()):
        if k.endswith("_s") or k.endswith("_vs_block") or k == "wall_time_s":
            if r.get(k) in ("", None):
                r[k] = None
            else:
                try:
                    r[k] = float(r[k])
                except (TypeError, ValueError):
                    pass
    return r


def _row_score(r: dict) -> tuple:
    exp = EXPECTED[r["network"]]
    batch_match = r.get("batch_size") == exp["batch_size"]
    status_score = {"ok": 3, "timeout": 2, "failed": 1}.get(r.get("status"), 0)
    has_metrics = r.get("block_sparse_s") is not None
    return (batch_match, status_score, has_metrics)


def load_all(input_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(input_dir.glob("results*.json")):
        for row in json.loads(path.read_text()):
            r = _norm(row)
            r["source_file"] = path.name
            rows.append(r)
    return rows


def consolidate(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["program"], r["network"])].append(r)

    out = []
    for prog in PROGRAMS:
        for net in NETWORKS:
            key = (prog, net)
            exp = EXPECTED[net]
            if key not in by_key:
                out.append(
                    {
                        "program": prog,
                        "network": net,
                        "dataset": exp["dataset"],
                        "expected_batch_size": exp["batch_size"],
                        "batch_match": False,
                        "eps": exp["eps"],
                        "status": "missing",
                        "dedup_note": "",
                        "source_file": "",
                    }
                )
                continue

            candidates = by_key[key]
            best = max(candidates, key=_row_score)
            note = ""
            if len(candidates) > 1:
                alts = sorted(
                    {
                        (c["source_file"], c.get("batch_size"), c.get("status"))
                        for c in candidates
                    }
                )
                note = (
                    f"{len(candidates)} entries {alts}; "
                    f"kept {best['source_file']} batch={best.get('batch_size')} {best['status']}"
                )
                if not any(c.get("batch_size") == exp["batch_size"] for c in candidates):
                    note += f" (no run with expected batch={exp['batch_size']})"

            row = dict(best)
            row["expected_batch_size"] = exp["batch_size"]
            row["batch_match"] = row.get("batch_size") == exp["batch_size"]
            row["dedup_note"] = note
            out.append(row)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def _fmt(v, digits=3) -> str:
    if v is None or v == "":
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# Consolidated sparse matmul experiments",
        "",
        "Merged from `final_results/results*.json`. "
        "Duplicates resolved by preferring **expected batch size**, then **ok** status, then rows with metrics.",
        "",
        "**Expected batch sizes:** n10/n11/n14/n22 → 100; n30/n32 → 10.",
        "",
        "## Multiply time vs block sparse (ratio)",
        "",
        "| Program | Network | Batch | Match | Status | Block (s) | CSR mul× | COO mul× | BSR 8×8 mul× | BSR dyn mul× |",
        "|---------|---------|------:|:-----:|--------|----------:|---------:|---------:|-------------:|-------------:|",
    ]
    for r in rows:
        prog = r["program"].replace(".cf", "")
        bs = r.get("batch_size", "")
        match = "✓" if r.get("batch_match") else "**≠**"
        lines.append(
            f"| {prog} | {r['network']} | {bs} | {match} | {r.get('status', '')} | "
            f"{_fmt(r.get('block_sparse_s'))} | "
            f"{_fmt(r.get('csr_mul_vs_block'))} | "
            f"{_fmt(r.get('coo_mul_vs_block'))} | "
            f"{_fmt(r.get('bsr_fixed_mul_vs_block'))} | "
            f"{_fmt(r.get('bsr_dynamic_mul_vs_block'))} |"
        )

    mism = [r for r in rows if r.get("status") == "ok" and not r.get("batch_match")]
    if mism:
        lines.extend(["", "## Batch size ≠ expected (included in table above)", ""])
        for r in mism:
            lines.append(
                f"- **{r['program']} × {r['network']}**: ran batch={r.get('batch_size')}, "
                f"expected {r.get('expected_batch_size')}"
            )

    notes = [r for r in rows if r.get("dedup_note")]
    if notes:
        lines.extend(["", "## Dedup notes", ""])
        for r in notes:
            lines.append(f"- **{r['program']} × {r['network']}**: {r['dedup_note']}")

    failed = [r for r in rows if r.get("status") not in ("ok", "missing")]
    if failed:
        lines.extend(["", "## Failed / timeout", ""])
        for r in failed:
            lines.append(f"- {r['program']} × {r['network']}: {r['status']}")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=REPO_ROOT / "final_results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "final_results",
    )
    args = parser.parse_args()
    rows = consolidate(load_all(args.input_dir))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "consolidated.csv", rows)
    write_markdown(args.out_dir / "consolidated.md", rows)
    print(f"Wrote {args.out_dir / 'consolidated.csv'} ({len(rows)} rows)")
    print(f"Wrote {args.out_dir / 'consolidated.md'}")


if __name__ == "__main__":
    main()
