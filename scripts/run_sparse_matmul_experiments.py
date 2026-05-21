#!/usr/bin/env python3
"""
Run sparse-format matmul benchmarks across ConstraintFlow programs and networks.

Each run invokes `constraintflow run ... --compile` with CONSTRAINTFLOW_TIMING_JSON=1
and records block-sparse baseline plus CSR / COO / BSR-fixed / BSR-dynamic encode & mul.

Usage:
  # Full grid (8 programs × 6 networks = 48 runs; can take many hours)
  python scripts/run_sparse_matmul_experiments.py --nets-dir ~/nets

  # Smoke test (one pair)
  python scripts/run_sparse_matmul_experiments.py \\
    --programs ibp.cf --networks n30 --timeout 600

  # Preview commands without running
  python scripts/run_sparse_matmul_experiments.py --dry-run --programs ibp.cf --networks n10

  # Resume after interruption
  python scripts/run_sparse_matmul_experiments.py --resume results/experiments/20260521T120000Z

Networks (ONNX in --nets-dir, default ~/nets):
  n10  eps=0.005   batch=100  mnist
  n11  eps=0.004   batch=100  mnist
  n14  eps=0.1     batch=100  mnist
  n22  eps=4e-5    batch=100  cifar
  n30  eps=5e-4    batch=10   cifar
  n32  eps=0.002   batch=10   cifar

Programs live under examples/compiler_examples/ (deeppoly_efficient.cf, not .py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRAMS_DIR = REPO_ROOT / "examples" / "compiler_examples"

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

NETWORKS: dict[str, dict[str, Any]] = {
    "n10": {"eps": 0.005, "batch_size": 100, "dataset": "mnist"},
    "n11": {"eps": 0.004, "batch_size": 100, "dataset": "mnist"},
    "n14": {"eps": 0.1, "batch_size": 100, "dataset": "mnist"},
    "n22": {"eps": 4e-5, "batch_size": 100, "dataset": "cifar"},
    "n30": {"eps": 5e-4, "batch_size": 10, "dataset": "cifar"},
    "n32": {"eps": 0.002, "batch_size": 10, "dataset": "cifar"},
}

TIMING_PREFIX = "CONSTRAINTFLOW_TIMING_JSON="

# Fallback regex when JSON line is missing (matches current CLI human output).
PARSE_PATTERNS = {
    "block_sparse_s": re.compile(
        r"block sparse \(SparseTensor\.matmul\):\s+([0-9.]+)\s+s\s+\((\d+)\s+calls\)"
    ),
    "csr_mul_s": re.compile(r"CSR multiply:\s+([0-9.]+)\s+s"),
    "coo_mul_s": re.compile(r"COO multiply:\s+([0-9.]+)\s+s"),
    "bsr_fixed_mul_s": re.compile(r"BSR multiply \(8x8, via COO\):\s+([0-9.]+)\s+s"),
    "bsr_dynamic_mul_s": re.compile(
        r"BSR multiply \(derived, via COO\):\s+([0-9.]+)\s+s"
    ),
    "wall_time_s": re.compile(r"Total time:\s+([0-9.]+)\s+seconds"),
}


@dataclass
class RunSpec:
    program: str
    network: str
    eps: float
    batch_size: int
    dataset: str


CSV_FIELDS = [
    "program",
    "network",
    "dataset",
    "eps",
    "batch_size",
    "status",
    "wall_time_s",
    "block_sparse_s",
    "block_sparse_calls",
    "csr_encode_s",
    "csr_mul_s",
    "csr_mul_calls",
    "coo_encode_s",
    "coo_mul_s",
    "coo_mul_calls",
    "bsr_fixed_encode_s",
    "bsr_fixed_mul_s",
    "bsr_fixed_mul_calls",
    "bsr_dynamic_encode_s",
    "bsr_dynamic_mul_s",
    "bsr_dynamic_mul_calls",
    "verify_s",
    "csr_mul_vs_block",
    "coo_mul_vs_block",
    "bsr_fixed_mul_vs_block",
    "bsr_dynamic_mul_vs_block",
    "error",
    "output_dir",
]


def _resolve_network_path(nets_dir: Path, name: str) -> Path:
    for ext in (".onnx", ".pt", ".pth", ""):
        candidate = nets_dir / f"{name}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Network file for '{name}' not found under {nets_dir}")


def _flatten_timing(timing: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "wall_time_s": timing.get("wall_time_s"),
        "block_sparse_s": timing.get("block_sparse", {}).get("seconds"),
        "block_sparse_calls": timing.get("block_sparse", {}).get("calls"),
        "csr_encode_s": timing.get("csr", {}).get("encode", {}).get("seconds"),
        "csr_mul_s": timing.get("csr", {}).get("mul", {}).get("seconds"),
        "csr_mul_calls": timing.get("csr", {}).get("mul", {}).get("calls"),
        "coo_encode_s": timing.get("coo", {}).get("encode", {}).get("seconds"),
        "coo_mul_s": timing.get("coo", {}).get("mul", {}).get("seconds"),
        "coo_mul_calls": timing.get("coo", {}).get("mul", {}).get("calls"),
        "bsr_fixed_encode_s": timing.get("bsr_fixed", {}).get("encode", {}).get("seconds"),
        "bsr_fixed_mul_s": timing.get("bsr_fixed", {}).get("mul", {}).get("seconds"),
        "bsr_fixed_mul_calls": timing.get("bsr_fixed", {}).get("mul", {}).get("calls"),
        "bsr_dynamic_encode_s": timing.get("bsr_dynamic", {}).get("encode", {}).get("seconds"),
        "bsr_dynamic_mul_s": timing.get("bsr_dynamic", {}).get("mul", {}).get("seconds"),
        "bsr_dynamic_mul_calls": timing.get("bsr_dynamic", {}).get("mul", {}).get("calls"),
        "verify_s": timing.get("verify", {}).get("seconds"),
    }
    base = row.get("block_sparse_s") or 0.0
    if base > 0:
        for key, mul_key in (
            ("csr_mul_vs_block", "csr_mul_s"),
            ("coo_mul_vs_block", "coo_mul_s"),
            ("bsr_fixed_mul_vs_block", "bsr_fixed_mul_s"),
            ("bsr_dynamic_mul_vs_block", "bsr_dynamic_mul_s"),
        ):
            mul = row.get(mul_key)
            row[key] = (mul / base) if mul is not None else None
    return row


def _parse_timing_from_stdout(stdout: str) -> dict[str, Any] | None:
    for line in stdout.splitlines():
        if line.startswith(TIMING_PREFIX):
            return json.loads(line[len(TIMING_PREFIX) :])
    return None


def _parse_timing_fallback(stdout: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, pattern in PARSE_PATTERNS.items():
        m = pattern.search(stdout)
        if m:
            out[key] = float(m.group(1))
            if key == "block_sparse_s" and m.lastindex and m.lastindex >= 2:
                out["block_sparse_calls"] = int(m.group(2))
    if out.get("block_sparse_s"):
        base = out["block_sparse_s"]
        for fmt, col in (
            ("csr_mul_s", "csr_mul_vs_block"),
            ("coo_mul_s", "coo_mul_vs_block"),
            ("bsr_fixed_mul_s", "bsr_fixed_mul_vs_block"),
            ("bsr_dynamic_mul_s", "bsr_dynamic_mul_vs_block"),
        ):
            if fmt in out:
                out[col] = out[fmt] / base
    return out


def run_one(
    spec: RunSpec,
    nets_dir: Path,
    out_base: Path,
    timeout_s: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    program_path = PROGRAMS_DIR / spec.program
    if not program_path.is_file():
        raise FileNotFoundError(f"Program not found: {program_path}")

    network_path = _resolve_network_path(nets_dir, spec.network)
    run_id = f"{Path(spec.program).stem}_{spec.network}"
    output_dir = out_base / run_id
    log_path = output_dir / "run.log"

    row: dict[str, Any] = {
        "program": spec.program,
        "network": spec.network,
        "dataset": spec.dataset,
        "eps": spec.eps,
        "batch_size": spec.batch_size,
        "status": "pending",
        "output_dir": str(output_dir),
        "error": "",
    }

    cmd = [
        "constraintflow",
        "run",
        str(program_path),
        "--network",
        str(network_path),
        "--network-format",
        "onnx",
        "--dataset",
        spec.dataset,
        "--eps",
        str(spec.eps),
        "--batch-size",
        str(spec.batch_size),
        "--compile",
        "--output-path",
        str(output_dir),
    ]

    if dry_run:
        row["status"] = "dry_run"
        row["error"] = " ".join(cmd)
        return row

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CONSTRAINTFLOW_TIMING_JSON"] = "1"

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        wall = time.perf_counter() - started
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        log_path.write_text(stdout + ("\n--- stderr ---\n" if stderr else "") + stderr)

        if proc.returncode != 0:
            row["status"] = "failed"
            row["error"] = (stderr or stdout)[-2000:]
            row["wall_time_s"] = wall
            return row

        timing = _parse_timing_from_stdout(stdout)
        if timing is None:
            flat = _parse_timing_fallback(stdout)
            if not flat:
                row["status"] = "failed"
                row["error"] = "No timing data in output (set CONSTRAINTFLOW_TIMING_JSON=1)"
                row["wall_time_s"] = wall
                return row
            row.update(flat)
        else:
            row.update(_flatten_timing(timing))
            if row.get("wall_time_s") is None:
                row["wall_time_s"] = wall

        row["status"] = "ok"
        return row

    except subprocess.TimeoutExpired as e:
        row["status"] = "timeout"
        row["error"] = f"Exceeded {timeout_s}s"
        if e.stdout:
            log_path.write_text(e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout)
        row["wall_time_s"] = time.perf_counter() - started
        return row
    except Exception as e:
        row["status"] = "failed"
        row["error"] = str(e)
        row["wall_time_s"] = time.perf_counter() - started
        return row


def iter_specs(
    programs: list[str],
    networks: list[str],
) -> list[RunSpec]:
    specs = []
    for program in programs:
        for network in networks:
            cfg = NETWORKS[network]
            specs.append(
                RunSpec(
                    program=program,
                    network=network,
                    eps=cfg["eps"],
                    batch_size=cfg["batch_size"],
                    dataset=cfg["dataset"],
                )
            )
    return specs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nets-dir",
        type=Path,
        default=Path(os.environ.get("CONSTRAINTFLOW_NETS_DIR", Path.home() / "nets")),
        help="Directory containing n10.onnx, n11.onnx, ...",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "results" / "experiments",
        help="Base output directory (timestamped subfolder created)",
    )
    parser.add_argument(
        "--programs",
        nargs="*",
        default=PROGRAMS,
        help="Subset of .cf programs (default: all 8)",
    )
    parser.add_argument(
        "--networks",
        nargs="*",
        default=list(NETWORKS.keys()),
        help="Subset of networks (default: n10..n32)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Per-run timeout in seconds (default 2h)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume into existing results dir (skip ok runs in results.csv)",
    )
    args = parser.parse_args()

    missing = [p for p in args.programs if not (PROGRAMS_DIR / p).is_file()]
    if missing:
        print(f"Missing programs under {PROGRAMS_DIR}: {missing}", file=sys.stderr)
        return 1

    if args.resume:
        out_base = args.resume
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_base = args.out_dir / stamp
    out_base.mkdir(parents=True, exist_ok=True)

    specs = iter_specs(args.programs, args.networks)
    results_path = out_base / "results.csv"
    json_path = out_base / "results.json"

    done: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    if results_path.is_file():
        with results_path.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
                if r.get("status") == "ok":
                    done.add((r["program"], r["network"]))

    total = len(specs)
    for i, spec in enumerate(specs, 1):
        key = (spec.program, spec.network)
        if key in done:
            print(f"[{i}/{total}] skip {spec.program} × {spec.network} (already ok)")
            continue
        print(f"[{i}/{total}] {spec.program} × {spec.network} "
              f"(eps={spec.eps}, batch={spec.batch_size}, {spec.dataset})")
        row = run_one(spec, args.nets_dir, out_base, args.timeout, args.dry_run)
        rows.append(row)
        write_csv(results_path, rows)
        json_path.write_text(json.dumps(rows, indent=2))
        print(f"  -> {row['status']}")

    if args.dry_run:
        print(f"\nDry-run: {len(specs)} command(s). Wrote {results_path}")
        return 0

    ok = sum(1 for r in rows if r.get("status") == "ok")
    failed = sum(1 for r in rows if r.get("status") in ("failed", "timeout"))
    print(f"\nFinished: {ok}/{len(specs)} ok, {failed} failed. Results: {results_path}")
    if failed:
        return 1
    return 0 if ok == len(specs) else 1


if __name__ == "__main__":
    sys.exit(main())
