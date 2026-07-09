import re
import subprocess
import sys
import ast
from typing import Tuple

import torch
import typer

app = typer.Typer(help="ConstraintFlow JIT test")


def _extract_bound(output: str, label: str) -> str:
    # Bounds can span multiple lines, so use DOTALL and non-greedy capture.
    match = re.search(rf"{label}:\s*(tensor\(.*?\))", output, re.DOTALL)
    if not match:
        raise ValueError(f"Could not find '{label}' in CLI output.")
    return match.group(1)


def _parse_tensor(tensor_text: str) -> torch.Tensor:
    # Parse CLI output like: tensor([[...]], dtype=torch.float64) without eval.
    if not tensor_text.startswith("tensor(") or not tensor_text.endswith(")"):
        raise ValueError(f"Unexpected tensor format: {tensor_text}")

    inner = tensor_text[len("tensor("):-1]
    dtype_match = re.search(r",\s*dtype=torch\.([a-zA-Z0-9_]+)\s*$", inner, re.DOTALL)

    dtype = None
    if dtype_match:
        dtype_name = dtype_match.group(1)
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            raise ValueError(f"Unsupported dtype in output: torch.{dtype_name}")
        data_text = inner[:dtype_match.start()]
    else:
        data_text = inner

    data = ast.literal_eval(data_text.strip())
    return torch.tensor(data, dtype=dtype) if dtype is not None else torch.tensor(data)


def _extract_peak_memory(output: str):
    # cli.py prints "Peak GPU memory: <int> bytes" on CUDA and
    # "Peak CPU memory: <int> bytes" (process peak RSS) on CPU. The compile-only
    # paths (`compile`, `jit`) print "Peak CPU memory" (host RSS) too.
    match = re.search(r"Peak (?:GPU|CPU) memory:\s*([0-9]+)\s*bytes", output)
    if not match:
        return None
    return int(match.group(1))


def _extract_total_time(output: str):
    # cli.py prints "Total time: <float> seconds" from run, compile, and jit.
    match = re.search(r"Total time:\s*([0-9.]+)\s*seconds", output)
    if not match:
        return None
    return float(match.group(1))


def _fail(cmd: list[str], result: subprocess.CompletedProcess) -> RuntimeError:
    return RuntimeError(
        "CLI command failed.\n"
        f"Command: {' '.join(cmd)}\n"
        f"Exit code: {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


def _run_cli(program_file: str, network: str, dataset: str, extra_args: list[str], compile=True, device: str = "cpu"):
    cmd = [
        sys.executable,
        "constraintflow/cli.py",
        "run",
        program_file,
        "--network",
        network,
        "--dataset",
        dataset,
        "--device",
        device,
        *extra_args,
    ]

    if compile:
        cmd.append("--compile")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise _fail(cmd, result)
    print(result.stdout)
    peak_mem = _extract_peak_memory(result.stdout)
    lower = _parse_tensor(_extract_bound(result.stdout, "Lower bounds"))
    upper = _parse_tensor(_extract_bound(result.stdout, "Upper bounds"))
    return lower, upper, peak_mem


def _run_jit(program_file: str, network: str, dataset: str, extra_args: list[str], device: str = "cpu"):
    """Run the whole simulacrum+reuse pipeline in one compile pass via `jit`.
    """
    cmd = [
        sys.executable,
        "constraintflow/cli.py",
        "jit",
        program_file,
        "--network",
        network,
        "--dataset",
        dataset,
        "--device",
        device,
        *extra_args,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise _fail(cmd, result)
    print(result.stdout)


def _run_and_capture(cli_args: list[str]) -> str:
    """Invoke `constraintflow/cli.py <cli_args>` and return its stdout.

    Used by profiling to drive the `compile`, `jit`, and `run` subcommands and
    parse the uniform 'Total time' / 'Peak ... memory' lines each of them prints.
    """
    cmd = [sys.executable, "constraintflow/cli.py", *cli_args]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise _fail(cmd, result)
    print(result.stdout)
    return result.stdout


def _assert_bounds_close(lhs: torch.Tensor, rhs: torch.Tensor, name: str) -> None:
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{name} shape mismatch: {lhs.shape} != {rhs.shape}")
    if not torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"{name} mismatch.\nLeft:\n{lhs}\nRight:\n{rhs}")


def _bounds_match(lb_a, ub_a, lb_b, ub_b) -> bool:
    try:
        _assert_bounds_close(lb_a, lb_b, "Lower bounds")
        _assert_bounds_close(ub_a, ub_b, "Upper bounds")
        return True
    except AssertionError:
        return False


def _format_mem(mem) -> str:
    if mem is None:
        return "N/A"
    return f"{mem / (1024 * 1024):.2f} MB"


def _format_time(t) -> str:
    if t is None:
        return "N/A"
    return f"{t:.2f} s"


def _format_ratio(num, den) -> str:
    if num is None or den is None or den == 0:
        return "N/A"
    return f"{num / den:.2f}x"


def _avg(values):
    """Mean of the non-None measurements, or None if there are none."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _print_profile_table(rows: list[dict], device: str, repeat: int) -> None:
    headers = [
        "Program", "Batch", "Eps", "Phase",
        "Normal Time", "JIT Time", "Speedup (N/J)",
        "Normal Mem", "JIT Mem", "Mem Ovhd (J/N)",
        "Match?",
    ]
    table_rows = [
        [
            r["program"],
            str(r["batch"]),
            str(r["eps"]),
            r["phase"],
            _format_time(r["n_time"]),
            _format_time(r["j_time"]),
            _format_ratio(r["n_time"], r["j_time"]),
            _format_mem(r["n_mem"]),
            _format_mem(r["j_mem"]),
            _format_ratio(r["j_mem"], r["n_mem"]),
            r["match"],
        ]
        for r in rows
    ]

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(cells):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)
    print()
    print(f"Profiling: Normal vs JIT compile & run (time and peak memory), device={device}, averaged over {repeat} run(s)")
    print("Note: run memory is the device peak (CPU process RSS or GPU CUDA); compile memory is host process RSS.")
    print(fmt_row(headers))
    print(sep)
    for row in table_rows:
        print(fmt_row(row))
    print()


def _run_profile_mode(program_files: str, network: str, dataset: str, device: str, repeat: int, in_memory: bool = False) -> None:
    jit_extra = ["--in-memory"] if in_memory else []
    epss = [0]
    batch_sizes = [1]
    rows = []
    for program_file in program_files.split(","):
        program_path = "examples/compiler_examples/" + program_file
        print(f"Profiling program file: {program_path} with network: {network}, dataset: {dataset}, device: {device}, repeat: {repeat}")
        for batch_size in batch_sizes:
            for eps in epss:
                run_args = [
                    program_path,
                    "--network", network,
                    "--dataset", dataset,
                    "--device", device,
                    "--eps", str(eps),
                    "--batch-size", str(batch_size),
                ]


                nc_t, nc_m, nr_t, nr_m = [], [], [], []
                jc_t, jc_m, jr_t, jr_m = [], [], [], []
                match = None
                for _ in range(repeat):
                    # --- Normal path: compile (device-agnostic codegen), then run
                    out = _run_and_capture(["compile", program_path, "--output-path", "output/"])
                    nc_t.append(_extract_total_time(out)); nc_m.append(_extract_peak_memory(out))

                    out = _run_and_capture(["run", *run_args])
                    nr_t.append(_extract_total_time(out)); nr_m.append(_extract_peak_memory(out))
                    n_lb = _parse_tensor(_extract_bound(out, "Lower bounds"))
                    n_ub = _parse_tensor(_extract_bound(out, "Upper bounds"))

                    # --- JIT path: one compile pass (probe + reuse compile,
                    out = _run_and_capture(["jit", *run_args, *jit_extra])
                    jc_t.append(_extract_total_time(out)); jc_m.append(_extract_peak_memory(out))

                    out = _run_and_capture(["run", *run_args])
                    jr_t.append(_extract_total_time(out)); jr_m.append(_extract_peak_memory(out))
                    j_lb = _parse_tensor(_extract_bound(out, "Lower bounds"))
                    j_ub = _parse_tensor(_extract_bound(out, "Upper bounds"))

                    # Bounds are deterministic, so decide the match once.
                    if match is None:
                        match = "Yes" if _bounds_match(n_lb, n_ub, j_lb, j_ub) else "No"

                base = {"program": program_file, "batch": batch_size, "eps": eps}
                # One row per phase, Normal and JIT side by side (time/memory are
                # the mean over `repeat` runs). Match only applies to the Run row.
                rows.append({**base, "phase": "Compile",
                             "n_time": _avg(nc_t), "j_time": _avg(jc_t),
                             "n_mem": _avg(nc_m), "j_mem": _avg(jc_m), "match": ""})
                rows.append({**base, "phase": "Run",
                             "n_time": _avg(nr_t), "j_time": _avg(jr_t),
                             "n_mem": _avg(nr_m), "j_mem": _avg(jr_m), "match": match})

    _print_profile_table(rows, device, repeat)


def _echo_context(program_file: str, network: str, dataset: str, device: str, batch_size: int, eps: float) -> None:
    typer.echo(f"Program file: {program_file}")
    typer.echo(f"Network: {network}")
    typer.echo(f"Dataset: {dataset}")
    typer.echo(f"Device: {device}")
    typer.echo(f"Batch size: {batch_size}")
    typer.echo(f"Eps: {eps}")


@app.command()
def test(
    program_files: str = typer.Argument(..., help="Comma-separated list of ConstraintFlow program files to test"),
    network: str = typer.Argument(..., help="Network path/name"),
    dataset: str = "mnist",
    device: str = typer.Option("cpu", help="Device to run on: cpu, gpu (CUDA), or gpumac (Apple MPS)."),
    profile: bool = typer.Option(False, "--profile", help="Profile compile & run time and peak memory for the Normal and JIT paths and print a table."),
    repeat: int = typer.Option(1, help="In --profile mode, run each configuration this many times and average the time and memory."),
    in_memory: bool = typer.Option(False, "--in-memory", help="Run the jit compile with --in-memory (keep captures in a process-local dict instead of on disk)."),
):
    if profile:
        _run_profile_mode(program_files, network, dataset, device, repeat, in_memory)
        return

    jit_extra = ["--in-memory"] if in_memory else []

    for program_file in program_files.split(","):
        program_file = "examples/compiler_examples/" + program_file
        print(f"Testing program file: {program_file} with network: {network}, dataset: {dataset}, device: {device}")
        epss = [0, 0.001]
        batch_sizes = [1, 2]
        for batch_size in batch_sizes:
            batch_args = ["--batch-size", str(batch_size)]

            # --- Baseline: normal compile + run for each eps. ---
            baseline_lbs, baseline_ubs = [], []
            for eps in epss:
                try:
                    baseline_lb, baseline_ub, _ = _run_cli(
                        program_file, network, dataset, ["--eps", str(eps), *batch_args], device=device
                    )
                except (AssertionError, RuntimeError) as e:
                    typer.echo("Baseline run failed")
                    _echo_context(program_file, network, dataset, device, batch_size, eps)
                    raise e
                baseline_lbs.append(baseline_lb)
                baseline_ubs.append(baseline_ub)

            # --- Simulacrum+reuse: ONE compile pass. 
            try:
                _run_jit(program_file, network, dataset, ["--eps", str(epss[0]), *batch_args, *jit_extra], device=device)
            except (AssertionError, RuntimeError) as e:
                typer.echo("Simulacrum (jit) compile pass failed")
                _echo_context(program_file, network, dataset, device, batch_size, epss[0])
                raise e

            reuse_lbs, reuse_ubs = [], []
            for eps in epss:
                try:
                    reuse_lb, reuse_ub, _ = _run_cli(
                        program_file, network, dataset, ["--eps", str(eps), *batch_args], compile=False, device=device
                    )
                except (AssertionError, RuntimeError) as e:
                    typer.echo("Reuse run failed")
                    _echo_context(program_file, network, dataset, device, batch_size, eps)
                    raise e
                reuse_lbs.append(reuse_lb)
                reuse_ubs.append(reuse_ub)

            # --- Compare baseline vs reuse bounds for each eps. ---
            for eps, baseline_lb, baseline_ub, reuse_lb, reuse_ub in zip(
                epss, baseline_lbs, baseline_ubs, reuse_lbs, reuse_ubs
            ):
                try:
                    _assert_bounds_close(baseline_lb, reuse_lb, "Lower bounds baseline vs reuse")
                    _assert_bounds_close(baseline_ub, reuse_ub, "Upper bounds baseline vs reuse")
                except AssertionError as e:
                    typer.echo(f"AssertionError: {e}")
                    typer.echo(f"Baseline LB: {baseline_lb}")
                    typer.echo(f"Reuse LB: {reuse_lb}")
                    typer.echo(f"Baseline UB: {baseline_ub}")
                    typer.echo(f"Reuse UB: {reuse_ub}")
                    _echo_context(program_file, network, dataset, device, batch_size, eps)
                    raise e

                typer.echo(f"JIT test passed: baseline and jit modes bounds match for eps={eps} and batch_size={batch_size} on {device}.")

    print("JIT test passed: all eps and batch sizes passed.")


if __name__ == "__main__":
    app()
