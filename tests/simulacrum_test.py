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
    # cli.py prints "Peak GPU memory: <int> bytes" only when running on CUDA.
    match = re.search(r"Peak GPU memory:\s*([0-9]+)\s*bytes", output)
    if not match:
        return None
    return int(match.group(1))


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
        raise RuntimeError(
            "CLI run failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"Exit code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    print(result.stdout)
    peak_mem = _extract_peak_memory(result.stdout)
    if "--simulacrum" in extra_args:
        return None, None, peak_mem
    lower = _parse_tensor(_extract_bound(result.stdout, "Lower bounds"))
    upper = _parse_tensor(_extract_bound(result.stdout, "Upper bounds"))
    return lower, upper, peak_mem


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


def _print_memory_table(rows: list[dict]) -> None:
    headers = ["Program", "Batch", "Eps", "Normal Compilation", "Simulacrum+GPU", "Match?"]
    table_rows = [
        [
            r["program"],
            str(r["batch"]),
            str(r["eps"]),
            _format_mem(r["normal_mem"]),
            _format_mem(r["simulacrum_mem"]),
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
    print("Peak GPU memory comparison")
    print(fmt_row(headers))
    print(sep)
    for row in table_rows:
        print(fmt_row(row))
    print()


def _run_memory_mode(program_files: str, network: str, dataset: str) -> None:
    epss = [0, 0.001]
    batch_sizes = [1, 2]
    rows = []
    for program_file in program_files.split(","):
        program_path = "examples/compiler_examples/" + program_file
        print(f"Measuring program file: {program_path} with network: {network} and dataset: {dataset}")
        for batch_size in batch_sizes:
            for eps in epss:
                common = ["--eps", str(eps), "--batch-size", str(batch_size)]
                # Normal Compilation: standard compiled run on GPU.
                normal_lb, normal_ub, normal_mem = _run_cli(
                    program_path, network, dataset, common, compile=True, device="gpu"
                )
                # Simulacrum+GPU: capture dummy blocks, then reuse the stored plan on GPU.
                _run_cli(
                    program_path, network, dataset, ["--simulacrum", *common], compile=True, device="gpu"
                )
                reuse_lb, reuse_ub, reuse_mem = _run_cli(
                    program_path, network, dataset, ["--reuse", *common], compile=True, device="gpu"
                )

                match = _bounds_match(normal_lb, normal_ub, reuse_lb, reuse_ub)
                rows.append({
                    "program": program_file,
                    "batch": batch_size,
                    "eps": eps,
                    "normal_mem": normal_mem,
                    "simulacrum_mem": reuse_mem,
                    "match": "Yes" if match else "No",
                })

    _print_memory_table(rows)


@app.command()
def test(
    program_files: str = typer.Argument(..., help="Comma-separated list of ConstraintFlow program files to test"),
    network: str = typer.Argument(..., help="Network path/name"),
    dataset: str = "mnist",
    memory: bool = typer.Option(False, "--memory", help="Measure peak GPU memory (CUDA) for Normal Compilation vs Simulacrum+GPU and print a comparison table."),
):
    if memory:
        _run_memory_mode(program_files, network, dataset)
        return

    for program_file in program_files.split(","):
        program_file = "examples/compiler_examples/" + program_file
        print(f"Testing program file: {program_file} with network: {network} and dataset: {dataset}")
        epss = [0, 0.001]
        batch_sizes = [1, 2]
        for batch_size in batch_sizes:
            baseline_lbs, baseline_ubs = [], []
            for eps in epss:
                try:
                    baseline_lb, baseline_ub, _ = _run_cli(program_file, network, dataset, ["--eps", str(eps), "--batch-size", str(batch_size)])
                except AssertionError as e:
                    typer.echo(f"Program File: {program_file}")
                    typer.echo(f"Program File: {program_file}")
                    typer.echo(f"Baseline run failed")
                    typer.echo(f"AssertionError: {e}")
                    typer.echo(f"Program file: {program_file}")
                    typer.echo(f"Network: {network}")
                    typer.echo(f"Dataset: {dataset}")
                    typer.echo(f"Batch size: {batch_size}")
                    typer.echo(f"Eps: {eps}")
                    raise e
                baseline_lbs.append(baseline_lb)
                baseline_ubs.append(baseline_ub)
            try:
                simulacrum_lb, simulacrum_ub, _ = _run_cli(program_file, network, dataset, ["--simulacrum", "--eps", str(eps), "--batch-size", str(batch_size)])
            except AssertionError as e:
                typer.echo(f"Program File: {program_file}")
                typer.echo(f"Simulacrum run failed")
                typer.echo(f"AssertionError: {e}")
                typer.echo(f"Program file: {program_file}")
                typer.echo(f"Network: {network}")
                typer.echo(f"Dataset: {dataset}")
                typer.echo(f"Batch size: {batch_size}")
                typer.echo(f"Eps: {eps}")
                raise e
            reuse_lbs, reuse_ubs = [], []
            for i, eps in enumerate(epss):
                if i==0:
                    try:
                        reuse_lb, reuse_ub, _ = _run_cli(program_file, network, dataset, ["--reuse", "--eps", str(eps), "--batch-size", str(batch_size)])
                    except AssertionError as e:
                        typer.echo(f"Program File: {program_file}")
                        typer.echo(f"Reuse run failed")
                        typer.echo(f"AssertionError: {e}")
                        typer.echo(f"Program file: {program_file}")
                        typer.echo(f"Network: {network}")
                        typer.echo(f"Dataset: {dataset}")
                        typer.echo(f"Batch size: {batch_size}")
                        typer.echo(f"Eps: {eps}")
                        raise e
                else:
                    try:
                        reuse_lb, reuse_ub, _ = _run_cli(program_file, network, dataset, ["--eps", str(eps), "--batch-size", str(batch_size)], compile=False)
                    except AssertionError as e:
                        typer.echo(f"Program File: {program_file}")
                        typer.echo(f"Reuse run failed")
                        typer.echo(f"AssertionError: {e}")
                        typer.echo(f"Program file: {program_file}")
                        typer.echo(f"Network: {network}")
                        typer.echo(f"Dataset: {dataset}")
                        typer.echo(f"Batch size: {batch_size}")
                        typer.echo(f"Eps: {eps}")
                        raise e
                reuse_lbs.append(reuse_lb)
                reuse_ubs.append(reuse_ub)

            for baseline_lb, baseline_ub, reuse_lb, reuse_ub in zip(baseline_lbs, baseline_ubs, reuse_lbs, reuse_ubs):
                try:
                    _assert_bounds_close(baseline_lb, reuse_lb, "Lower bounds baseline vs reuse")
                    _assert_bounds_close(baseline_ub, reuse_ub, "Upper bounds baseline vs reuse")
                except AssertionError as e:
                    typer.echo(f"Program File: {program_file}")
                    typer.echo(f"AssertionError: {e}")
                    typer.echo(f"Baseline LB: {baseline_lb}")
                    typer.echo(f"Reuse LB: {reuse_lb}")
                    typer.echo(f"Baseline UB: {baseline_ub}")
                    typer.echo(f"Reuse UB: {reuse_ub}")
                    typer.echo(f"Program file: {program_file}")
                    typer.echo(f"Network: {network}")
                    typer.echo(f"Dataset: {dataset}")
                    typer.echo(f"Batch size: {batch_size}")
                    typer.echo(f"Eps: {eps}")
                    raise e

                typer.echo(f"JIT test passed: baseline and jit modes bounds match for eps={eps} and batch_size={batch_size}.")
    
    print("JIT test passed: all eps and batch sizes passed.")


if __name__ == "__main__":
    app()