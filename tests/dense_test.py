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


def _run_cli(program_file: str, network: str, dataset: str, extra_args: list[str], compile=True) -> Tuple[torch.Tensor, torch.Tensor]:
    cmd = [
        sys.executable,
        "constraintflow/cli.py",
        "run",
        program_file,
        "--network",
        network,
        "--dataset",
        dataset,
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
    if "--simulacrum" in extra_args:
        return None, None
    lower = _parse_tensor(_extract_bound(result.stdout, "Lower bounds"))
    upper = _parse_tensor(_extract_bound(result.stdout, "Upper bounds"))
    return lower, upper


def _assert_bounds_close(lhs: torch.Tensor, rhs: torch.Tensor, name: str) -> None:
    if lhs.shape != rhs.shape:
        raise AssertionError(f"{name} shape mismatch: {lhs.shape} != {rhs.shape}")
    if not torch.allclose(lhs, rhs, atol=1e-6, rtol=1e-6):
        raise AssertionError(f"{name} mismatch.\nLeft:\n{lhs}\nRight:\n{rhs}")


@app.command()
def test(
    program_files: str = typer.Argument(..., help="Comma-separated list of ConstraintFlow program files to test"),
    network: str = typer.Argument(..., help="Network path/name"),
    dataset: str = "mnist",
):
    for program_file in program_files.split(","):
        program_file = "examples/compiler_examples/" + program_file
        print(f"Testing program file: {program_file} with network: {network} and dataset: {dataset}")
        epss = [0, 0.001]
        batch_sizes = [1, 2]
        for batch_size in batch_sizes:
            baseline_lbs, baseline_ubs = [], []
            for eps in epss:
                try:
                    baseline_lb, baseline_ub = _run_cli(program_file, network, dataset, ["--eps", str(eps), "--batch-size", str(batch_size)])
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
            
            reuse_lbs, reuse_ubs = [], []
            for eps in epss:
                try:
                    reuse_lb, reuse_ub = _run_cli(program_file, network, dataset, ["--dense", "--eps", str(eps), "--batch-size", str(batch_size)], compile=False)
                except AssertionError as e:
                    typer.echo(f"Program File: {program_file}")
                    typer.echo(f"Dense run failed")
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