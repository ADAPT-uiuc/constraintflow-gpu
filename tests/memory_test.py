import re
import subprocess
import sys

import typer

app = typer.Typer(help="ConstraintFlow normal-compilation peak GPU memory measurement")


def _extract_peak_memory(output: str):
    # cli.py prints "Peak GPU memory: <int> bytes" only when running on CUDA.
    match = re.search(r"Peak GPU memory:\s*([0-9]+)\s*bytes", output)
    if not match:
        return None
    return int(match.group(1))


def _run_cli(program_file: str, network: str, dataset: str, extra_args: list[str], compile=True, device: str = "gpu"):
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
    return _extract_peak_memory(result.stdout)


def _format_mem(mem) -> str:
    if mem is None:
        return "N/A"
    return f"{mem / (1024 * 1024):.2f} MB"


def _print_memory_table(rows: list[dict]) -> None:
    headers = ["Program", "Batch", "Eps", "Normal Compilation"]
    table_rows = [
        [r["program"], str(r["batch"]), str(r["eps"]), _format_mem(r["normal_mem"])]
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
    print("Normal compilation peak GPU memory")
    print(fmt_row(headers))
    print(sep)
    for row in table_rows:
        print(fmt_row(row))
    print()


@app.command()
def test(
    program_files: str = typer.Argument(..., help="Comma-separated list of ConstraintFlow program files to test"),
    network: str = typer.Argument(..., help="Network path/name"),
    dataset: str = "mnist",
):
    epss = [0, 0.001]
    batch_sizes = [1, 2]
    rows = []
    for program_file in program_files.split(","):
        program_path = "examples/compiler_examples/" + program_file
        print(f"Measuring program file: {program_path} with network: {network} and dataset: {dataset}")
        for batch_size in batch_sizes:
            for eps in epss:
                normal_mem = _run_cli(
                    program_path,
                    network,
                    dataset,
                    ["--eps", str(eps), "--batch-size", str(batch_size)],
                    compile=True,
                    device="gpu",
                )
                rows.append({
                    "program": program_file,
                    "batch": batch_size,
                    "eps": eps,
                    "normal_mem": normal_mem,
                })

    _print_memory_table(rows)


if __name__ == "__main__":
    app()
