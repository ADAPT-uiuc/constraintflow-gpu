import os
import gc
import sys
import constraintflow.lib.globals as globals

os.environ.setdefault("TORCHINDUCTOR_VEC_ISA_OK", "1")

argv = sys.argv[1:]
globals.dummy_mode.set_flag() if "--simulacrum" in argv else globals.dummy_mode.reset_flag()
globals.reuse_mode.set_flag() if "--reuse" in argv else globals.reuse_mode.reset_flag()
globals.dense_default_mode.set_flag() if "--dense" in argv else globals.dense_default_mode.reset_flag()
globals.no_barriers.set_flag() if "--no-barriers" in argv else globals.no_barriers.reset_flag()
globals.inductor_mode.set_flag() if "--inductor" in argv else globals.inductor_mode.reset_flag()
globals.sroa.reset_flag() if "--no-sroa" in argv else globals.sroa.set_flag()

print(f'dummy_mode in cli: {globals.dummy_mode}')
print(f'reuse_mode in cli: {globals.reuse_mode}')
print(f'no_barriers in cli: {globals.no_barriers}')
print(f'inductor_mode in cli: {globals.inductor_mode}')


import os
import shutil
import sys
import torch
import typer
import time
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from constraintflow.gbcsr.tensor_ops import binary
from constraintflow.lib.abs_elem import Abs_elem_sparse
from constraintflow.lib.llist import Llist
from constraintflow.lib.flow_sparse import Flow
from constraintflow.lib.globals import *
from constraintflow.lib.globals import dummy_mode, reuse_mode
import constraintflow.lib.globals 
from constraintflow.compiler.compile import compile as _compile
from constraintflow.verifier.provesound import provesound as _provesound
from constraintflow.lib.spec import get_network_and_input_spec

app = typer.Typer(help="ConstraintFlow CLI for verification and compilation of DSL programs.")

def clear_jit_captures():
    # Every jit_* capture directory lives under the common parent folder
    # (globals.jit_root), so clearing that one folder wipes all captures.
    root = globals.jit_root
    if os.path.isdir(root):
        shutil.rmtree(root)


# --------------------------
# Utility Functions
# --------------------------

def get_program(program_file: str) -> str:
    return program_file

def get_network(network: str, network_format: str, dataset: str) -> str:
    if dataset not in ["mnist", "cifar"]:
        return network
    return network


def _perturbed_network_copy(onnx_path: str, std: float, out_dir: str, tag: str, rng) -> str:
    """Write a copy of `onnx_path` with iid Gaussian noise (given std) added to every
    initializer tensor, and return its path. Shapes/dtypes are unchanged, so it loads
    through the same compiled kernel as the original -- see the recompile smoke test in
    scratchpad/recompile_check.py, which found no recompilation from a value-only change."""
    import onnx
    from onnx import numpy_helper
    model = onnx.load(onnx_path)
    for init in model.graph.initializer:
        arr = numpy_helper.to_array(init).copy()
        arr = arr + std * rng.standard_normal(arr.shape).astype(arr.dtype)
        init.CopyFrom(numpy_helper.from_array(arr, name=init.name))
    out_path = os.path.join(out_dir, f"{_stem(onnx_path)}_{tag}.onnx")
    onnx.save(model, out_path)
    return out_path


def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _configure_cuda_cpu_threads() -> None:
    """Cap eager ATen CPU parallelism before the first compiled CUDA run."""
    raw_limit = os.environ.get("CF_CPU_THREADS", "8")
    try:
        limit = int(raw_limit)
    except ValueError:
        typer.echo(f"Error: CF_CPU_THREADS must be a positive integer, got {raw_limit!r}.")
        raise typer.Exit(code=1)
    if limit < 1:
        typer.echo(f"Error: CF_CPU_THREADS must be a positive integer, got {raw_limit!r}.")
        raise typer.Exit(code=1)

    torch.set_num_threads(min(torch.get_num_threads(), limit))


def get_dataset(
    batch_size: int,
    dataset: str,
    train: bool = False,
    device: str | torch.device | None = None,
):
    if dataset == "mnist":
        transform = transforms.ToTensor()  # keep 28x28
        data = datasets.MNIST(root=".", train=train, download=True, transform=transform)
    elif dataset == "cifar10" or dataset == "cifar":
        transform = transforms.ToTensor()  # keep 32x32
        data = datasets.CIFAR10(root=".", train=train, download=True, transform=transform)
    elif dataset == "tinyimagenet":
        train = True
        transform = transforms.Compose([
            # Match the reference input: resize to 64, then take the 56x56
            # top-left crop below. Resizing directly to 56 changes the pixels
            # and therefore the verification problem despite identical shapes.
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        root_dir = "tinyimagenet/tiny-imagenet-200"
        split = "train" if train else "test"
        data_dir = os.path.join(root_dir, split)
        if train:
            data = datasets.ImageFolder(root=data_dir, transform=transform)
            # -----
            # data = torch.utils.data.Subset(data, range(batch_size))
            # -----
        else:
            # TinyImageNet test: all images in one folder
            from torchvision.datasets.folder import default_loader
            class TinyImageNetTest(torch.utils.data.Dataset):
                def __init__(self, root, transform=None):
                    self.root = root
                    self.transform = transform
                    self.loader = default_loader
                    self.images = sorted(os.listdir(root))
                def __len__(self):
                    return len(self.images)
                def __getitem__(self, idx):
                    path = os.path.join(self.root, self.images[idx])
                    img = self.loader(path)
                    if self.transform:
                        img = self.transform(img)
                    return img, -1
            data = TinyImageNetTest(data_dir, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    target_device = torch.device(device) if device is not None else None
    pin_memory = target_device is not None and target_device.type == "cuda"
    dataloader = DataLoader(
        data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    image, label = next(iter(dataloader))
    if dataset == 'tinyimagenet':
        image = image[:, :, :56, :56]  # ensure 3 channels
    # ensure labels are a tensor
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
    if target_device is not None:
        image = image.to(target_device, non_blocking=pin_memory)
        label = label.to(target_device, non_blocking=pin_memory)
    return image, label



def get_precision(lb):
    verified = (lb >= 0).all(dim=1)
    precision = verified.sum() / verified.shape[0]
    return precision


# --------------------------
# CLI Commands
# --------------------------

@app.command()
def provesound(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    nprev: int = typer.Option(1, help="Number of previous states"),
    nsymb: int = typer.Option(1, help="Number of symbols"),
):
    """
    Prove soundness of a ConstraintFlow program.
    """
    program = get_program(program_file)
    res = _provesound(program, nprev=nprev, nsymb=nsymb)
    typer.echo(f"Provesound result: {res}")


def compile_code(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    output_path: str = typer.Option("output/", help="Output path for generated code"),
):
    """
    Compile a ConstraintFlow program into Python.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)


    program = get_program(program_file)
    res = _compile(program, output_path)
    if res:
        typer.echo("Compilation successful ✅")
    else:
        typer.echo("Compilation failed ❌")
        raise typer.Exit(code=1)

@app.command()
def compile(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    output_path: str = typer.Option("output/", help="Output path for generated code"),
    fuse_affine_subst: bool = typer.Option(False, "--fuse-affine-subst", help="Two optimizations gated by one flag: (1) skip both concretizing traversals at any Affine layer that feeds only further Affine layers (always sound; single_bound.py). (2) Assert every Affine op's L and U outputs are identical (true for all deeppoly*/crown specs here) and drop the redundant sign-split when a traverse() substitution step crosses an Affine layer -- only affects a jit reuse compile (no-op on plain compile, which never runs tensor_to_block); unsound if the assertion doesn't hold."),
):
    start_time = time.perf_counter()
    globals.fuse_affine_subst.set_flag() if fuse_affine_subst else globals.fuse_affine_subst.reset_flag()
    compile_code(program_file, output_path)
    total_time = time.perf_counter() - start_time
    typer.echo(f"Total time: {total_time:.6f} seconds")
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024
    typer.echo(f"Peak CPU memory: {peak_bytes} bytes")


@app.command(name="jit")
def simulacrum_compile(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    network: str = typer.Option("mnist_relu_3_50", help="Network name"),
    network_format: str = typer.Option("onnx", help="Network format"),
    dataset: str = typer.Option("mnist", help="Dataset (mnist or cifar)"),
    batch_size: int = typer.Option(1, help="Batch size"),
    eps: float = typer.Option(0, help="Epsilon"),
    train: bool = typer.Option(False, help="Trace on training dataset"),
    no_sparsity: bool = typer.Option(False, help="Disable sparsity optimizations"),
    device: str = typer.Option("cpu", help="Device mode: cpu, gpu (CUDA), or gpumac (Apple MPS)"),
    output_path: str = typer.Option("output/", help="Output path for generated code"),
    print_intermediate_results: bool = typer.Option(False, help="Print intermediate results during the simulacrum trace pass"),
    dense: bool = typer.Option(False, help="Use dense blocks by default"),
    jit_dir: str = typer.Option("jit_captures", help="Common parent folder for all jit_* capture files"),
    in_memory: bool = typer.Option(True, "--in-memory/--disk-captures", help="Keep jit captures in a process-local dict instead of writing/reading capture files on disk (jit only)."),
    no_barriers: bool = typer.Option(False, "--no-barriers", help="Inline every single-use temporary unconditionally (skip is_safe_to_inline's safety analysis)."),
    inductor: bool = typer.Option(False, help="Emit @torch.compile(backend='inductor') on the reuse build"),
    compact_patches: bool = typer.Option(True, "--compact-patches", help="Use direct patch-to-dense gathering and switch representation when a composed patch is at least as large as the full input feature map. Set for both capture and reuse."),
    paired_unroll: bool = typer.Option(False, "--paired-unroll", help="Interleave the paired lower/upper traverse() loops when unrolling them, instead of emitting one traversal after the other. Reuse compile only; falls back to sequential unrolling whenever the two traversals are not provably independent."),
    fused_flow: bool = typer.Option(True, "--fused-flow/--no-fused-flow", help="Emit a layer-unrolled flow() into transformers.py instead of using the interpretive Flow.flow, replaying abs_elem.update from its simulacrum capture, and (under --inductor) compile the whole flow as one graph instead of one per op. Reuse compile only."),
    fuse_affine_subst: bool = typer.Option(True, "--fuse-affine-subst/--no-fuse-affine-subst", help="Two optimizations gated by one flag: (1) skip both concretizing traversals at any Affine layer that feeds only further Affine layers (always sound; single_bound.py). (2) Assert every Affine op's L and U outputs are identical (true for all deeppoly*/crown specs here) and drop the redundant sign-split when a traverse() substitution step crosses an Affine layer -- unsound if the assertion doesn't hold. Both take effect on the simulacrum and reuse compile passes below. Functional --sroa also proves and removes matching sign-split convolution pairs across residual branches."),
    sroa: bool = typer.Option(True, "--sroa/--no-sroa", help="Splice every layer into one flow() and scalar-replace the Jit* aggregates, so the compiled region is pure tensor code. Requires --fused-flow. Reuse compile only."),
    early_reductions: bool = typer.Option(True, "--early-reductions", help="Compute traversal sums as soon as their inputs exist, releasing large coefficients before later traversal steps. Requires functional --sroa; preserves the arithmetic tree."),
    fuse_sign_convs: bool = typer.Option(True, "--fuse-sign-convs", help="Replace positive/negative convolution pairs by one convolution only when their inputs, weights, views and settings provably match. Requires functional --sroa. Uses linearity and can change floating-point rounding."),
    flow_segment_mb: float = typer.Option(0.0, "--flow-segment-mb", help="Split functional SSA flow into regions with approximately this many MB of named allocations. This is a compilation-region budget, not a bound on total GPU memory. Requires --sroa; Inductor automatically compiles each segment separately. 0 disables."),
):
    """
    Compile a ConstraintFlow program through the whole simulacrum+reuse pipeline
    in one shot.
    """
    start_time = time.perf_counter()
    globals.fuse_affine_subst.set_flag() if fuse_affine_subst else globals.fuse_affine_subst.reset_flag()
    globals.compact_patches.set_flag() if compact_patches else globals.compact_patches.reset_flag()
    globals.paired_unroll.set_flag() if paired_unroll else globals.paired_unroll.reset_flag()
    globals.fused_flow.set_flag() if fused_flow else globals.fused_flow.reset_flag()
    globals.sroa.set_flag() if sroa else globals.sroa.reset_flag()
    if sroa and not fused_flow:
        typer.echo("Error: --sroa requires --fused-flow.")
        raise typer.Exit(code=1)
    if (early_reductions or flow_segment_mb > 0) and not sroa:
        raise typer.BadParameter('--early-reductions and --flow-segment-mb require --sroa')
    if fuse_sign_convs and not sroa:
        raise typer.BadParameter('--fuse-sign-convs requires --sroa')
    if flow_segment_mb < 0:
        raise typer.BadParameter('--flow-segment-mb must be nonnegative')
    if fused_flow and print_intermediate_results:
        typer.echo("Error: --fused-flow drops the per-layer densification, so --print-intermediate-results does not apply.")
        raise typer.Exit(code=1)
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from bench import configs as bench_configs
    onnx_path = network if os.path.isfile(network) else os.path.join(bench_configs.REPO_ROOT, network)
    network = onnx_path

    globals.set_jit_root(jit_dir)
    globals.in_memory_captures.set_flag() if in_memory else globals.in_memory_captures.reset_flag()
    globals.dense_default_mode.set_flag() if dense else globals.dense_default_mode.reset_flag()
    globals.no_barriers.set_flag() if no_barriers else globals.no_barriers.reset_flag()

    valid_devices = {"cpu", "gpu", "gpumac"}
    if device not in valid_devices:
        typer.echo(f"Error: unknown device '{device}'. Choose from: {sorted(valid_devices)}")
        raise typer.Exit(code=1)
    if device == "gpu" and not torch.cuda.is_available():
        typer.echo("Error: device='gpu' requested but CUDA is not available.")
        raise typer.Exit(code=1)
    if device == "gpumac" and not torch.backends.mps.is_available():
        typer.echo("Error: device='gpumac' requested but MPS is not available.")
        raise typer.Exit(code=1)
    device_mode.set_mode(device)

    is_cuda = device_mode.get_device() == "cuda"

    if in_memory:
        globals.jit_store_clear()
    else:
        clear_jit_captures()

    # Simulacrum.
    globals.inductor_mode.reset_flag()
    globals.dummy_mode.set_flag()
    globals.reuse_mode.reset_flag()
    compile_code(program_file, output_path)

    sys.path.insert(0, os.path.abspath(output_path))
    network_file = get_network(network, network_format, dataset)

    from main import run as _probe_run  # probe build provides this

    dataset_device = device_mode.get_device() if is_cuda else None
    X, y = get_dataset(batch_size, dataset, train=train, device=dataset_device)
    _probe_run(
        network_file,
        batch_size,
        eps,
        X,
        y,
        dataset=dataset,
        train=train,
        print_intermediate_results=print_intermediate_results,
        no_sparsity=no_sparsity,
    )

    if not globals.capture_exists("jit_layers/layers.json"):
        where = "the in-memory store" if in_memory else f"'{globals.jit_path('jit_layers', 'layers.json')}'"
        typer.echo(
            f"Error: simulacrum pass did not produce a trace in {where}. "
            "Cannot proceed to the reuse compile."
        )
        raise typer.Exit(code=1)

    # Reuse
    globals.dummy_mode.reset_flag()
    globals.reuse_mode.set_flag()
    if inductor:
        globals.inductor_mode.set_flag()
    globals.early_reductions.set_flag() if early_reductions else globals.early_reductions.reset_flag()
    globals.fuse_sign_convs.set_flag() if fuse_sign_convs else globals.fuse_sign_convs.reset_flag()
    globals.flow_segment_mb.set_value(flow_segment_mb)
    globals.set_network_path(network_file)
    try:
        compile_code(program_file, output_path)
    finally:
        globals.reuse_mode.reset_flag()
        globals.inductor_mode.reset_flag()
        globals.fused_flow.reset_flag()
        globals.compact_patches.reset_flag()
        globals.early_reductions.reset_flag()
        globals.fuse_sign_convs.reset_flag()
        globals.flow_segment_mb.set_value(0.0)
        globals.set_network_path(None)
        if in_memory:
            globals.jit_store_clear()

    typer.echo("Simulacrum+reuse compile complete ✅")
    typer.echo(f"Optimized code written to: {os.path.abspath(output_path)}")
    if is_cuda:
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    typer.echo(f"Total time: {total_time:.6f} seconds")
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024
    typer.echo(f"Peak CPU memory: {peak_bytes} bytes")


@app.command()
def run(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    network: str = typer.Option("mnist_relu_3_50", help="Network name"),
    network_format: str = typer.Option("onnx", help="Network format"),
    dataset: str = typer.Option("mnist", help="Dataset (mnist or cifar)"),
    batch_size: int = typer.Option(1, help="Batch size"),
    eps: float = typer.Option(0.01, help="Epsilon"),
    train: bool = typer.Option(False, help="Run on training dataset"),
    print_intermediate_results: bool = False,
    no_sparsity: bool = typer.Option(False, help="Disable sparsity optimizations"),
    device: str = typer.Option("cpu", help="Device mode: cpu, gpu (CUDA), or gpumac (Apple MPS)"),
    output_path: str = typer.Option("output/", help="Path where compiled program is stored"),
    compile: bool = typer.Option(False, help="Run compilation before execution"),
    opt: bool = typer.Option(False, help="Static shape analysis and direct computation over blocks."),
    simulacrum: bool = typer.Option(False, help="Run Simulacrum (dummy blocks)"),
    reuse: bool = typer.Option(False, help="Reuse the stored indices that were stored by running dummy blocks"),
    dense: bool = typer.Option(False, help="Use dense blocks by default"),
    inductor: bool = typer.Option(False, help="Use PyTorch Inductor for JIT compilation"),
    jit_dir: str = typer.Option("jit_captures", help="Common parent folder for all jit_* capture files"),
    no_barriers: bool = typer.Option(False, "--no-barriers", help="Inline every single-use temporary unconditionally (skip is_safe_to_inline's safety analysis). Lower peak memory, not guaranteed value-preserving."),
    warmup: int = typer.Option(0, help="Number of warmup runs on different data before the timed run"),
    repeat: int = typer.Option(1, "--repeat", help="Number of timed runs, each reported separately. Independent of --warmup: the warmup runs (if any) still happen once, before the first timed run."),
    use_cache: bool = typer.Option(False, "--use-cache", help="Point --output-path at the shared kernel_cache entry (see bench/configs.py:kernel_dir), keyed by network/dataset/certifier/batch-size/inductor. Hard errors if that cache entry is missing."),
    perturb_eps: float = typer.Option(0.0, "--perturb-eps", help="Std of iid Gaussian noise added to the network's weights independently before each warmup and each timed run, to test whether weight values (not just shapes) affect measured runtime. 0 (default) disables perturbation and every run uses the unmodified network."),
    perturb_seed: int = typer.Option(0, "--perturb-seed", help="Seed for --perturb-eps's noise, for reproducible sweeps."),
):
    """
    Run a compiled ConstraintFlow program.
    """
    if use_cache:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from bench import configs as bench_configs

        if network.endswith(".onnx"):
            onnx_path = network if os.path.isfile(network) else os.path.join(bench_configs.REPO_ROOT, network)
        else:
            onnx_path = bench_configs.network_path(dataset, network)
        network = onnx_path
        output_path = bench_configs.require_kernel(
            bench_configs.kernel_dir("jit", program_file, onnx_path, batch_size, inductor),
            f"python3 bench/build_kernels.py --tool jit --network {onnx_path} --batch-size {batch_size} --device {device}",
        )

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from bench import configs as bench_configs
    onnx_path = network if os.path.isfile(network) else os.path.join(bench_configs.REPO_ROOT, network)
    network = onnx_path

    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)

    globals.set_jit_root(jit_dir)

    if simulacrum:
        clear_jit_captures()
    
    if compile:
        compile_code(program_file, output_path)

    valid_devices = {"cpu", "gpu", "gpumac"}
    if device not in valid_devices:
        typer.echo(f"Error: unknown device '{device}'. Choose from: {sorted(valid_devices)}")
        raise typer.Exit(code=1)
    if device == "gpu" and not torch.cuda.is_available():
        typer.echo("Error: device='gpu' requested but CUDA is not available.")
        raise typer.Exit(code=1)
    if device == "gpumac" and not torch.backends.mps.is_available():
        typer.echo("Error: device='gpumac' requested but MPS is not available.")
        raise typer.Exit(code=1)
    device_mode.set_mode(device)
    if device == "gpu":
        _configure_cuda_cpu_threads()

    if repeat < 1:
        typer.echo("Error: --repeat must be >= 1.")
        raise typer.Exit(code=1)

    sys.path.insert(0, os.path.abspath(output_path))
    from main import run  # compiled code provides this

    network_file = get_network(network, network_format, dataset)
    dataset_device = device_mode.get_device() if device == "gpu" else None
    X, y = get_dataset(
        batch_size * (warmup + 1),
        dataset,
        train=train,
        device=dataset_device,
    )

    perturb_rng = None
    perturb_dir = None
    if perturb_eps > 0:
        import numpy as np
        import tempfile
        perturb_rng = np.random.default_rng(perturb_seed)
        perturb_dir = tempfile.mkdtemp(prefix="cf_perturb_")

    def _network_for(tag: str) -> str:
        """The network path to hand `run()` for one warmup/repeat call: a fresh perturbed
        copy per call when --perturb-eps is set, else the unmodified network every time."""
        if perturb_eps <= 0:
            return network_file
        return _perturbed_network_copy(network_file, perturb_eps, perturb_dir, tag, perturb_rng)

    is_cuda = device_mode.get_device() == "cuda"
    print(f"WARMMMMMM")
    torch._dynamo.reset()

    for i in range(warmup):
        warmup_network = _network_for(f"warmup{i}")
        warmup_start = time.perf_counter()
        with torch.no_grad():
            run(
                warmup_network,
                batch_size,
                eps,
                X[(i + 1) * batch_size : (i + 2) * batch_size],
                y[(i + 1) * batch_size : (i + 2) * batch_size],
                dataset=dataset,
                train=train,
                print_intermediate_results=False,
                no_sparsity=no_sparsity,
            )
        if is_cuda:
            torch.cuda.synchronize()
        typer.echo(f"Warmup run {i + 1}/{warmup}: {time.perf_counter() - warmup_start:.6f} s")
        ### Free Memory ####
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

    mem_label = "Peak GPU memory" if is_cuda else "Peak CPU memory"

    def _peak_bytes():
        if is_cuda:
            return torch.cuda.max_memory_allocated()
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return maxrss if sys.platform == "darwin" else maxrss * 1024

    repeat_times = []
    repeat_peaks = []
    print("REPEAT")
    for i in range(repeat):
        repeat_network = _network_for(f"repeat{i}")
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()
        with torch.no_grad():
            lb, ub = run(
                repeat_network,
                batch_size,
                eps,
                X[:batch_size],
                y[:batch_size],
                dataset=dataset,
                train=train,
                print_intermediate_results=print_intermediate_results,
                no_sparsity=no_sparsity,
            )
        if is_cuda:
            torch.cuda.synchronize()
        run_time = time.perf_counter() - start_time
        run_peak = _peak_bytes()
        repeat_times.append(run_time)
        repeat_peaks.append(run_peak)
        if repeat > 1:
            typer.echo(f"Run {i + 1}/{repeat}: {run_time:.6f} s, {mem_label}: {run_peak} bytes")

        ### Free Memory ####
        lb, ub = lb.detach().cpu(), ub.detach().cpu()
        gc.collect()
        if is_cuda:
            torch.cuda.empty_cache()

    total_time = sum(repeat_times)
    peak_bytes = max(repeat_peaks)

    typer.echo(f"Lower bounds: {lb}")
    typer.echo(f"Upper bounds: {ub}")
    typer.echo(f"Total time: {total_time:.6f} seconds")
    typer.echo(f"{mem_label}: {peak_bytes} bytes")
def main():
    app()


if __name__ == "__main__":
    main()
