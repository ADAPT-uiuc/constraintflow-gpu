import os
import sys
import constraintflow.lib.globals as globals

os.environ.setdefault("TORCHINDUCTOR_VEC_ISA_OK", "1")

argv = sys.argv[1:]
globals.dummy_mode.set_flag() if "--simulacrum" in argv else globals.dummy_mode.reset_flag()
globals.reuse_mode.set_flag() if "--reuse" in argv else globals.reuse_mode.reset_flag()
globals.dense_default_mode.set_flag() if "--dense" in argv else globals.dense_default_mode.reset_flag()
globals.no_barriers.set_flag() if "--no-barriers" in argv else globals.no_barriers.reset_flag()
globals.inductor_mode.set_flag() if "--inductor" in argv else globals.inductor_mode.reset_flag()

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


def get_dataset(batch_size: int, dataset: str, train: bool = False):
    if dataset == "mnist":
        transform = transforms.ToTensor()  # keep 28x28
        data = datasets.MNIST(root=".", train=train, download=True, transform=transform)
    elif dataset == "cifar10" or dataset == "cifar":
        transform = transforms.ToTensor()  # keep 32x32
        data = datasets.CIFAR10(root=".", train=train, download=True, transform=transform)
    elif dataset == "tinyimagenet":
        train = True
        transform = transforms.Compose([
            # But its onnx model expects 56x56, why?
            transforms.Resize((56, 56)),  # TinyImageNet images are 64x64
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

    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    image, label = next(iter(dataloader))
    if dataset == 'tinyimagenet':
        image = image[:, :, :56, :56]  # ensure 3 channels
    # ensure labels are a tensor
    if not isinstance(label, torch.Tensor):
        label = torch.tensor(label)
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
):
    start_time = time.perf_counter()
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
    in_memory: bool = typer.Option(False, "--in-memory", help="Keep jit captures in a process-local dict instead of writing/reading capture files on disk (jit only)."),
    no_barriers: bool = typer.Option(False, "--no-barriers", help="Inline every single-use temporary unconditionally (skip is_safe_to_inline's safety analysis)."),
    inductor: bool = typer.Option(False, help="Emit @torch.compile(backend='inductor') on the reuse build"),
):
    """
    Compile a ConstraintFlow program through the whole simulacrum+reuse pipeline
    in one shot.
    """
    start_time = time.perf_counter()
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
    from main import run as _probe_run  # probe build provides this

    network_file = get_network(network, network_format, dataset)
    X, y = get_dataset(batch_size, dataset, train=train)
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
    try:
        compile_code(program_file, output_path)
    finally:
        globals.reuse_mode.reset_flag()
        globals.inductor_mode.reset_flag()
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
    aot_save: bool = typer.Option(False, "--aot-save", help="Compile each inductor kernel ahead of time and store it under <output-path>/aot, instead of timing a run."),
    aot: bool = typer.Option(False, "--aot", help="Load the kernels stored by --aot-save instead of compiling them. Errors out if they are missing or stale."),
    use_cache: bool = typer.Option(False, "--use-cache", help="Point --output-path at the shared kernel_cache entry (see bench/configs.py:kernel_dir), keyed by network/dataset/certifier/batch-size/inductor. Hard errors if that cache entry is missing. Combine with --aot to also load its prebuilt AOT kernels."),
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

    if aot_save and aot:
        typer.echo("Error: --aot-save builds the kernels and --aot loads them; pass one or the other.")
        raise typer.Exit(code=1)
    if (aot_save or aot) and not inductor:
        typer.echo("Error: --aot-save/--aot is not possible without inductor")
        raise typer.Exit(code=1)
    if aot_save and warmup:
        typer.echo("Error: --aot-save measures build time and returns before the timed run; --warmup does not apply.")
        raise typer.Exit(code=1)
    if aot_save and repeat != 1:
        typer.echo("Error: --aot-save measures build time and returns before the timed run; --repeat does not apply.")
        raise typer.Exit(code=1)
    if repeat < 1:
        typer.echo("Error: --repeat must be >= 1.")
        raise typer.Exit(code=1)
    if aot_save:
        torch._dynamo.config.enable_aot_compile = True

    sys.path.insert(0, os.path.abspath(output_path))
    from main import run  # compiled code provides this

    aot_dir = os.path.join(os.path.abspath(output_path), "aot")
    # if aot_save or aot:
    #     import builtins
    #     import hashlib
    #     import io
    #     import json
    #     import re
    #     import torch._inductor.runtime.triton_heuristics as triton_heuristics
    #     autotune_to_one_config = triton_heuristics.CachingAutotuner.autotune_to_one_config
    #     transformers_module = sys.modules["transformers"]
    #     source_hash = hashlib.sha256(
    #         open(os.path.join(os.path.abspath(output_path), "transformers.py"), "rb").read()
    #     ).hexdigest()
    #     stamp = {
    #         "torch": torch.__version__,
    #         "transformers_sha256": source_hash,
    #         "vec_isa_ok": os.environ.get("TORCHINDUCTOR_VEC_ISA_OK"),
    #         "device": device,
    #         "capability": list(torch.cuda.get_device_capability()) if device == "gpu" else None,
    #     }

    # if aot_save:
    #     os.makedirs(aot_dir, exist_ok=True)
    #     saved = []
    #     tuned = {}

    #     def record_tuning(self, *args, **kwargs):
    #         autotune_to_one_config(self, *args, **kwargs)
    #         kernel_name = (self.inductor_meta or {}).get("kernel_name")
    #         if kernel_name and self.launchers:
    #             tuned[kernel_name] = str(self.launchers[0].config)

    #     triton_heuristics.CachingAutotuner.autotune_to_one_config = record_tuning
    #     for cls_name, cls in list(vars(transformers_module).items()):
    #         if not isinstance(cls, type):
    #             continue
    #         for method_name, method in list(vars(cls).items()):
    #             if not hasattr(method, "aot_compile"):
    #                 continue

    #             def shim(_cls_name=cls_name, _method_name=method_name, _method=method, _state={}):
    #                 def call(*args, **kwargs):
    #                     if "fn" not in _state:
    #                         name = f"{_cls_name}.{_method_name}"
    #                         typer.echo(f"  compiling {name} ...")
    #                         compiled = _method.aot_compile((args, kwargs))
    #                         compiled.save_compiled_function(os.path.join(aot_dir, name + ".pt"))
    #                         _state["fn"] = compiled
    #                         saved.append(name)
    #                     return _state["fn"](*args, **kwargs)

    #                 return call

    #             setattr(cls, method_name, shim())

    # if aot:
    #     manifest_path = os.path.join(aot_dir, "manifest.json")
    #     if not os.path.exists(manifest_path):
    #         typer.echo(f"Error: no AOT kernels found at {aot_dir}. Build them with --aot-save.")
    #         raise typer.Exit(code=1)
    #     with open(manifest_path) as f:
    #         manifest = json.load(f)
    #     for key, value in stamp.items():
    #         if manifest.get(key) != value:
    #             typer.echo(
    #                 f"Error: AOT kernels in {aot_dir} are stale: {key} was "
    #                 f"{manifest.get(key)!r} at build time, is {value!r} now. Rebuild with --aot-save."
    #             )
    #             raise typer.Exit(code=1)
    #     tuned = manifest.get("tuned_configs", {})

    #     def replay_tuning(self, *args, **kwargs):
    #         kernel_name = (self.inductor_meta or {}).get("kernel_name")
    #         wanted = tuned.get(kernel_name)
    #         if wanted and self.launchers:
    #             for launcher in self.launchers:
    #                 if str(launcher.config) == wanted:
    #                     self.launchers = [launcher]
    #                     return
    #         autotune_to_one_config(self, *args, **kwargs)

    #     triton_heuristics.CachingAutotuner.autotune_to_one_config = replay_tuning

    #     module_globals = dict(vars(transformers_module))
    #     for name in manifest["methods"]:
    #         with open(os.path.join(aot_dir, name + ".pt"), "rb") as f:
    #             blob = f.read()
    #         while True:
    #             try:
    #                 loaded = torch.compiler.load_compiled_function(
    #                     io.BytesIO(blob), f_globals=module_globals
    #                 )
    #                 break
    #             except RuntimeError as e:
    #                 missing = re.findall(r"__builtins_dict___\d+", str(e))
    #                 if not missing or "Missing required external references" not in str(e):
    #                     raise
    #                 for ref in missing:
    #                     module_globals[ref] = vars(builtins)
    #         cls_name, method_name = name.split(".")
    #         setattr(
    #             getattr(transformers_module, cls_name),
    #             method_name,
    #             (lambda _f: lambda *a, **k: _f(*a, **k))(loaded),
    #         )
    #     typer.echo(f"Loaded {len(manifest['methods'])} AOT kernels from {aot_dir}")

    network_file = get_network(network, network_format, dataset)
    X, y = get_dataset(batch_size * (warmup + 1), dataset, train=train)

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

    # if aot_save:
    #     start_time = time.perf_counter()
    #     lb, ub = run(
    #         network_file,
    #         batch_size,
    #         eps,
    #         X,
    #         y,
    #         dataset=dataset,
    #         train=train,
    #         print_intermediate_results=print_intermediate_results,
    #         no_sparsity=no_sparsity,
    #     )
    #     if device_mode.get_device() == "cuda":
    #         torch.cuda.synchronize()
    #     stamp["methods"] = saved
    #     stamp["tuned_configs"] = tuned
    #     with open(os.path.join(aot_dir, "manifest.json"), "w") as f:
    #         json.dump(stamp, f, indent=2)
    #     typer.echo(f"Saved {len(saved)} AOT kernels and {len(tuned)} tuned configs to {aot_dir}")
    #     typer.echo(f"Build time: {time.perf_counter() - start_time:.6f} seconds")
    #     typer.echo(f"Lower bounds: {lb}")
    #     typer.echo(f"Upper bounds: {ub}")
    #     return

    is_cuda = device_mode.get_device() == "cuda"
    print(f"WARMMMMMM")

    for i in range(warmup):
        warmup_network = _network_for(f"warmup{i}")
        warmup_start = time.perf_counter()
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
