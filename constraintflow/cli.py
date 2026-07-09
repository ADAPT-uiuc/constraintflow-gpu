import sys
import constraintflow.lib.globals as globals

argv = sys.argv[1:]
globals.dummy_mode.set_flag() if "--simulacrum" in argv else globals.dummy_mode.reset_flag()
globals.reuse_mode.set_flag() if "--reuse" in argv else globals.reuse_mode.reset_flag()
globals.dense_default_mode.set_flag() if "--dense" in argv else globals.dense_default_mode.reset_flag()
globals.no_barriers.set_flag() if "--no-barriers" in argv else globals.no_barriers.reset_flag()

print(f'dummy_mode in cli: {globals.dummy_mode}')
print(f'reuse_mode in cli: {globals.reuse_mode}')
print(f'no_barriers in cli: {globals.no_barriers}')



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
    typer.echo(f"Total time: {total_time:.2f} seconds")
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
    no_barriers: bool = typer.Option(False, "--no-barriers", help="Inline every single-use temporary unconditionally (skip is_safe_to_inline's safety analysis)."),
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

    globals.set_jit_root(jit_dir)
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

    clear_jit_captures()

    # Simulacrum
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

    trace_path = globals.jit_path("jit_layers", "layers.json")
    if not os.path.isfile(trace_path):
        typer.echo(
            f"Error: simulacrum pass did not produce a trace at '{trace_path}'. "
            "Cannot proceed to the reuse compile."
        )
        raise typer.Exit(code=1)

    # Reuse
    globals.dummy_mode.reset_flag()
    globals.reuse_mode.set_flag()
    try:
        compile_code(program_file, output_path)
    finally:
        globals.reuse_mode.reset_flag()

    typer.echo("Simulacrum+reuse compile complete ✅")
    typer.echo(f"Optimized code written to: {os.path.abspath(output_path)}")
    total_time = time.perf_counter() - start_time
    typer.echo(f"Total time: {total_time:.2f} seconds")
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
    jit_dir: str = typer.Option("jit_captures", help="Common parent folder for all jit_* capture files"),
    no_barriers: bool = typer.Option(False, "--no-barriers", help="Inline every single-use temporary unconditionally (skip is_safe_to_inline's safety analysis). Lower peak memory, not guaranteed value-preserving."),
):
    """
    Run a compiled ConstraintFlow program.
    """
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

    
        
    sys.path.insert(0, os.path.abspath(output_path))
    from main import run  # compiled code provides this

    network_file = get_network(network, network_format, dataset)
    X, y = get_dataset(batch_size, dataset, train=train)
    
    is_cuda = device_mode.get_device() == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    lb, ub = run(
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
    if is_cuda:
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    total_time = end_time - start_time

    typer.echo(f"Lower bounds: {lb}")
    typer.echo(f"Upper bounds: {ub}")
    typer.echo(f"Total time: {total_time:.2f} seconds")
    if is_cuda:
        peak_bytes = torch.cuda.max_memory_allocated()
        typer.echo(f"Peak GPU memory: {peak_bytes} bytes")
    else:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = maxrss if sys.platform == "darwin" else maxrss * 1024
        typer.echo(f"Peak CPU memory: {peak_bytes} bytes")
def main():
    app()


if __name__ == "__main__":
    main()
