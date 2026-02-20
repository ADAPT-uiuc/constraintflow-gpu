import os
import sys
import torch
import typer
import time
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from constraintflow.lib.globals import *
from constraintflow.compiler.compile import compile as _compile
from constraintflow.verifier.provesound import provesound as _provesound

app = typer.Typer(help="ConstraintFlow CLI for verification and compilation of DSL programs.")


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
            transforms.Resize((64, 64)),  # TinyImageNet images are 64x64
            transforms.ToTensor(),
        ])
        root_dir = "tinyimagenet/tiny-imagenet-200"
        split = "train" if train else "test"
        data_dir = os.path.join(root_dir, split)
        if train:
            data = datasets.ImageFolder(root=data_dir, transform=transform)
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
    compile_code(program_file, output_path)


@app.command()
def run(
    program_file: str = typer.Argument(..., help="ConstraintFlow program file"),
    network: str = typer.Option("mnist_relu_3_50", help="Network name"),
    network_format: str = typer.Option("onnx", help="Network format"),
    dataset: str = typer.Option("mnist", help="Dataset (mnist or cifar)"),
    batch_size: int = typer.Option(1, help="Batch size"),
    eps: float = typer.Option(0.01, help="Epsilon"),
    train: bool = typer.Option(False, help="Run on training dataset"),
    print_intermediate_results: bool = typer.Option(False, help="Print intermediate results"),
    no_sparsity: bool = typer.Option(False, help="Disable sparsity optimizations"),
    output_path: str = typer.Option("output/", help="Path where compiled program is stored"),
    compile: bool = typer.Option(False, help="Run compilation before execution"),
):
    """
    Run a compiled ConstraintFlow program.
    """
    try:
        os.makedirs(output_path, exist_ok=True)
    except OSError as e:
        typer.echo(f"Error creating folder '{output_path}': {e}")
        raise typer.Exit(code=1)

    if compile:
        compile_code(program_file, output_path)
        
    sys.path.insert(0, os.path.abspath(output_path))
    from main import run  # compiled code provides this

    network_file = get_network(network, network_format, dataset)
    X, y = get_dataset(batch_size, dataset, train=train)
    filename =network_file.split('/')[-1]+f'_{dataset}_cpu.csv'
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
    end_time = time.perf_counter()
    total_time = end_time - start_time

    typer.echo(f"Lower bound: {lb}")
    typer.echo(f"Upper bound: {ub}")
    precision = get_precision(lb)
    typer.echo(f"Precision: {precision}")
    typer.echo(f"Total fixed time: {fixed_cost1.get_total_time():.3f} seconds")
    typer.echo(f"Total fixed num calls: {fixed_cost1.num_used} calls")

    typer.echo(f"Total fixed time: {fixed_cost2.get_total_time():.3f} seconds")
    typer.echo(f"Total fixed num calls: {fixed_cost2.num_used} calls")

    rows = [
        [
            "Total Time",
            total_time,
        ],

        [
            "Binary",
            binary_profilier.get_actual_op_time(),
            binary_profilier.get_data_transfer_time(),
            binary_profilier.get_num_ops(),
            binary_profilier.get_total_time(),
            binary_time.num_used,
            binary_time.get_total_time(),
            binary_1_time.num_used,
            binary_1_time.get_total_time(),
            binary_2_time.num_used,
            binary_2_time.get_total_time(),
            binary_3_time.num_used,
            binary_3_time.get_total_time(),
            binary_4_time.num_used,
            binary_4_time.get_total_time(),
            binary_5_time.num_used,
            binary_5_time.get_total_time(),
            binary_6_time.num_used,
            binary_6_time.get_total_time(),
            new_sanity_time.num_used,
            new_sanity_time.get_total_time(),
        ],
        [
            "Unary",
            unary_profilier.get_actual_op_time(),
            unary_profilier.get_data_transfer_time(),
            unary_profilier.get_num_ops(),
            unary_profilier.get_total_time(),
            unary_time.get_total_time()
        ],
        [
            "Equal Matmul",
            equal_matmul_profilier.get_actual_op_time(),
            equal_matmul_profilier.get_data_transfer_time(),
            equal_matmul_profilier.get_num_ops(),
            equal_matmul_profilier.get_total_time(),
            matmul_time.get_total_time()
        ],
        [
            "Unequal Matmul",
            unequal_matmul_profilier.get_actual_op_time(),
            unequal_matmul_profilier.get_data_transfer_time(),
            unequal_matmul_profilier.get_num_ops(),
            unequal_matmul_profilier.get_total_time(),
            matmul_time.get_total_time()
        ],
        [
            "Clamp",
            clamp_profilier.get_actual_op_time(),
            clamp_profilier.get_data_transfer_time(),
            clamp_profilier.get_num_ops(),
            clamp_profilier.get_total_time(),
            clamp_time.get_total_time()
        ]
    ]   

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Operation", "Op Time (s)", "Transfer Time (s)", "Num Ops", "Total Op Time (s)", "Tensor Time (s)"])
        writer.writerows(rows)
def main():
    app()


if __name__ == "__main__":
    main()
