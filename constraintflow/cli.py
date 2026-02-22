import os
import sys
import torch
import typer
import time
import csv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from constraintflow.gbcsr.tensor_ops import binary
from constraintflow.lib.globals import *
from constraintflow.compiler.compile import compile as _compile
from constraintflow.verifier.provesound import provesound as _provesound

app = typer.Typer(help="ConstraintFlow CLI for verification and compilation of DSL programs.")



def binary(total_time, filename):
    round_num = 5
    all_expenses_time = (
        binary_sparse_tensor_expenses.get_total_time()
        + binary_sparse_tensor_overlap_expenses.get_total_time()
        + binary_sparse_tensor_dom1_expenses.get_total_time()
        + binary_sparse_tensor_dom2_expenses.get_total_time()
        + binary_block_expenses.get_total_time()
        - binary_profilier.get_total_time()
        + binary_tensor_ops_expenses.get_total_time()
    )
    total_binary_time = total_binary_tensor_ops.get_total_time() - binary_tensor_ops_no_sparse.get_total_time() - binary_fixed_costs.get_total_time()
    rows = [
        ["Total Time", total_time, 1],
        ["total_binary_tensor_ops", round(total_binary_time,round_num), total_binary_tensor_ops.num_used],
        ["binary_tensor_ops_expenses", round(binary_tensor_ops_expenses.get_total_time(),round_num), binary_tensor_ops_expenses.num_used],
        ["binary_tensor_ops_x_sparsity", round(binary_tensor_ops_x_sparsity.get_total_time(),round_num), binary_tensor_ops_x_sparsity.num_used],
        ["binary_tensor_ops_y_sparsity", round(binary_tensor_ops_y_sparsity.get_total_time(),round_num), binary_tensor_ops_y_sparsity.num_used],
        ["binary_tensor_ops_no_sparse", round(binary_tensor_ops_no_sparse.get_total_time(),round_num), binary_tensor_ops_no_sparse.num_used],
        ["total_binary_sparse_tensor", round(total_binary_sparse_tensor.get_total_time(),round_num), total_binary_sparse_tensor.num_used],
        ["binary_sparse_tensor_expenses", round(binary_sparse_tensor_expenses.get_total_time(),round_num), binary_sparse_tensor_expenses.num_used],
        ["binary_sparse_tensor_dom2", round(binary_sparse_tensor_dom2.get_total_time(),round_num), binary_sparse_tensor_dom2.num_used],
        ["binary_sparse_tensor_dom1", round(binary_sparse_tensor_dom1.get_total_time(),round_num), binary_sparse_tensor_dom1.num_used],
        ["binary_sparse_tensor_overlap", round(binary_sparse_tensor_overlap.get_total_time(),round_num), binary_sparse_tensor_overlap.num_used],
        ["binary_sparse_tensor_overlap_expenses", round(binary_sparse_tensor_overlap_expenses.get_total_time(),round_num), binary_sparse_tensor_overlap_expenses.num_used],
        ["binary_sparse_tensor_dom1_expenses", round(binary_sparse_tensor_dom1_expenses.get_total_time(),round_num), binary_sparse_tensor_dom1_expenses.num_used],
        ["binary_sparse_tensor_dom2_expenses", round(binary_sparse_tensor_dom2_expenses.get_total_time(),round_num), binary_sparse_tensor_dom2_expenses.num_used],
        ["binary_profilier", round(binary_profilier.get_total_time(),round_num), binary_profilier.get_num_ops()],
        ["binary_block_expenses", round(binary_block_expenses.get_total_time(),round_num), binary_block_expenses.num_used],
        ["all_expenses", round(all_expenses_time, round_num), -1],
        ["expenses_percentage", round((all_expenses_time/total_binary_time)*100), -1],
        ["block_operations_percentage", round((binary_profilier.get_total_time()/total_binary_time)*100), -1],
    ]

    from anytree import Node, RenderTree
    binary_tensor_ops_node = Node(f"binary_tensor_ops {total_binary_tensor_ops.num_used} ")
    binary_tensor_both_sparsity_node = Node(f"binary_tensor_ops_x&y {binary_tensor_ops_x_sparsity.num_used+binary_tensor_ops_y_sparsity.num_used}", parent=binary_tensor_ops_node)
    binary_tensor_ops_no_sparse_node = Node(f"binary_tensor_ops_no_sparse {binary_tensor_ops_no_sparse.num_used}", parent=binary_tensor_ops_node)
    binary_sparse_tensor_dom1_node = Node(f"binary_sparse_tensor_dom1 {binary_sparse_tensor_dom1.num_used}", parent=binary_tensor_both_sparsity_node)
    binary_sparse_tensor_dom2_node = Node(f"binary_sparse_tensor_dom2 {binary_sparse_tensor_dom2.num_used}", parent=binary_tensor_both_sparsity_node)
    binary_sparse_tensor_overlap_node = Node(f"binary_sparse_tensor_overlap {binary_sparse_tensor_overlap.num_used}", parent=binary_tensor_both_sparsity_node)
    assert(binary_sparse_tensor_count.num_used == (binary_tensor_ops_x_sparsity.num_used + binary_tensor_ops_y_sparsity.num_used))
    from anytree.exporter import UniqueDotExporter
    for pre, fill, node in RenderTree(binary_tensor_ops_node):
        print("%s%s" % (pre, node.name))
    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Operations","Total_Time", "Num_Ops"])
        writer.writerows(rows)

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
    gpu: bool = typer.Option(False, "--gpu/--no-gpu", help="Enable GPU execution path"),
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

    if gpu:
        if not torch.cuda.is_available():
            typer.echo("Error: --gpu was provided but CUDA is not available in this environment.")
            raise typer.Exit(code=1)
        baseline_gpu_mode.set_flag()
        
    sys.path.insert(0, os.path.abspath(output_path))
    from main import run  # compiled code provides this

    network_file = get_network(network, network_format, dataset)
    X, y = get_dataset(batch_size, dataset, train=train)
    
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
    filename = network_file.split('/')[-1]+f'_{dataset}_cpu.csv'
    if gpu:
        filename = network_file.split('/')[-1]+f'_{dataset}_gpu.csv'
    
    typer.echo(f"Total time: {total_time:.2f} seconds")
    typer.echo(f"Total Matmul Time: {round(matmul_tensor_ops.get_total_time(), 5)} seconds")
    typer.echo(f"Matmul Expenses Time: {round((matmul_tensor_ops_expenses.get_total_time()+matmul_sparse_tensor_expenses.get_total_time()+unequal_matmul_profilier.get_total_time()+equal_matmul_profilier.get_total_time()-unequal_matmul_profilier.get_actual_op_time()-equal_matmul_profilier.get_actual_op_time()), 5)} seconds")
    typer.echo(f"Matmul Actual Op Time: {round((unequal_matmul_profilier.get_actual_op_time()+equal_matmul_profilier.get_actual_op_time()), 5)} seconds")


def main():
    app()


if __name__ == "__main__":
    main()
