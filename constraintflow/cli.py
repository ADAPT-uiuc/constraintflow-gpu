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
    device: str = typer.Option("cpu", help="Device mode: cpu, gpu (CUDA), or gpumac (Apple MPS)"),
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

    typer.echo(f"Lower bounds: {lb}")
    typer.echo(f"Upper bounds: {ub}")
    if eps == 0:
        assert(lb == ub).all(), "Bounds should be equal when eps=0"

    # filename = network_file.split('/')[-1] + f'_{dataset}_{device}.csv'
    # typer.echo(f"Total time: {total_time:.2f} seconds")
    # typer.echo("Matmul Statistics")
    # typer.echo(f"Total Matmul Time: {round(matmul_tensor_ops.get_total_time(), 5)} seconds")
    # typer.echo(f"Matmul Expenses Time: {round((matmul_tensor_ops_expenses.get_total_time()+matmul_sparse_tensor_expenses.get_total_time()+unequal_matmul_profilier.get_total_time()+equal_matmul_profilier.get_total_time()-unequal_matmul_profilier.get_actual_op_time()-equal_matmul_profilier.get_actual_op_time()), 5)} seconds")
    # typer.echo(f"Matmul Actual Op Time: {round((unequal_matmul_profilier.get_actual_op_time()+equal_matmul_profilier.get_actual_op_time()), 5)} seconds")

    # percentage_matmul_operator_time = round(((unequal_matmul_profilier.get_actual_op_time() + equal_matmul_profilier.get_actual_op_time()) / matmul_tensor_ops.get_total_time()) * 100, 5)
    # percentage_matmul_expenses_time = round(((matmul_tensor_ops_expenses.get_total_time() + matmul_sparse_tensor_expenses.get_total_time() + unequal_matmul_profilier.get_total_time() + equal_matmul_profilier.get_total_time() - unequal_matmul_profilier.get_actual_op_time() - equal_matmul_profilier.get_actual_op_time()) / matmul_tensor_ops.get_total_time()) * 100, 5)
    # typer.echo(f"Percentage Matmul Operator Time: {percentage_matmul_operator_time}%")
    # typer.echo(f"Percentage Matmul Expenses Time: {percentage_matmul_expenses_time}%")
    # assert(percentage_matmul_operator_time + percentage_matmul_expenses_time <= 100.0), "Percentages should sum to at most 100%"

    # round_num = 5
    # all_expenses_time = (
    #     binary_sparse_tensor_expenses.get_total_time()
    #     + binary_sparse_tensor_overlap_expenses.get_total_time()
    #     + binary_sparse_tensor_dom1_expenses.get_total_time()
    #     + binary_sparse_tensor_dom2_expenses.get_total_time()
    #     + binary_block_expenses.get_total_time()
    #     - binary_profilier.get_total_time()
    #     + binary_tensor_ops_expenses.get_total_time()
    # )
    # total_binary_time = total_binary_tensor_ops.get_total_time() - binary_tensor_ops_no_sparse.get_total_time() - binary_fixed_costs.get_total_time()

    # percentage_binary_operator_time = round((binary_profilier.get_total_time() / total_binary_time) * 100, round_num)
    # percentage_binary_transfer_time = round((all_expenses_time / total_binary_time) * 100, round_num)
    # typer.echo(f"Percentage Binary Operator Time: {percentage_binary_operator_time}%")
    # typer.echo(f"Percentage Binary Expenses Time: {percentage_binary_transfer_time}%")
    # assert(percentage_binary_operator_time + percentage_binary_transfer_time <= 100.0), "Percentages should sum to at most 100%"

    # typer.echo(f'Total Clamp Time: {round(clamp_total_time.get_total_time(), 5)} seconds')
    # typer.echo(f'Clamp Expense Time: {round(clamp_const_block_expense.get_total_time() + clamp_sparse_block_expense.get_total_time() + clamp_repeat_block_expense.get_total_time(), 5)} seconds')
    # typer.echo(f'Clamp Actual Op Time: {round(clamp_sparse_block_op_time.get_total_time() + clamp_repeat_block_op_time.get_total_time() + clamp_const_block_op_time.get_total_time(), 5)} seconds')
    
    # percentage_clamp_operator_time = round(((clamp_sparse_block_op_time.get_total_time() + clamp_repeat_block_op_time.get_total_time() + clamp_const_block_op_time.get_total_time()) / clamp_total_time.get_total_time()) * 100, 5)
    # percentage_clamp_expenses_time = round(((clamp_const_block_expense.get_total_time() + clamp_sparse_block_expense.get_total_time() + clamp_repeat_block_expense.get_total_time()) / clamp_total_time.get_total_time()) * 100, 5)
    # typer.echo(f'Percentage Clamp Operator Time: {percentage_clamp_operator_time}%')
    # typer.echo(f'Percentage Clamp Expenses Time: {percentage_clamp_expenses_time}%')
    # assert(percentage_clamp_operator_time + percentage_clamp_expenses_time <= 100.0), "Percentages should sum to at most 100%"

    base = network_file.split('/')[-1] + f'_{dataset}_{device}'

    # ── derived quantities ────────────────────────────────────────────────
    matmul_total    = matmul_tensor_ops.get_total_time()
    matmul_actual   = unequal_matmul_profilier.get_actual_op_time() + equal_matmul_profilier.get_actual_op_time()
    matmul_expenses = (matmul_tensor_ops_expenses.get_total_time()
                       + matmul_sparse_tensor_expenses.get_total_time()
                       + unequal_matmul_profilier.get_total_time()
                       + equal_matmul_profilier.get_total_time()
                       - matmul_actual)

    binary_total    = (total_binary_tensor_ops.get_total_time()
                       - binary_tensor_ops_no_sparse.get_total_time()
                       - binary_fixed_costs.get_total_time())
    binary_actual   = binary_profilier.get_total_time()
    binary_expenses = (binary_sparse_tensor_expenses.get_total_time()
                       + binary_sparse_tensor_overlap_expenses.get_total_time()
                       + binary_sparse_tensor_dom1_expenses.get_total_time()
                       + binary_sparse_tensor_dom2_expenses.get_total_time()
                       + binary_block_expenses.get_total_time()
                       - binary_actual
                       + binary_tensor_ops_expenses.get_total_time())

    clamp_total    = clamp_total_time.get_total_time()
    clamp_actual   = (clamp_sparse_block_op_time.get_total_time()
                      + clamp_repeat_block_op_time.get_total_time()
                      + clamp_const_block_op_time.get_total_time())
    clamp_expenses = (clamp_const_block_expense.get_total_time()
                      + clamp_sparse_block_expense.get_total_time()
                      + clamp_repeat_block_expense.get_total_time())

    # ── raw times ─────────────────────────────────────────────────────────
    raw_file = 'stats/' + base + '_stats_raw.csv'
    with open(raw_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Operation', 'Total_Time', 'Expenses_Time', 'Actual_Op_Time'])
        writer.writerow(['Total',  round(total_time,    5), '-', '-'])
        writer.writerow(['Matmul', round(matmul_total,  5), round(matmul_expenses,  5), round(matmul_actual,  5)])
        writer.writerow(['Binary', round(binary_total,  5), round(binary_expenses,  5), round(binary_actual,  5)])
        writer.writerow(['Clamp',  round(clamp_total,   5), round(clamp_expenses,   5), round(clamp_actual,   5)])

    # ── percentages ───────────────────────────────────────────────────────
    def safe_pct(numerator, denominator):
        return round((numerator / denominator) * 100, 5) if denominator else 0.0

    pct_file = 'stats/' + base + '_stats_pct.csv'
    with open(pct_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Operation', 'Pct_Operator_Time', 'Pct_Expenses_Time'])
        writer.writerow(['Matmul', safe_pct(matmul_actual,  matmul_total), safe_pct(matmul_expenses,  matmul_total)])
        writer.writerow(['Binary', safe_pct(binary_actual,  binary_total), safe_pct(binary_expenses,  binary_total)])
        writer.writerow(['Clamp',  safe_pct(clamp_actual,   clamp_total),  safe_pct(clamp_expenses,   clamp_total)])

    typer.echo(f"Total time: {total_time:.2f} seconds")
    typer.echo(f"Raw stats written to:        {raw_file}")
    typer.echo(f"Percentage stats written to: {pct_file}")
    typer.echo(f'Stop condition time: {round(stop_condition_time.get_total_time(), 5)} seconds')


def main():
    app()


if __name__ == "__main__":
    main()
