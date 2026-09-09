# ConstraintFlow

**ConstraintFlow** is a domain-specific language (DSL) and toolchain for specifying, verifying, and compiling neural network certifiers. It bridges the gap between high-level formal specifications and efficient tensor-based runtimes, enabling precise and verifiable DNN analysis.

---

## 📚 Features

ConstraintFlow allows you to:

* **Specify** certifiers declaratively using `.cf` files.
* **Verify** certifiers automatically for soundness.
* **Compile** high-level specifications into optimized tensor-based code.
* **Execute** compiled certifiers on neural network models.

---

## 🚀 Quick Start

### Installation

Pip installation ([Pypi](https://pypi.org/project/constraintflow/0.1.1/))

```bash
pip install constraintflow==0.1.1
```

Or install from source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/constraintflow.git
cd constraintflow
pip install -e .
```

Several fixes (fix PyTorch version to prevent it to require a higher CUDA version; solve `pkg_resources` problem with z3):

```bash
pip install -e . -c requirements.txt
pip uninstall z3_solver
pip install z3_solver
```

### Prepare Models

Create a directory for neural networks:

```bash
mkdir nets/
```

Download pretrained DNNs from [ERAN](https://github.com/eth-sri/eran) and place them inside `nets/`.


---

### CLI Usage

ConstraintFlow provides the following commands with optional flags:

#### `provesound`

```bash
constraintflow provesound example.cf [OPTIONS]
```

**Options:**

| Flag      | Description                           | Default |
| --------- | ------------------------------------- | ------- |
| `--nprev` | Number of previous states to consider | 1       |
| `--nsymb` | Number of symbols to track            | 1       |

#### `compile`

```bash
constraintflow compile example.cf [OPTIONS]
```

**Options:**

| Flag            | Description                  | Default   |
| --------------- | ---------------------------- | --------- |
| `--output-path` | Directory for generated code | `output/` |


#### `run`

```bash
constraintflow run example.cf [OPTIONS]
```

**Options:**

| Flag                           | Description                                 | Default           |
| ------------------------------ | ------------------------------------------- | ----------------- |
| `--network`                    | Network name                                | `mnist_relu_3_50` |
| `--network-format`             | Format of the network file                  | `onnx`            |
| `--dataset`                    | Dataset to use (`mnist` or `cifar`)         | `mnist`           |
| `--batch-size`                 | Batch size                                  | 1                 |
| `--eps`                        | Epsilon                                     | 0.01              |
| `--train`                      | Use training dataset                        | False             |
| `--print-intermediate-results` | Print intermediate results during execution | False             |
| `--no-sparsity`                | Disable sparsity optimizations              | False             |
| `--output-path`                | Path where compiled program is stored       | `output/`         |
| `--compile`                    | Compile the program before running          | False             |
| `--warmup`                      | Number of warmup runs on different data before the timed run               | 0                 |
| `--repeat`                      | Number of timed runs, each reported separately                             | 1                 |
| `--simulacrum`                  | Run Simulacrum (dummy blocks)                                               | False             |
| `--reuse`                       | Reuse stored indices from a prior dummy-blocks run                         | False             |

#### `JIT Optimization`

To apply the JIT optimization, there is a two-step process.

##### `Step 1: Profiling Pass (Simulacrum)`

Run the first pass to profile shape and index metadata:

```bash
constraintflow compile example.cf --simulacrum --compile [OPTIONS]
```

##### `Step 2: Reuse Pass`

Using the profiled information, run the second pass to generate optimized code:

```bash
constraintflow compile example.cf --reuse --compile [OPTIONS]
```


**Options:**

| Flag                           | Description                                 | Default           |
| ------------------------------ | ------------------------------------------- | ----------------- |
| `--network`                    | Network name                                | `mnist_relu_3_50` |
| `--network-format`             | Format of the network file                  | `onnx`            |
| `--dataset`                    | Dataset to use (`mnist` or `cifar`)         | `mnist`           |
| `--batch-size`                 | Batch size                                  | 1                 |
| `--eps`                        | Epsilon                                     | 0.01              |
| `--train`                      | Use training dataset                        | False             |
| `--print-intermediate-results` | Print intermediate results during execution | False             |
| `--no-sparsity`                | Disable sparsity optimizations              | False             |
| `--output-path`                | Path where compiled program is stored       | `output/`         |


##### `Both Passes in One Go (JIT)`
Both passes can be run in one go using the jit command (in-memory keeps the simulacrum metadata in the memory instead of saving it in json files):
```bash
constraintflow jit example.cf --in-memory [OPTIONS]
```

**Options:**

| Flag                            | Description                                                                         | Default           |
| -------------------------------- | -------------------------------------------------------------------------------------- | ----------------- |
| `--network`                     | Network name                                                                        | `mnist_relu_3_50` |
| `--network-format`              | Format of the network file                                                          | `onnx`            |
| `--dataset`                     | Dataset to use (`mnist` or `cifar`)                                                 | `mnist`           |
| `--batch-size`                  | Batch size                                                                          | 1                 |
| `--eps`                         | Epsilon                                                                             | 0                 |
| `--train`                       | Trace on training dataset                                                          | False             |
| `--no-sparsity`                 | Disable sparsity optimizations                                                      | False             |
| `--device`                      | Device mode: `cpu`, `gpu` (CUDA), or `gpumac` (Apple MPS)                           | `cpu`             |
| `--output-path`                 | Output path for generated code                                                     | `output/`         |
| `--print-intermediate-results`  | Print intermediate results during the simulacrum trace pass                        | False             |
| `--dense`                       | Use dense blocks by default                                                        | False             |
| `--jit-dir`                     | Common parent folder for all `jit_*` capture files                                 | `jit_captures`    |
| `--in-memory`                   | Keep jit captures in a process-local dict instead of on disk                        | False             |
| `--no-barriers`                 | Inline every single-use temporary unconditionally, skipping the safety analysis     | False             |
| `--inductor`                    | Emit `@torch.compile(backend='inductor')` on the reuse build                        | False             |
| `--paired-unroll`               | Interleave the paired lower/upper `traverse()` loops when unrolling                 | False             |
| `--fused-flow` / `--no-fused-flow` | Emit a single `flow()` instead of a layered flow           | True              |
| `--fuse-affine-subst` / `--no-fuse-affine-subst` | Pass to optimize redundant Affine calculations (only sound for deeppoly/crown)  | False       |
| `--sroa` / `--no-sroa`          | Scalar-replace the Jit* aggregates into pure tensor code (requires `--fused-flow`)  | True              |



## 📄 Citations

If you use ConstraintFlow in your research, please cite the following papers:

<p>
    <a href="https://arxiv.org/abs/2501.01234"><img src="https://img.shields.io/badge/Paper-arXiv-blue"></a>
    <a href="https://dl.acm.org/doi/10.1145/OOPSLA2025"><img src="https://img.shields.io/badge/Paper-OOPSLA2025-blue"></a>
    <a href="https://example.com/dsl-paper"><img src="https://img.shields.io/badge/Paper-SAS2024-blue"></a>
</p>

```bibtex
@InProceedings{constraintflow,
  author = {Avaljot Singh and Yasmin Sarita and Charith Mendis and Gagandeep Singh},
  title = {ConstraintFlow: A DSL for Specification and Verification of Neural Network Analyses},
  booktitle = {Static Analysis},
  year = {2024},
  publisher = {Springer Nature Switzerland},
}

@InProceedings{provesound,
  author = {Avaljot Singh and Yasmin Sarita and Charith Mendis and Gagandeep Singh},
  title = {Automated Verification of Soundness of DNN Certifiers},
  booktitle = {OOPSLA},
  year = {2025},
}

@Article{compiler,
  author = {Avaljot Singh and Yasmin Sarita and Aditya Mishra and Ishaan Goyal and Gagandeep Singh and Charith Mendis},
  title = {A Tensor-Based Compiler and Runtime for Neuron-Level DNN Certifier Specifications},
  journal = {arXiv},
  year = {2025},
}
```

---

## 🛠 Development

### Requirements

* Python 3.9+
* `antlr4-python3-runtime==4.9.2`

### Running Locally

Install the requirements:

```bash
pip install -r requirements.txt
```

You can then run, compile, or verify any `.cf` file using the CLI.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
