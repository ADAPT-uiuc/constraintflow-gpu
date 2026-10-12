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

To run the JIT optimization, first use the jit command and then use the run command. 

#### `JIT Optimization`

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
| `--jit-dir`                     | Common parent folder for all `jit_*` capture files                                 | `jit_captures`    |
| `--in-memory`                   | Keep jit captures in a process-local dict instead of on disk                        | False             |
| `--inductor`                    | Emit `@torch.compile(backend='inductor')` on the reuse build                        | False             |
| `--fused-flow` / `--no-fused-flow` | Emit a single `flow()` instead of a layered flow           | True              |
| `--fuse-affine-subst` / `--no-fuse-affine-subst` | Pass to optimize redundant Affine calculations (only sound for deeppoly/crown)  | False       |
| `--sroa` / `--no-sroa`          | Scalar-replace the Jit* aggregates into pure tensor code (requires `--fused-flow`)  | True              |
| `--memory-budget-mb`            | Over this estimated peak, reorder/recompute the flow and offload idle values to host memory (`-1`: 70% of GPU memory, `0`: off) | `-1`              |




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

## JAX target after JIT optimization

`jit --target jax` uses the existing Torch simulacrum capture and shared compiler
passes, then emits JAX from the final functional tensor IR. Torch remains the
default target. Install the optional CPU dependency with `python -m pip install -e '.[jax]'`;
for GPU execution, install a JAX build appropriate for your CUDA
environment using the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html).

```bash
python -m constraintflow.cli jit examples/compiler_examples/deeppoly.cf \
  --network /absolute/path/to/network.onnx --dataset mnist --batch-size 1 \
  --eps 0.01 --device cpu --target jax --output-path output/jax

python -m constraintflow.cli run examples/compiler_examples/deeppoly.cf \
  --network /absolute/path/to/network.onnx --dataset mnist --batch-size 1 \
  --eps 0.01 --device cpu --output-path output/jax
```

Use `--device gpu` for CUDA capture/preparation and a CUDA-enabled JAX runtime.
Do not combine `--target jax` with `--inductor`, `--no-sroa`, or
`--no-fused-flow`. Unsupported residual aggregate mutations, storage-stride
operations, and unspecialized control flow fail during emission.

The output contains:

- `jax_flow.py`: pure JAX tensor code with `jax.jit`; this kernel accepts JAX
  arrays directly, including weights as runtime inputs. Shape-dependent
  `batch_size` arguments are static. Kernels specialize to the captured network
  geometry and batch size.
- `transformers.py`: the input extractor and a Torch/JAX boundary adapter.
- `main.py`: the existing network and input preparation interface. It returns
  Torch bounds using DLPack and waits for JAX completion before returning, so
  the existing `run` command can execute the artifact.

The shared passes include block optimization, inlining, sign-convolution and
sign-reduction simplification, early reductions, convolution partials, padding
preparation, and memory planning according to their existing flags.
`--flow-segment-mb` splits the optimized flow and JIT-compiles regions separately;
equivalent regions reuse generated functions. `--memory-budget-mb` still drives
the shared reorder/recompute planner, using Torch shape-probe estimates. Those
estimates are not an XLA memory guarantee. The Torch CUDA stream host-offload
driver is not emitted for JAX, and Torch CUDA peak-memory counters do not measure
JAX allocations. JAX chooses allocation and fusion within each region.

Numerical comparisons use float32 tolerances: XLA and Torch may reorder
floating-point arithmetic. Matmul and convolution request JAX's highest
precision. Explicit 64-bit bridge inputs require `JAX_ENABLE_X64=1`.

Run CPU regression tests, including synthetic ONNX models (no dataset downloads):

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p 'test_jax*.py' -v
```
