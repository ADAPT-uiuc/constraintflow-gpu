# Reuse path: `SparseTensor` API inventory and `sparse_tensor` method roles

This document satisfies the **inventory-reuse-api** and **split-layout-vs-trace** tasks: what reuse-mode codegen can emit into `transformers.py`, how that relates to `tensor_ops`, and how `SparseTensor` methods split between tracing, reuse runtime, and shared library use.

Primary sources: [`constraintflow/compiler/codeGen.py`](../constraintflow/compiler/codeGen.py), [`constraintflow/gbcsr/sparse_tensor.py`](../constraintflow/gbcsr/sparse_tensor.py), [`constraintflow/gbcsr/tensor_ops.py`](../constraintflow/gbcsr/tensor_ops.py), and a spot-check of [`output/transformers.py`](../output/transformers.py) (treat generated trees under `output/` as samples; re-run codegen to validate your checkout).

---

## 1. Reuse `transformers.py` header (no `tensor_ops`)

When `reuse_mode` is set, codegen writes `transformers.py` with:

- `from constraintflow.gbcsr.sparse_tensor import SparseTensor`
- **No** `from constraintflow.gbcsr.tensor_ops import *`
- A local `convert_to_float` that dispatches `torch.Tensor` and `SparseTensor` (`codeGen.py` around lines 103–109).

Other imports include `symexp` and `polyexp`; `from constraintflow.lib.symexp import *` re-exports names from `sparse_tensor` and `sparse_block` (including `binary_to_identity_unary` via `op_helper` and `sp_where_block` from `sparse_block`).

---

## 2. `SparseTensor`-related surfaces **emitted by codegen**

These are the expressions that appear **on a `SparseTensor` instance or constructor** when visiting IR (not block-level `.binary` on `SparseBlock`, which is separate).

| Emitted pattern | Visitor(s) | Notes |
|-----------------|------------|--------|
| `SparseTensor(...)` | `visitIrSparseTensor`, `visitIrAddDimensionConst` | Constructor with `start_indices`, `blocks`, `dims`, `total_size`, optional `end_indices`, `type`, `dense_const`. |
| `.get_sub_block_custom_range(start, end, block_id, tensor_bool)` | `visitIrGetSubBlockCustomRange` | Replay of sub-block extraction from traces. |
| `.unsqueeze(i)` (chain) | `visitIrAddDimension` | One or more calls. |
| `.squeeze(n)` | `visitIrRemoveDimension` | |
| `.repeat(torch.tensor([...]))` | `visitIrAddDimensionConst` | After optional `unsqueeze` chain. |
| `.float()` (indirect) | `visitIrConvertBoolToFloat` via `convert_to_float(x)` | |
| `.unary(operator.not_)` | `visitIrPolyExpNotStopFloat` | **Instance** method on `SparseTensor`, not `tensor_ops.unary`. |

**Layout field access (on values that may be `SparseTensor` in IR):**

- `visitIrBlockExtract` → `.blocks[i]`

Codegen also emits many **`SparseBlock`** calls: `.binary(...)`, `.unary(...)`, `.clamp(...)`, `.repeat(...)`, `.matmul_equal_dims` / `.matmul_unequal_dims`, `.any()`, `.all()`, `.dims`, `.create_similar(...)`, plus `sp_where_block(...)`, `binary_to_identity_unary(...)`, `ConstBlock(...)`, `phi(...)`, `PolyExpSparse`, `SymExpSparse`, `abs_elem` accessors, etc. Those are not `SparseTensor` methods but appear alongside the above in reuse output.

---

## 3. `tensor_ops` symbols codegen **can** still print (TTB must erase them for reuse)

These visitors emit **top-level** calls that are defined in `tensor_ops` when **not** stripped by [`tensor_to_block`](../constraintflow/compiler/optimizations/tensor_to_block.py):

| Symbol | Visitor(s) |
|--------|------------|
| `binary(...)` | `visitIrBinaryOp` (non–`cf_max`/`cf_min`), `visitIrMult` |
| `unary(...)` | `visitIrUnaryOp` (non–`any`/`all`/`get_dims`/`get_shape_*`) |
| `any(...)`, `all(...)`, `get_dims(...)` | `visitIrUnaryOp` |
| `get_shape_0(...)`, `get_shape_1(...)` | `visitIrUnaryOp` |
| `where(...)` | `visitIrTernary` |
| `clamp(...)` | `visitIrClamp` |
| `inner_prod(...)` | `visitIrInnerProduct`, `visitIrDot` (float branch) |
| `cf_max(...)`, `cf_min(...)` | `visitIrBinaryOp` |
| `repeat(...)` | `visitIrRepeat` — **tensor_ops** `repeat`, not `.repeat` on `SparseTensor` |
| `get_default_stop(...)` | `visitIrGetDefaultStop` |
| `get_max_priority(...)` | `visitIrGetPriorityLList` |
| `filter_trav_exp_stop(...)`, `filter_trav_exp_not_stop(...)` | `visitIrGetPolyexpStop`, `visitIrGetPolyexpNotStop` |
| `convert_to_tensor(...)` | `visitIrConvertToTensor` |

Under reuse, **any** of these still present in the IR tree after TTB is a **latent `NameError`** unless another import defines the name.

**Symbols that look like `tensor_ops` but are satisfied without `tensor_ops` in reuse:**

- `convert_to_float` — defined in generated `transformers.py` (reuse branch).
- `binary_to_identity_unary`, `sp_where_block` — pulled in via `symexp` / `sparse_block` / `op_helper`, not `tensor_ops`.

---

## 4. Spot-check: [`output/transformers.py`](../output/transformers.py) (sample)

Checks on this checkout’s generated file:

| Check | Result |
|-------|--------|
| `from constraintflow.gbcsr.tensor_ops` | **Absent** (consistent with reuse header). |
| `SparseTensor(`, `get_sub_block_custom_range`, `.squeeze` | **Present** (matches codegen inventory). |
| `.binary(` on block temporaries | **Present** — these are **`SparseBlock.binary`**, not `tensor_ops.binary`. |
| Stray `tensor_ops`-style **top-level** `unary(...` | **Present** (e.g. lines ~863–864: `unary(abs_elem.get_elem(...), 'sigma', layer_index=..., ...)`). That call is **undefined** in this file’s imports — a **TTB / codegen gap** relative to the reuse header. |

Re-run a full reuse compile and grep for `\bbinary(`, `\bunary(`, `\bwhere(`, `\bclamp(`, `\binner_prod(`, `get_default_stop`, etc., to audit your exact IR output.

---

## 5. Split: `sparse_tensor.py` — trace vs reuse-runtime vs lib / shared

Legend:

- **Trace hooks**: default `json_list` / `lhs_index`-style parameters and JSON append logic used to build JIT tapes under simulacrum.
- **Reuse-runtime**: still needed when executing **reuse** `transformers.py` as emitted today (constructor + layout helpers + typing).
- **Lib / shared**: used from [`constraintflow/lib/`](../constraintflow/lib/) or runtime paths that are not specific to trace recording.

### Module-level helpers (`sparse_tensor.py`)

| Names (representative) | Role |
|------------------------|------|
| `get_operator_func`, `tensor_to_list`, index/geometry helpers (`compare_index`, `union_block`, …) | **Shared / algorithmic** — used by `SparseTensor` internals and related code; not codegen emission by themselves. |
| `convert_dense_to_sparse`, `sp_tensor_from_overlap_classes`, `split_blocks`, … | **Trace / simulacrum-heavy** — support recording and block pairing; not reuse `transformers` surface. |
| `sparse_max`, `sparse_min`, `sp_where` | **Shared** entry points wrapping tensor-level ops (may participate in traces when used from simulacrum). |

### `SparseTensor` methods

| Method | Trace hooks? | Reuse-runtime (current codegen)? | Lib / shared |
|--------|--------------|-------------------------------------|--------------|
| `__init__` | No | **Yes** (`SparseTensor(...)`) | **Yes** — widespread construction |
| `plot_3d`, `__str__` | No | No | Debug / printing |
| `expand_symexp_mat` | No | No | **Yes** (`symexp` / codegen `visitIrExpandSymExp`) |
| `get_dense`, `get_dense_custom_range` | No | Uncommon in emitted transformers | **Yes** (`spec`, `polyexp`, `flow_sparse`, `llist`) |
| `get_sub_block_custom_range`, `get_sub_block_fast` | No | **Yes** (custom range in reuse) | Internal / tooling |
| `exists_block`, `exists_sub_block`, `get_block_id` | No | No | Layout queries |
| `get_sparse_custom_range`, `reduce_size`, `increase_size` | No | No | **Heavy layout** — simulacrum / non-TTB paths |
| `merge_no_overlap`, `union_tensors`, `copy` | No | No | **Heavy layout** / full interpreter |
| `check_dense` | No | No | Validation |
| `unary` | **Yes** (`json_list`) | **Possible** via `visitIrPolyExpNotStopFloat` (`.unary(...)`) | **Yes** when running full stack / `flow_sparse` |
| `binary`, `matmul` | **Yes** | Only if IR still has tensor-level ops (normally TTB lowers to blocks) | **Yes** — e.g. `flow_sparse` `.binary` |
| `any` | No | Lowered to block `.any()` in typical TTB output | Instance method exists on tensor |
| `float` | No | **Yes** (via `convert_to_float`) | Typing |
| `add_block_no_overlap`, `overwrite_block`, `overwrite_from_index`, `overwrite` | No | No | **Heavy layout** |
| `squeeze`, `unsqueeze`, `repeat` | **`repeat` has trace hooks** | **Yes** (`squeeze` / `unsqueeze` / `repeat` chains) | **Yes** (`abs_elem`, `llist`, etc.) |
| `clamp` | **Yes** | Lowered to block `.clamp` when TTB applies | Full-path clamp on tensor |
| `sum` | No | Possible if `visitIrReduce` targets a type with `.sum` | Reductions |

**Static nested helpers** (`lex_cmp`, `dfs`): internal to `SparseTensor`; **not** reuse API.

---

## 6. Summary

- Reuse `transformers.py` **depends on** `SparseTensor` for construction, **layout** (`get_sub_block_custom_range`, `squeeze`, `unsqueeze`, `repeat`), **typing** (`float` / `convert_to_float`), and occasionally **instance** `.unary` for boolean normalization.
- Reuse mode **omits** `tensor_ops`; TTB must remove every `binary`/`unary`/`where`/… **top-level** call, or imports must be restored. The checked sample still contains **`unary(..., 'sigma', ...)`**, which is inconsistent with the reuse header and should be tracked as a coverage or ordering bug.
- **Trace-only** aspect is not a separate class of methods so much as **optional `json_list` recording** on the heavy ops (`unary`, `binary`, `matmul`, `clamp`, `repeat`) plus simulacrum-only module helpers; **reuse-runtime** is the slimmer slice codegen actually emits; **lib/shared** is everything domains still call when modeling state as `SparseTensor`.
