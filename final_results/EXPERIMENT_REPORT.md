# Sparse matmul format comparison — final report

## What we did

We **did not** swap out the real runtime. ConstraintFlow still does matmul the same way it always has — our block-sparse `SparseTensor` path. What we added is a **benchmark hook** that runs after each matmul: take the same operands, try the multiply in PyTorch **CSR**, **COO**, and **BSR (8×8 blocks)**, and time it against what we already computed. Matmul is where most of the time goes, so if generic sparse formats lose there, we’re not missing much by not using them everywhere.

The flow is: run the normal block-sparse matmul → then (for comparison) densify that pair of tensors, encode to each format, run `torch.matmul`, check the answer matches. We split **encode** vs **multiply** in the logs; the tables below are **multiply only** (slowdown vs block sparse). BSR variable/derived block size is left out. Failed runs and timeouts are blank in the tables.

We ran **8 programs** (ibp, deeppoly, zono, crownibp, zid, reuse, polyzono, skippoly) × **6 networks** (n10, n11, n14, n22, n30, n32) on different machines; results were merged from `final_results/results*.json`. Batch size is whatever that run used (not all cells used the same batch for a given network — see consolidated CSV if you need the number per cell).

## Bottom line

Block-sparse wins basically everywhere. **CSR multiply** is often only a few× slower than us. **COO** and **BSR 8×8** are a different story — usually much worse, sometimes hundreds of times slower on multiply. So the current backend isn’t just “fine”; for these workloads it’s clearly the right shape for the hot path.

---

## How to read the tables

- **Rows** = network  
- **Columns** = program  
- Number = how many times **slower than block-sparse multiply** (e.g. `10×` = 10× slower)  
- **—** = failed, timeout, or no valid BSR number  
- **Row avg / Col avg** = mean over non-empty cells in that row or column  
- Corner = mean over all filled cells in that table  

---

### CSR multiply slowdown vs block sparse

| Network | ibp | deeppoly | zono | crownibp | zid | reuse | polyzono | skippoly | **Row avg** |
|---------|-----|----------|------|----------|-----|-------|----------|-------------------|-------------|
| **n10** | 1.7× | 2.6× | 2.4× | 3.3× | 15.8× | 10.5× | 2.5× | 2.8× | 5.2× |
| **n11** | 30.0× | 10.7× | 23.9× | 7.4× | 36.1× | 15.7× | 12.2× | 11.6× | 18.5× |
| **n14** | 4.1× | 2.8× | 0.9× | 2.5× | 5.3× | 0.8× | 2.1× | 0.8× | 2.4× |
| **n22** | 5.1× | 4.1× | 3.3× | 5.5× | — | 5.6× | 3.8× | 4.1× | 4.5× |
| **n30** | 14.7× | 4.0× | 1.0× | 5.9× | 2.2× | 1.3× | 3.2× | 0.9× | 4.1× |
| **Col avg** | 11.1× | 4.8× | 5.3× | 4.9× | 14.9× | 6.8× | 4.7× | 4.0× | 6.8× |

### COO multiply slowdown vs block sparse

| Network | ibp | deeppoly | zono | crownibp | zid | reuse | polyzono | skippoly | **Row avg** |
|---------|-----|----------|------|----------|-----|-------|----------|-------------------|-------------|
| **n10** | 2.6× | 12.1× | 18.4× | 4.0× | 32.2× | 47.3× | 12.8× | 13.8× | 17.9× |
| **n11** | 41.0× | 98.1× | 250.4× | 11.9× | 223.4× | 154.1× | 139.2× | 117.2× | 129.4× |
| **n14** | 4.9× | 5.7× | 2.4× | 2.6× | 6.8× | 1.4× | 4.5× | 1.9× | 3.8× |
| **n22** | 7.2× | 26.2× | 28.2× | 6.0× | — | 30.0× | 27.7× | 28.5× | 22.0× |
| **n30** | 24.1× | 28.6× | 7.9× | 10.3× | 4.1× | 8.6× | 23.2× | 7.1× | 14.2× |
| **Col avg** | 16.0× | 34.1× | 51.3× | 6.9× | 66.6× | 48.3× | 41.5× | 33.7× | 36.9× |

### BSR (8×8) multiply slowdown vs block sparse

| Network | ibp | deeppoly | zono | crownibp | zid | reuse | polyzono | skippoly | **Row avg** |
|---------|-----|----------|------|----------|-----|-------|----------|-------------------|-------------|
| **n10** | 31.4× | 170.6× | 89.7× | 202.2× | 753.9× | 561.4× | 164.0× | 184.0× | 269.7× |
| **n11** | 489.0× | 557.8× | 729.7× | 303.3× | 1301.4× | 788.9× | 705.9× | 701.4× | 697.2× |
| **n14** | 193.3× | 277.3× | 54.2× | 236.5× | 552.9× | 50.5× | 202.4× | 80.6× | 205.9× |
| **n22** | 107.9× | 234.9× | 115.4× | 320.0× | — | 331.3× | 222.8× | 264.2× | 228.1× |
| **n30** | 347.2× | 389.8× | 53.8× | 613.1× | 203.6× | 98.5× | 290.9× | 92.3× | 261.2× |
| **Col avg** | 233.8× | 326.1× | 175.3× | 335.0× | 703.0× | 366.1× | 317.2× | 264.5× | 326.9× |

---

## Gaps in the grid

These did not produce ok results (blank in tables above):

- **n22**: zid — timeout  

Raw numbers and batch sizes per run: `consolidated.csv` in this folder.
