# Consolidated sparse matmul experiments

Merged from `final_results/results*.json`. Duplicates resolved by preferring **expected batch size**, then **ok** status, then rows with metrics.

**Expected batch sizes:** n10/n11/n14/n22 → 100; n30/n32 → 10.

## Multiply time vs block sparse (ratio)

| Program | Network | Batch | Match | Status | Block (s) | CSR mul× | COO mul× | BSR 8×8 mul× | BSR dyn mul× |
|---------|---------|------:|:-----:|--------|----------:|---------:|---------:|-------------:|-------------:|
| ibp | n10 | 100 | ✓ | ok | 2.16 | 1.65 | 2.55 | 31.4 | 21.3 |
| ibp | n11 | 100 | ✓ | ok | 0.23 | 30 | 41 | 489 | 383 |
| ibp | n14 | 100 | ✓ | ok | 0.25 | 4.07 | 4.94 | 193 | 154 |
| ibp | n22 | 100 | ✓ | ok | 1.05 | 5.11 | 7.19 | 108 | 70.7 |
| ibp | n30 | 10 | ✓ | ok | 0.0429 | 14.7 | 24.1 | 347 | 824 |
| ibp | n32 | 10 | ✓ | failed | — | — | — | — | — |
| deeppoly | n10 | 10 | **≠** | ok | 1.43 | 2.6 | 12.1 | 171 | 190 |
| deeppoly | n11 | 10 | **≠** | ok | 0.453 | 10.7 | 98.1 | 558 | 557 |
| deeppoly | n14 | 10 | **≠** | ok | 0.184 | 2.79 | 5.74 | 277 | 0 |
| deeppoly | n22 | 10 | **≠** | ok | 2.35 | 4.06 | 26.2 | 235 | 0 |
| deeppoly | n30 | 10 | ✓ | ok | 0.946 | 3.95 | 28.6 | 390 | 0 |
| deeppoly | n32 | 10 | ✓ | failed | — | — | — | — | — |
| zono | n10 | 100 | ✓ | ok | 2.71 | 2.41 | 18.4 | 89.7 | 118 |
| zono | n11 | 100 | ✓ | ok | 0.941 | 23.9 | 250 | 730 | 865 |
| zono | n14 | 100 | ✓ | ok | 0.762 | 0.903 | 2.39 | 54.2 | 141 |
| zono | n22 | 100 | ✓ | ok | 8.54 | 3.34 | 28.2 | 115 | 152 |
| zono | n30 | 10 | ✓ | ok | 1.16 | 0.995 | 7.91 | 53.8 | 299 |
| zono | n32 | 10 | ✓ | ok | 36.2 | 0.0784 | 0.44 | 9.11 | 0 |
| crownibp | n10 | 10 | **≠** | ok | 0.215 | 3.26 | 4 | 202 | 25.1 |
| crownibp | n11 | 10 | **≠** | ok | 0.144 | 7.4 | 11.9 | 303 | 61.9 |
| crownibp | n14 | 10 | **≠** | ok | 0.0934 | 2.48 | 2.55 | 236 | 0 |
| crownibp | n22 | 10 | **≠** | ok | 0.223 | 5.53 | 6.03 | 320 | 0 |
| crownibp | n30 | 10 | ✓ | ok | 0.152 | 5.89 | 10.3 | 613 | 0 |
| crownibp | n32 | 10 | ✓ | failed | — | — | — | — | — |
| zid | n10 | 100 | ✓ | ok | 0.8 | 15.8 | 32.2 | 754 | 174 |
| zid | n11 | 100 | ✓ | ok | 0.583 | 36.1 | 223 | 1.3e+03 | 605 |
| zid | n14 | 100 | ✓ | ok | 0.646 | 5.33 | 6.79 | 553 | 0 |
| zid | n22 | 100 | ✓ | timeout | — | — | — | — | — |
| zid | n30 | 10 | ✓ | ok | 0.646 | 2.22 | 4.09 | 204 | 448 |
| zid | n32 | 10 | ✓ | timeout | — | — | — | — | — |
| reuse | n10 | 100 | ✓ | ok | 1.45 | 10.5 | 47.3 | 561 | 411 |
| reuse | n11 | 10 | **≠** | ok | 0.167 | 15.7 | 154 | 789 | 0 |
| reuse | n14 | 10 | **≠** | ok | 0.247 | 0.837 | 1.45 | 50.5 | 0 |
| reuse | n22 | 10 | **≠** | ok | 1.06 | 5.63 | 30 | 331 | 0 |
| reuse | n30 | 10 | ✓ | ok | 1.79 | 1.29 | 8.55 | 98.5 | 0 |
| reuse | n32 | 10 | ✓ | failed | — | — | — | — | — |
| polyzono | n10 | 10 | **≠** | ok | 1.55 | 2.46 | 12.8 | 164 | 0 |
| polyzono | n11 | 10 | **≠** | ok | 0.407 | 12.2 | 139 | 706 | 0 |
| polyzono | n14 | 10 | **≠** | ok | 0.273 | 2.06 | 4.54 | 202 | 0 |
| polyzono | n22 | 10 | **≠** | ok | 2.71 | 3.82 | 27.7 | 223 | 0 |
| polyzono | n30 | 10 | ✓ | ok | 1.43 | 3.19 | 23.2 | 291 | 0 |
| polyzono | n32 | 10 | ✓ | timeout | — | — | — | — | — |
| deeppoly_efficient | n10 | 10 | **≠** | ok | 1.25 | 2.82 | 13.8 | 184 | 0 |
| deeppoly_efficient | n11 | 10 | **≠** | ok | 0.376 | 11.6 | 117 | 701 | 0 |
| deeppoly_efficient | n14 | 10 | **≠** | ok | 0.512 | 0.753 | 1.87 | 80.6 | 0 |
| deeppoly_efficient | n22 | 10 | **≠** | ok | 2.13 | 4.08 | 28.5 | 264 | 0 |
| deeppoly_efficient | n30 | 10 | ✓ | ok | 3.84 | 0.894 | 7.08 | 92.3 | 0 |
| deeppoly_efficient | n32 | 10 | ✓ | failed | — | — | — | — | — |

## Batch size ≠ expected (included in table above)

- **deeppoly.cf × n10**: ran batch=10, expected 100
- **deeppoly.cf × n11**: ran batch=10, expected 100
- **deeppoly.cf × n14**: ran batch=10, expected 100
- **deeppoly.cf × n22**: ran batch=10, expected 100
- **crownibp.cf × n10**: ran batch=10, expected 100
- **crownibp.cf × n11**: ran batch=10, expected 100
- **crownibp.cf × n14**: ran batch=10, expected 100
- **crownibp.cf × n22**: ran batch=10, expected 100
- **reuse.cf × n11**: ran batch=10, expected 100
- **reuse.cf × n14**: ran batch=10, expected 100
- **reuse.cf × n22**: ran batch=10, expected 100
- **polyzono.cf × n10**: ran batch=10, expected 100
- **polyzono.cf × n11**: ran batch=10, expected 100
- **polyzono.cf × n14**: ran batch=10, expected 100
- **polyzono.cf × n22**: ran batch=10, expected 100
- **deeppoly_efficient.cf × n10**: ran batch=10, expected 100
- **deeppoly_efficient.cf × n11**: ran batch=10, expected 100
- **deeppoly_efficient.cf × n14**: ran batch=10, expected 100
- **deeppoly_efficient.cf × n22**: ran batch=10, expected 100

## Dedup notes

- **deeppoly.cf × n14**: 2 entries [('results2.json', 10, 'failed'), ('results2.json', 10, 'ok')]; kept results2.json batch=10 ok (no run with expected batch=100)
- **crownibp.cf × n14**: 2 entries [('results2.json', 10, 'failed'), ('results2.json', 10, 'ok')]; kept results2.json batch=10 ok (no run with expected batch=100)
- **crownibp.cf × n22**: 2 entries [('results2.json', 10, 'failed'), ('results2.json', 10, 'ok')]; kept results2.json batch=10 ok (no run with expected batch=100)
- **zid.cf × n10**: 2 entries [('results1.json', 100, 'ok'), ('results4.json', 100, 'ok')]; kept results1.json batch=100 ok
- **zid.cf × n14**: 2 entries [('results1.json', 100, 'failed'), ('results1.json', 100, 'ok')]; kept results1.json batch=100 ok
- **zid.cf × n22**: 2 entries [('results1.json', 100, 'failed'), ('results1.json', 100, 'timeout')]; kept results1.json batch=100 timeout
- **reuse.cf × n10**: 2 entries [('results2.json', 10, 'ok'), ('results3.json', 100, 'ok')]; kept results3.json batch=100 ok

## Failed / timeout

- ibp.cf × n32: failed
- deeppoly.cf × n32: failed
- crownibp.cf × n32: failed
- zid.cf × n22: timeout
- zid.cf × n32: timeout
- reuse.cf × n32: failed
- polyzono.cf × n32: timeout
- deeppoly_efficient.cf × n32: failed
