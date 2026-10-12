# SymboLM-1.5B-v1.0-DSR1-Qwen2.5 — 1,000-Question Multi-Year Benchmark Report
**Author & Principal Architect:** Sanish Gyawali (Rupandehi, Nepal)
**Hardware:** Kaggle NVIDIA Tesla P100 GPU (HBM2 High-Bandwidth Memory)
**Total Problems Evaluated:** 1,000 across 5 consecutive years (2021–2026)
**Overall Pass@1 Accuracy:** 67.60% (676/1000)
**Overall Token Compression:** 50.65% (125,258 vs 253,800 English baseline)
**Evaluation Duration:** 4767.83s (79.46 min)

## Temporal Generalization Breakdown (Per Year)

| Year | Sample Count | Pass@1 Accuracy | Avg Generated Tokens | Avg Baseline CoT | Token Compression |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2021** | 200 | **72.0%** (144/200) | 114.3 tok | 198.0 tok | **42.3%** |
| **2022** | 200 | **71.0%** (142/200) | 128.0 tok | 214.5 tok | **40.3%** |
| **2023** | 200 | **84.5%** (169/200) | 128.0 tok | 264.5 tok | **51.6%** |
| **2024** | 200 | **72.5%** (145/200) | 128.0 tok | 283.0 tok | **54.8%** |
| **2026** | 200 | **38.0%** (76/200) | 128.0 tok | 309.0 tok | **58.6%** |

## Routing Distribution Across Registers

- **Year 2021:** {'plan+deduce': 124, 'unknown': 35, 'deduce': 1, 'bypass': 40}
- **Year 2022:** {'plan+deduce': 156, 'unknown': 44}
- **Year 2023:** {'plan+deduce': 200}
- **Year 2024:** {'plan+deduce': 200}
- **Year 2026:** {'plan+deduce': 160, 'unknown': 40}