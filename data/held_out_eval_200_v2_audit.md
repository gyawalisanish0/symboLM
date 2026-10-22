# Held-Out Benchmark v2 Audit

Generated: `2026-10-22T07:55:34.354093+00:00`

- Rows: **200**
- Unique prompts: **173**
- Duplicate groups: **20** (27 redundant rows)
- Ground-truth overrides: **1**

## Ground-truth overrides

| ID | Original | Corrected | Reason |
|---|---|---|---|
| `MY2021_001` | `169` | `169.5` | 123 * 1.5 = 184.5; 184.5 - 15 = 169.5. The original benchmark value 169 silently truncates the half-dollar. |

## Duplicate prompt groups

Identical prompts in the benchmark are not independent samples. The v2 file flags them with `duplicate_prompt`, `duplicate_group_size`, and `duplicate_group_ids`; use a cluster-aware or unique-prompt sensitivity analysis alongside the primary n=200 analysis.

| Group size | IDs | Prompt |
|---:|---|---|
| 4 | `MY2026_143`, `MY2026_147`, `MY2026_148`, `MY2026_151` | The Lucas sequence satisfies L(0) = 2, L(1) = 1, and L(n) = L(n-1) + L(n-2) for n >= 2. What is the value of L(4)? |
| 3 | `MY2023_113`, `MY2023_114`, `MY2023_115` | Find the number of positive integers less than or equal to 150 that are relatively prime to 150. |
| 3 | `MY2023_061`, `MY2023_063`, `MY2023_071` | In how many ways can 4 distinct letters be placed into 4 distinctly addressed envelopes such that no letter is placed in its correct envelope? |
| 3 | `MY2023_064`, `MY2023_065`, `MY2023_069` | In how many ways can 5 distinct letters be placed into 5 distinctly addressed envelopes such that no letter is placed in its correct envelope? |
| 3 | `MY2026_062`, `MY2026_064`, `MY2026_069` | Pipe A fills a reservoir in 12 hours, Pipe B fills it in 15 hours, and Pipe C fills it in 20 hours. How many hours will it take to fill the reservoir if all three pipes operate sim |
| 3 | `MY2026_066`, `MY2026_067`, `MY2026_070` | Pipe A fills a reservoir in 20 hours, Pipe B fills it in 30 hours, and Pipe C fills it in 60 hours. How many hours will it take to fill the reservoir if all three pipes operate sim |
| 2 | `MY2023_164`, `MY2023_166` | Evaluate the binomial coefficient C(18, 4). |
| 2 | `MY2023_169`, `MY2023_171` | Evaluate the binomial coefficient C(7, 5). |
| 2 | `MY2023_112`, `MY2023_116` | Find the number of positive integers less than or equal to 36 that are relatively prime to 36. |
| 2 | `MY2024_061`, `MY2024_066` | Find the positive integer solution (x, y) with x > y > 0 to the equation x^2 - y^2 = 15. What is the value of x + y? |
| 2 | `MY2023_009`, `MY2023_012` | If x + 1/x = 13, find the exact value of x^4 + 1/x^4. |
| 2 | `MY2023_002`, `MY2023_014` | If x + 1/x = 14, find the exact value of x^3 + 1/x^3. |
| 2 | `MY2023_066`, `MY2023_068` | In how many ways can 3 distinct letters be placed into 3 distinctly addressed envelopes such that no letter is placed in its correct envelope? |
| 2 | `MY2023_062`, `MY2023_067` | In how many ways can 7 distinct letters be placed into 7 distinctly addressed envelopes such that no letter is placed in its correct envelope? |
| 2 | `MY2026_061`, `MY2026_068` | Pipe A fills a reservoir in 10 hours, Pipe B fills it in 15 hours, and Pipe C fills it in 30 hours. How many hours will it take to fill the reservoir if all three pipes operate sim |
| 2 | `MY2026_063`, `MY2026_065` | Pipe A fills a reservoir in 6 hours, Pipe B fills it in 10 hours, and Pipe C fills it in 15 hours. How many hours will it take to fill the reservoir if all three pipes operate simu |
| 2 | `MY2026_144`, `MY2026_146` | The Lucas sequence satisfies L(0) = 2, L(1) = 1, and L(n) = L(n-1) + L(n-2) for n >= 2. What is the value of L(10)? |
| 2 | `MY2026_142`, `MY2026_149` | The Lucas sequence satisfies L(0) = 2, L(1) = 1, and L(n) = L(n-1) + L(n-2) for n >= 2. What is the value of L(5)? |
| 2 | `MY2026_141`, `MY2026_145` | The Lucas sequence satisfies L(0) = 2, L(1) = 1, and L(n) = L(n-1) + L(n-2) for n >= 2. What is the value of L(8)? |
| 2 | `MY2024_004`, `MY2024_009` | What is the exponent of the highest power of 5 that divides 180! ? |

## Output files

- `C:\Users\user\Dev\symboLM\data\held_out_eval_200_v2.jsonl`
- `C:\Users\user\Dev\symboLM\data\held_out_eval_200_v2_manifest.json`

