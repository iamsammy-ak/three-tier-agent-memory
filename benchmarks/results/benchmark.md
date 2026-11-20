# Benchmark results

Synthetic data: 50 users × 10 sessions (7568 turns), 450 probe questions {'changed': 174, 'stable': 226, 'event': 50}, seed 7, embedder `fastembed:BAAI/bge-small-en-v1.5`.
Latency is retrieval only (best of 3 per question); embedding a question takes a median 23.559 ms on top for every system except recency-window.

Reproduce: `python -m tiered_memory bench --users 50 --sessions 10 --seed 7 --embedder fastembed --k 5 --budget 200 --growth-users 10 --growth-sessions 40`

| System | Recall@5 | Stale@5 | Recall@200 tok | Stale@200 tok | Recall changed / stable / event | p50 ms | p95 ms | Items/user | KB/user |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| recency-window | 0.042 | 0.000 | 0.227 | 0.011 | 0.08 / 0.02 / 0.02 | 0.07 | 0.09 | 151.4 | 4.4 |
| flat-vector | 0.733 | 0.776 | 0.849 | 0.851 | 0.61 / 0.87 / 0.52 | 0.20 | 0.27 | 151.4 | 231.5 |
| tiered | 0.804 | 0.224 | 0.878 | 0.305 | 0.84 / 0.83 / 0.58 | 1.65 | 4.46 | 19.5 | 79.3 |
| tiered-no-supersede | 0.778 | 0.805 | 0.878 | 0.885 | 0.82 / 0.81 / 0.50 | 1.46 | 2.62 | 19.5 | 79.3 |
| tiered-no-eviction | 0.804 | 0.241 | 0.878 | 0.316 | 0.84 / 0.83 / 0.58 | 1.52 | 2.24 | 19.7 | 81.3 |

Stale rates are computed over probes whose fact changed at least once.

### Stale@5 split by extraction coverage

Changed facts whose every statement used a phrasing the rule-based extractor covers (96 probes) versus facts with at least one statement it cannot parse (78 probes).

| System | all phrasings covered | some phrasings uncovered |
|---|---:|---:|
| recency-window | 0.000 | 0.000 |
| flat-vector | 0.802 | 0.744 |
| tiered | 0.000 | 0.500 |
| tiered-no-supersede | 0.958 | 0.615 |
| tiered-no-eviction | 0.000 | 0.538 |

## Rule-based fact extraction against ground truth

- Mentions: 629; recall 0.773 (covered phrasings 1.000, uncovered phrasings 0.000); precision 1.000 over 486 predictions.

## Consolidation (tiered system)

- Episodes created: 500; they keep 32.4% of the raw tokens (18315 of 56609).
- Facts added 339, superseded 147, reinforced or merged 0.

## Store growth (mean per user after each session)

| Session | recency-window | flat-vector | tiered | tiered-no-supersede | tiered-no-eviction |
|---:|---:|---:|---:|---:|---:|
| 1 | 20.9 items / 0.6 KB | 20.9 items / 31.9 KB | 4.5 items / 15.3 KB | 4.5 items / 15.3 KB | 4.5 items / 15.3 KB |
| 2 | 39.5 items / 1.1 KB | 39.5 items / 60.4 KB | 7.8 items / 27.5 KB | 7.8 items / 27.5 KB | 7.8 items / 27.5 KB |
| 3 | 54.6 items / 1.6 KB | 54.6 items / 83.5 KB | 9.7 items / 35.5 KB | 9.7 items / 35.5 KB | 9.7 items / 35.5 KB |
| 4 | 68.7 items / 2.0 KB | 68.7 items / 105.1 KB | 11.0 items / 42.0 KB | 11.0 items / 42.0 KB | 11.0 items / 42.0 KB |
| 5 | 82.7 items / 2.4 KB | 82.7 items / 126.5 KB | 12.4 items / 48.6 KB | 12.4 items / 48.6 KB | 12.4 items / 48.6 KB |
| 6 | 96.8 items / 2.8 KB | 96.8 items / 148.0 KB | 13.9 items / 55.4 KB | 13.9 items / 55.4 KB | 13.9 items / 55.4 KB |
| 7 | 110.2 items / 3.2 KB | 110.2 items / 168.6 KB | 15.5 items / 61.9 KB | 15.5 items / 61.9 KB | 15.5 items / 61.9 KB |
| 8 | 124.4 items / 3.6 KB | 124.4 items / 190.2 KB | 17.1 items / 69.0 KB | 17.1 items / 69.0 KB | 17.1 items / 69.0 KB |
| 9 | 137.3 items / 4.0 KB | 137.3 items / 210.0 KB | 18.3 items / 74.7 KB | 18.3 items / 74.7 KB | 18.3 items / 74.8 KB |
| 10 | 151.4 items / 4.4 KB | 151.4 items / 231.5 KB | 19.5 items / 79.7 KB | 19.5 items / 79.7 KB | 19.7 items / 81.3 KB |

## Long-horizon store growth (10 users × 40 sessions, seed 8)

| Session | flat-vector | tiered-no-eviction | tiered |
|---:|---:|---:|---:|
| 1 | 21.2 items / 32.4 KB | 3.8 items / 13.8 KB | 3.8 items / 13.8 KB |
| 6 | 94.0 items / 143.7 KB | 12.5 items / 51.1 KB | 12.5 items / 51.1 KB |
| 11 | 158.4 items / 242.2 KB | 17.8 items / 78.9 KB | 16.9 items / 72.4 KB |
| 16 | 224.4 items / 343.1 KB | 23.0 items / 107.4 KB | 16.5 items / 65.4 KB |
| 21 | 285.6 items / 436.7 KB | 28.5 items / 134.3 KB | 16.5 items / 62.6 KB |
| 26 | 348.6 items / 533.0 KB | 34.0 items / 161.7 KB | 16.4 items / 60.8 KB |
| 31 | 413.4 items / 632.1 KB | 39.5 items / 190.0 KB | 17.2 items / 65.8 KB |
| 36 | 481.6 items / 736.4 KB | 44.9 items / 218.3 KB | 17.4 items / 68.0 KB |
| 40 | 535.6 items / 818.9 KB | 49.4 items / 242.4 KB | 17.5 items / 70.7 KB |
