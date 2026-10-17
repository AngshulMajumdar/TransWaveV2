# Benchmark protocol and interpretation

## Final stored causal TransWave runs

The four JSON files under `benchmarks/final_causal_results/` are the authoritative stored causal records for arXiv, FineWeb-Edu, PG19 and WikiText-103.

Each record reports exact causal next-byte NLL, byte perplexity `exp(NLL)`, bits per byte `NLL / ln(2)`, `future_leakage: false`, parameter count 96,130,040, raw context 32,768 bytes, validation history and final validation.

These are the values used for the four GitHub result figures.

## Historical Mamba-2 / Hyena / RetNet table

The Mamba-2, Hyena and RetNet rows are retained because they are the recorded historical comparison in the project. They **do not use the same objective/tokenization protocol** as the final causal TransWave runs: those rows came from an earlier tokenized 15% masked-language-model protocol.

Therefore the stored numerical table may be reported, and it is correct to say the recorded TransWave value is numerically lower in every row, but it is **not** an objective-identical causal head-to-head.

A controlled comparison requires retraining the baselines under the same raw-byte causal objective, context, data split and evaluation unit.

## 4.5B release-scale run

`release/result_4p5b.json` is a separate experiment: one release-scale model trained for exactly 4,499,996,672 raw-byte tokens.

It finished successfully and has its own held-out validation. Its values are not substituted into the four earlier per-corpus benchmark records.
